"""Train and score a 30-day telecom churn model."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyspark

from data import DAY, KEY, label_snapshot, load_cohort, load_data
from model import evaluate, predict, rank, train_model
from spark_features import preprocess


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               sort_keys=True) + "\n", encoding="utf-8")


def load_config(path, args):
    config = read_json(path)
    for key, value in (("data_path", args.data), ("output_path", args.output)):
        config[key] = str(Path(value).resolve() if value else path.parent / config[key])
    if config["horizon_days"] != 30 or config["window_days"] < 1:
        raise ValueError("Expected a 30-day horizon and a positive feature window")
    for key in ("threshold", "top_fraction", "min_coverage"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    features = config["features"]
    if not features or len(set(features)) != len(features):
        raise ValueError("The feature list must be nonempty and unique")
    return config


def split_dates(config, start, end):
    train = sorted(start + offset * DAY for offset in set(config["train_offsets"]))
    test = sorted(start + offset * DAY for offset in set(config["test_offsets"]))
    horizon = config["horizon_days"] * DAY
    if not train or not test or train[-1] + horizon > test[0]:
        raise ValueError("Train labels must end by the first test snapshot")
    if train[0] - 2 * config["window_days"] * DAY < start or test[-1] + horizon > end:
        raise ValueError("The dataset does not cover all history and label windows. Use TDG S or longer.")
    return train, test


def prepare_scores(dataset, model):
    scores = dataset[[KEY, "as_of"]].copy()
    scores["probability"] = predict(dataset, model)
    scores["flagged"] = scores["probability"].ge(model["threshold"])
    if "target" in dataset:
        scores["target"] = dataset["target"]
    return rank(scores, model["top_fraction"])


def publish_training(output, model, metrics, profile, quality, scores):
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as pending:
        stage = Path(pending) / "run"
        stage.mkdir()
        write_json(stage / "model.json", model)
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "data_quality.json", quality)
        profile.to_csv(stage / "train_feature_profile.csv", index_label="statistic")
        scores.to_parquet(stage / "test_predictions.parquet", index=False, engine="pyarrow")
        digest = hashlib.sha256()
        for file in sorted(stage.iterdir()):
            digest.update(file.name.encode("utf-8"))
            with file.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        run = output / ("run-" + digest.hexdigest()[:16])
        try:
            stage.rename(run)
        except OSError:
            if not run.is_dir() or any(not (run / file.name).is_file() for file in stage.iterdir()):
                raise
        latest = Path(pending) / "latest.json"
        write_json(latest, {"run": run.name})
        os.replace(latest, output / "latest.json")
    return run


def train(config):
    start, end, cohort, dates, passport = load_data(config)
    train_days, test_days = split_dates(config, start, end)
    all_days = train_days + test_days
    snapshots, quality = preprocess(cohort, all_days, config)
    horizon = config["horizon_days"] * DAY
    frames = {day: label_snapshot(snapshots[day], dates, day, horizon) for day in all_days}
    training = pd.concat([frames[day] for day in train_days], ignore_index=True)
    testing = pd.concat([frames[day] for day in test_days], ignore_index=True)
    model = train_model(training, config)
    scores = prepare_scores(testing, model)
    metrics = {
        "dataset": {key: passport[key] for key in ("generator_version", "level", "seed")},
        "estimator": model["estimator"],
        "runtime": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "pyspark": pyspark.__version__, "xgboost": model["xgboost_version"],
        },
        "xgboost_params": model["xgboost_params"],
        "train_dates": [day.date().isoformat() for day in train_days],
        "train_rows": len(training),
        "test": evaluate(scores, model),
        "test_by_date": {day: evaluate(group, model) for day, group in scores.groupby("as_of")},
    }
    profile = training[config["features"]].describe(percentiles=[0.25, 0.5, 0.75, 0.99])
    run = publish_training(Path(config["output_path"]), model, metrics, profile, quality, scores)
    logging.info("Saved results to %s", run)
    logging.info("Test AP %.4f, ROC-AUC %.4f", metrics["test"]["average_precision"],
                 metrics["test"]["roc_auc"])


def score(config, args):
    output = Path(config["output_path"])
    path = args.model or (output / read_json(output / "latest.json")["run"] / "model.json")
    model = read_json(path)
    if model["format_version"] != 1 or model.get("estimator") != "xgboost":
        raise ValueError("Expected an XGBoost model in format version 1")
    cutoff = pd.Timestamp(args.as_of, tz="UTC")
    if cutoff != cutoff.normalize():
        raise ValueError("--as-of must be a date at UTC midnight")
    cohort_path = args.cohort or (
        Path(config["data_path"]) / "raw/dictionaries"
        / f"subscribers_snapshot_{cutoff:%Y-%m-%d}.csv"
    )
    cohort = load_cohort(cohort_path)
    scoring_config = {**config, **model["preprocess"], "window_days": model["window_days"]}
    snapshots, _ = preprocess(cohort, [cutoff], scoring_config)
    dataset = snapshots[cutoff].reset_index()
    dataset["as_of"] = cutoff.date().isoformat()
    scores = prepare_scores(dataset, model)
    scores["horizon_end_exclusive"] = (cutoff + model["horizon_days"] * DAY).date().isoformat()
    scores["model_version"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    out_dir = output / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as pending:
        name = f"score-{cutoff:%Y-%m-%d}.parquet"
        stage = Path(pending) / name
        scores.to_parquet(stage, index=False, engine="pyarrow")
        os.replace(stage, out_dir / name)
    logging.info("Saved %d subscriber scores to %s", len(scores), out_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    scoring = commands.add_parser("score")
    scoring.add_argument("--as-of", required=True)
    scoring.add_argument("--cohort", type=Path)
    scoring.add_argument("--model", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config.resolve(), args)
        if args.command == "train":
            train(config)
        else:
            score(config, args)
    except Exception:
        logging.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
