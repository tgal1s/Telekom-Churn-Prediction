"""XGBoost churn prediction and binary ranking metrics."""

import base64
import logging

import numpy as np
import pandas as pd
import xgboost as xgb


def train_model(dataset, config):
    features = config["features"]
    matrix = dataset[features].to_numpy()
    target = dataset["target"].to_numpy(dtype=float)
    if not 0 < target.mean() < 1:
        raise ValueError("Training requires at least one row of each class")
    params = config["xgboost_params"].copy()
    rounds = params.pop("num_boost_round")
    training = xgb.DMatrix(matrix, label=target, nthread=params["nthread"])
    booster = xgb.train(params, training, num_boost_round=rounds)
    logging.info("XGBoost fit completed with %d boosting rounds", rounds)
    return {
        "format_version": 1,
        "estimator": "xgboost",
        "xgboost_version": xgb.__version__,
        "xgboost_params": {**params, "num_boost_round": rounds},
        "booster_ubj_base64": base64.b64encode(booster.save_raw(raw_format="ubj")).decode("ascii"),
        "features": features,
        "threshold": config["threshold"],
        "window_days": config["window_days"],
        "horizon_days": config["horizon_days"],
        "top_fraction": config["top_fraction"],
        "preprocess": {key: config[key] for key in ("min_coverage", "max_bytes_per_event")},
    }


def predict(dataset, model):
    values = dataset[model["features"]].to_numpy()
    booster = xgb.Booster()
    booster.load_model(bytearray(base64.b64decode(model["booster_ubj_base64"])))
    params = model["xgboost_params"]
    booster.set_param({"nthread": params["nthread"], "device": params["device"]})
    return booster.inplace_predict(values).astype(float)


def rank(scores, fraction):
    ranked = scores.sort_values(
        ["as_of", "probability", "subscriber_key"], ascending=[True, False, True]
    ).copy()
    groups = ranked.groupby("as_of", sort=False)["probability"]
    ranked["priority_rank"] = groups.cumcount() + 1
    ranked["in_top_group"] = ranked["priority_rank"] <= np.ceil(groups.transform("size") * fraction)
    return ranked


def evaluate(scores, model):
    target = scores["target"].to_numpy()
    probability = scores["probability"].to_numpy()
    positives = int(target.sum())
    negatives = len(target) - positives
    thresholds = scores.groupby("probability", sort=True)["target"].agg(["sum", "size"]).iloc[::-1]
    tp = thresholds["sum"].cumsum().to_numpy()
    seen = thresholds["size"].cumsum().to_numpy()
    predicted = probability >= model["threshold"]
    correct = int(target[predicted].sum())
    top = scores["in_top_group"].to_numpy()
    prevalence = positives / len(target)
    return {
        "rows": len(target),
        "positives": positives,
        "prevalence": prevalence,
        "average_precision": float(np.dot(np.diff(tp, prepend=0), tp / seen) / positives)
        if positives else None,
        "roc_auc": float(np.trapezoid(np.r_[0, tp / positives], np.r_[0, (seen - tp) / negatives]))
        if positives and negatives else None,
        "brier": float(np.mean((target - probability) ** 2)),
        "threshold": model["threshold"],
        "flagged": int(predicted.sum()),
        "precision": correct / predicted.sum() if predicted.any() else 0.0,
        "recall": correct / positives if positives else None,
        "f1": 2 * correct / (int(predicted.sum()) + positives) if predicted.any() or positives else None,
        "precision_at_top": float(target[top].mean()),
        "recall_at_top": float(target[top].sum() / positives) if positives else None,
        "lift_at_top": float(target[top].mean() / prevalence) if positives else None,
    }
