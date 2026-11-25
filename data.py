"""TDG source contracts and point-in-time subscriber features."""

import json
from pathlib import Path

import pandas as pd


KEY = "subscriber_key"
DAY = pd.Timedelta(days=1)


def load_cohort(path):
    cohort = pd.read_csv(path, usecols=[KEY, "activated_at"], dtype={KEY: "str"})
    cohort["activated_at"] = timestamps(cohort["activated_at"]).dt.normalize()
    if cohort.empty or cohort.isna().any().any() or cohort[KEY].duplicated().any():
        raise ValueError(f"Invalid subscriber cohort: {path}")
    return cohort.set_index(KEY).sort_index()


def load_data(config):
    root = Path(config["data_path"])
    reference = root / "reference"
    with (reference / "passport.json").open(encoding="utf-8") as file:
        passport = json.load(file)
    start = pd.Timestamp(passport["period_start"], tz="UTC")
    end = start + passport["period_days"] * DAY
    cohort = load_cohort(
        root / "raw/dictionaries" / f"subscribers_snapshot_{start:%Y-%m-%d}.csv"
    )
    truth = pd.read_csv(reference / "churn_truth.csv", usecols=[KEY, "churn_day_index"])
    if (truth[KEY].isna().any() or truth[KEY].duplicated().any()
            or not truth["churn_day_index"].isin(range(-1, passport["period_days"])).all()):
        raise ValueError("Invalid TDG churn labels")
    indexes = truth.set_index(KEY)["churn_day_index"].reindex(cohort.index)
    if indexes.isna().any():
        raise ValueError("Churn labels must cover the initial cohort")
    events = (start + pd.to_timedelta(indexes, unit="D")).where(indexes.ge(0))
    return start, end, cohort, events, passport


def timestamps(values):
    parsed = pd.to_datetime(values, format="ISO8601", errors="coerce", utc=True)
    fallback = parsed.isna() & values.notna()
    parsed.loc[fallback] = pd.to_datetime(
        values.loc[fallback], format="%d.%m.%Y %H:%M:%S", errors="coerce", utc=True
    )
    return parsed


def label_snapshot(features, events, cutoff, horizon):
    dates = events.reindex(features.index)
    eligible = dates.isna() | dates.ge(cutoff)
    dataset = features.loc[eligible].copy()
    dataset["target"] = dates.loc[eligible].lt(cutoff + horizon).astype("int8")
    dataset["as_of"] = cutoff.date().isoformat()
    return dataset.reset_index()
