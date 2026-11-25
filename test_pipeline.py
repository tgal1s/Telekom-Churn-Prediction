"""Checks for timestamp boundaries, leakage and metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import DAY, KEY, label_snapshot, timestamps
from main import split_dates
from model import evaluate, rank
from spark_features import TICKETS, read_source, spark_session, window_sums


class PipelineTests(unittest.TestCase):
    cutoff = pd.Timestamp("2025-11-01", tz="UTC")

    def test_target_interval_and_active_cohort(self):
        index = pd.Index(list("abcde"), name=KEY)
        features = pd.DataFrame({"usage": range(5)}, index=index)
        events = pd.Series([self.cutoff - DAY, self.cutoff, self.cutoff + 29 * DAY,
                            self.cutoff + 30 * DAY, pd.NaT], index=index)
        result = label_snapshot(features, events, self.cutoff, 30 * DAY)
        self.assertEqual(result[KEY].tolist(), list("bcde"))
        self.assertEqual(result["target"].tolist(), [1, 1, 0, 0])

    def test_ambiguous_iso_and_dotted_dates_agree(self):
        dates = timestamps(pd.Series(["2025-11-01 08:00:00", "01.11.2025 08:00:00"]))
        expected = self.cutoff + pd.Timedelta(hours=8)
        self.assertEqual(dates.tolist(), [expected, expected])

    def test_split_rejects_overlapping_outcomes(self):
        config = {"train_offsets": [14, 21], "test_offsets": [50, 60],
                  "horizon_days": 30, "window_days": 7}
        with self.assertRaisesRegex(ValueError, "Train labels"):
            split_dates(config, self.cutoff, self.cutoff + 90 * DAY)

    def test_ranking_metrics_for_tied_probabilities(self):
        data = pd.DataFrame({KEY: list("abcd"), "as_of": ["test"] * 4,
                             "probability": [0.25] * 4, "target": [0, 1, 0, 0]})
        scores = rank(data, 0.25)
        metrics = evaluate(scores, {"threshold": 0.2})
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["average_precision"], 0.25)
        self.assertTrue(np.isclose(metrics["brier"], 0.1875))
        self.assertEqual(scores["in_top_group"].sum(), 1)


class SparkTests(unittest.TestCase):
    cutoff = pd.Timestamp("2025-11-01", tz="UTC")

    @classmethod
    def setUpClass(cls):
        cls.spark = spark_session({"spark": {"master": "local[2]", "settings": {
            "spark.sql.shuffle.partitions": "2", "spark.ui.enabled": "false",
        }}})

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_cdr_arriving_at_cutoff_is_excluded(self):
        day = (self.cutoff - DAY).date()
        earlier = (self.cutoff - pd.Timedelta(seconds=1)).to_pydatetime()
        events = self.spark.createDataFrame([
            (day, "a", earlier, 1, 3, 7, 1),
            (day, "a", self.cutoff.to_pydatetime(), 1, 20, 99, 0),
        ], ["_day", KEY, "_ingest", "cdr", "duration", "traffic", "voice"])
        windows = self.spark.createDataFrame([(self.cutoff.date(), day, "recent")],
                                             "_as_of date, _day date, _window string")
        result = window_sums(events, windows, "cdr").first()
        self.assertEqual((result._cdr_recent, result._duration_recent, result._traffic_recent,
                          result._voice_recent), (1, 3, 7, 1))

    def test_ticket_reader_deduplicates_and_quarantines_invalid_utf8(self):
        day = self.cutoff.date()
        bad_day = (self.cutoff + DAY).date()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = root / "raw/tickets" / f"dt={day}" / "tickets.csv"
            bad = root / "raw/tickets" / f"dt={bad_day}" / "tickets.csv"
            good.parent.mkdir(parents=True)
            bad.parent.mkdir(parents=True)
            frame = pd.DataFrame({
                KEY: ["a", "a", "a", "a"], "ticket_id": [1, 1, 2, 3],
                "created_ts": ["01.11.2025 08:00:00", "2025-11-01 08:00:00",
                               "2025-11-02 00:00:00", "2025-11-01 09:00:00"],
                "category": [" QUALITY ", " QUALITY ", "billing", "billing"],
                "retention_contact": [True, False, True, True],
            })
            frame.to_csv(good, index=False)
            bad.write_bytes(good.read_bytes() + b"\xff\n")
            result, stats = read_source(self.spark, root, "tickets", [day, bad_day], 10 ** 9)
            self.addCleanup(result.unpersist)
            self.assertNotIn("retention_contact", TICKETS)
            self.assertNotIn("retention_contact", result.columns)
            self.assertEqual(stats[day]["valid_rows"], 2)
            self.assertEqual(stats[bad_day]["status"], "corrupt")
            self.assertEqual(sum(row.quality for row in result.collect()), 1)


if __name__ == "__main__":
    unittest.main()
