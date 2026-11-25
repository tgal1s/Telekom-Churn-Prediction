"""Distributed preparation of point-in-time subscriber features in Spark SQL."""

import csv
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType

from data import DAY, KEY


CDR = ("record_id", KEY, "event_ts", "ingest_ts", "duration_sec",
       "bytes_up", "bytes_down", "service_type")
TICKETS = ("ticket_id", KEY, "created_ts", "category")
COUNTERS = {"cdr": ("cdr", "duration", "traffic", "voice"),
            "tickets": ("tickets", "quality", "billing")}


def spark_session(config):
    settings = config["spark"]
    builder = SparkSession.builder.appName("telecom-churn").master(settings["master"])
    for key, value in settings["settings"].items():
        builder = builder.config(key, value)
    spark = (builder.config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.csv.parser.columnPruning.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def file_date():
    return F.to_date(F.regexp_extract(F.input_file_name(), r"/dt(?:=|%3D)(\d{4}-\d{2}-\d{2})/", 1))


def event_time(column):
    return F.coalesce(F.try_to_timestamp(column),
                      F.try_to_timestamp(column, F.lit("dd.MM.yyyy HH:mm:ss")))


def read_source(spark, root, source, days, max_bytes):
    """Read daily partitions together and quarantine files with parse/encoding errors."""
    expected = CDR if source == "cdr" else TICKETS
    paths = {}
    stats = {}
    for day in days:
        path = root / "raw" / source / f"dt={day.isoformat()}" / f"{source}.csv"
        if path.is_file():
            paths[day] = path
        else:
            late = root / "late_delivery" / source / path.parent.name / path.name
            stats[day] = {"status": "late_without_delivery_time" if late.is_file() else "missing"}

    if paths:
        lines = spark.read.text([path.resolve().as_posix() for path in paths.values()])
        invalid_utf8 = lines.where(~F.is_valid_utf8("value")).select(file_date().alias("_day")).distinct()
        for row in invalid_utf8.collect():
            stats[row._day] = {"status": "corrupt"}
            paths.pop(row._day)

    if not paths:
        raise ValueError(f"No readable {source} partitions in the requested history")

    with next(iter(paths.values())).open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), [])
    if len(set(header)) != len(header) or not set(expected).issubset(header):
        raise ValueError(f"Invalid CSV header for {source}")
    schema = StructType([StructField(name, StringType()) for name in (*header, "_bad_record")])
    raw = (spark.read.schema(schema)
           .options(header=True, enforceSchema=False, mode="PERMISSIVE", escape='"',
                    multiLine=source == "tickets", columnNameOfCorruptRecord="_bad_record")
           .csv([path.resolve().as_posix() for path in paths.values()])
           .select(file_date().alias("_day"), *expected, "_bad_record"))
    # A physical CSV column must be read alongside Spark's virtual corrupt-record field.
    raw_counts = raw.groupBy("_day").agg(F.count("*").alias("rows"),
                                        F.count(KEY).alias("subscriber_keys"),
                                        F.count("_bad_record").alias("bad_records")).collect()
    bad_days = [row._day for row in raw_counts if row.bad_records]
    for day in bad_days:
        stats[day] = {"status": "corrupt"}

    if bad_days:
        raw = raw.where(~F.col("_day").isin(bad_days))
    clean = clean_events(raw, source, max_bytes).persist()
    valid_counts = {row._day: row["count"] for row in clean.groupBy("_day").count().collect()}
    raw_by_day = {row._day: row.rows for row in raw_counts}
    for day in paths:
        if day in stats:
            continue
        count = raw_by_day.get(day, 0)
        valid = valid_counts.get(day, 0)
        stats[day] = {"status": "invalid", "raw_rows": count} if count and not valid else {
            "status": "ok", "raw_rows": count, "valid_rows": valid,
        }
    for day, detail in stats.items():
        if detail["status"] != "ok":
            logging.warning("%s %s: %s", source, day, detail["status"])
    return clean, stats


def clean_events(raw, source, max_bytes):
    identifier, _, timestamp, *_ = CDR if source == "cdr" else TICKETS
    raw = raw.withColumns({"_event": event_time(timestamp), "_id": F.col(identifier).try_cast("long")})
    valid = (F.col(KEY).isNotNull() & F.col("_id").isNotNull()
             & (F.to_date("_event") == F.col("_day")))
    if source == "cdr":
        raw = raw.withColumns({
            "_ingest": event_time("ingest_ts"),
            "_duration": F.col("duration_sec").try_cast("double"),
            "_bytes_up": F.col("bytes_up").try_cast("double"),
            "_bytes_down": F.col("bytes_down").try_cast("double"),
        })
        valid &= (F.col("_ingest") >= F.col("_event")) & F.col("_duration").between(0, float("inf"))
        valid &= ~F.isnan("_duration") & (F.col("_duration") != float("inf"))
        valid &= F.col("_bytes_up").between(0, max_bytes) & F.col("_bytes_down").between(0, max_bytes)
        fields = [F.col("_ingest"), F.lit(1).alias("cdr"), F.col("_duration").alias("duration"),
                  (F.col("_bytes_up") + F.col("_bytes_down")).alias("traffic"),
                  F.coalesce(F.lower(F.trim("service_type")) == "voice", F.lit(False))
                  .cast("int").alias("voice")]
    else:
        category = F.lower(F.trim("category"))
        fields = [F.lit(1).alias("tickets"),
                  *[F.coalesce(category == name, F.lit(False)).cast("int").alias(name)
                    for name in ("quality", "billing")]]
    # Keep one canonical payload per daily business key, including conflicting duplicates.
    payload = F.min(F.struct(F.col(KEY), *fields)).alias("_payload")
    return raw.where(valid).groupBy("_day", "_id").agg(payload).select("_day", "_payload.*")


def window_sums(events, windows, source):
    matched = events.join(F.broadcast(windows), "_day")
    if source == "cdr":
        matched = matched.where(F.col("_ingest") < F.col("_as_of").cast("timestamp"))
    return matched.groupBy(KEY, "_as_of").agg(*[
        F.sum(F.when(F.col("_window") == half, F.col(counter)).otherwise(0)).alias(f"_{counter}_{half}")
        for counter in COUNTERS[source] for half in ("recent", "prior")
    ])


def preprocess(cohort, cutoffs, config):
    """Use Spark for raw events and collect only the final XGBoost feature matrix."""
    width = config["window_days"]
    periods = [(cutoff.date(), (cutoff - offset * DAY).date(), "recent" if offset <= width else "prior")
               for cutoff in cutoffs for offset in range(1, 2 * width + 1)]
    days = sorted({day for _, day, _ in periods})
    spark = spark_session(config)
    try:
        root = Path(config["data_path"])
        as_of, event_days, halves = zip(*periods)
        windows = spark.createDataFrame(pa.table({
            "_as_of": pa.array(as_of, type=pa.date32()),
            "_day": pa.array(event_days, type=pa.date32()),
            "_window": pa.array(halves, type=pa.string()),
        }))
        customers = spark.createDataFrame(pa.table({
            KEY: pa.array(cohort.index, type=pa.string()),
            "activated_at": pa.array(cohort["activated_at"].dt.date, type=pa.date32()),
        }))
        coverage = {cutoff.date(): {} for cutoff in cutoffs}
        totals = {}
        quality = []
        for source in COUNTERS:
            events, stats = read_source(spark, root, source, days, config["max_bytes_per_event"])
            quality.extend({"day": day.isoformat(), "source": source, **stats[day]} for day in days)
            for cutoff, columns in coverage.items():
                for half in ("recent", "prior"):
                    count = sum(stats[day]["status"] == "ok" for date, day, window in periods
                                if date == cutoff and window == half)
                    if count / width < config["min_coverage"]:
                        raise ValueError(f"{cutoff} {source} {half}: coverage {count}/{width}")
                    columns[f"{source}_{half}_days"] = count
            totals[source] = window_sums(events, windows, source)

        coverage_columns = next(iter(coverage.values()))
        coverage_frame = spark.createDataFrame(pa.table({
            "_as_of": pa.array(list(coverage), type=pa.date32()),
            **{name: pa.array([values[name] for values in coverage.values()], type=pa.int32())
               for name in coverage_columns},
        }))
        features = customers.crossJoin(F.broadcast(coverage_frame)).where(F.col("activated_at") < F.col("_as_of"))
        for total in totals.values():
            features = features.join(total, [KEY, "_as_of"], "left")

        required = {
            "cdr": ("cdr_recent", "cdr_prior", "duration_recent", "traffic_recent", "voice_recent"),
            "tickets": ("tickets_recent", "tickets_prior", "quality_recent", "quality_prior", "billing_recent"),
        }
        transforms = {"tenure": F.log1p(F.datediff("_as_of", "activated_at"))}
        for source, names in required.items():
            for name in names:
                half = name.rsplit("_", 1)[-1]
                transforms[name] = F.log1p(F.coalesce(F.col(f"_{name}"), F.lit(0))
                                            / F.col(f"{source}_{half}_days"))
        features = features.select(KEY, "_as_of", *[col.alias(name) for name, col in transforms.items()])
        logging.info("Building features for %d snapshot(s) with Spark %s", len(cutoffs), spark.version)
        matrix = (features.withColumn("cdr_change", F.col("cdr_recent") - F.col("cdr_prior"))
                  .orderBy("_as_of", KEY).toArrow().to_pandas())
        by_date = {day: frame.drop(columns="_as_of").set_index(KEY) for day, frame in matrix.groupby("_as_of")}
        snapshots = {cutoff: by_date[cutoff.date()] for cutoff in cutoffs}
        return snapshots, sorted(quality, key=lambda row: (row["day"], row["source"]))
    finally:
        spark.stop()
