"""Optional PostgreSQL warehouse mirror for events, snapshots, and predictions."""

import json
import sqlite3


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS ff_events (
  event_id text PRIMARY KEY,
  user_id text NOT NULL,
  event_time timestamptz NOT NULL,
  amount double precision NOT NULL,
  merchant_category text NOT NULL,
  device_id text NOT NULL,
  label smallint,
  ingested_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS ff_events_user_time ON ff_events(user_id,event_time,event_id);
CREATE TABLE IF NOT EXISTS ff_feature_snapshots (
  event_id text PRIMARY KEY REFERENCES ff_events(event_id),
  user_id text NOT NULL,
  event_time timestamptz NOT NULL,
  feature_version text NOT NULL,
  features jsonb NOT NULL,
  computed_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS ff_predictions (
  event_id text PRIMARY KEY REFERENCES ff_events(event_id),
  model_version text NOT NULL,
  score double precision NOT NULL,
  decision text NOT NULL,
  scored_at timestamptz NOT NULL
);
"""


def _driver():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install the optional Postgres dependency: pip install '.[postgres]'") from exc
    return psycopg


def sync(connection: sqlite3.Connection, dsn: str, batch_size: int = 500) -> dict:
    """Idempotently copy local operational records into PostgreSQL."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    psycopg = _driver()
    counts = {"events": 0, "feature_snapshots": 0, "predictions": 0}
    with psycopg.connect(dsn) as pg:
        with pg.cursor() as cursor:
            cursor.execute(POSTGRES_SCHEMA)
            specs = (
                ("events", "SELECT * FROM events ORDER BY event_id",
                 """INSERT INTO ff_events(event_id,user_id,event_time,amount,merchant_category,device_id,label,ingested_at)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                 ON CONFLICT(event_id) DO UPDATE SET label=EXCLUDED.label""",
                 lambda row: tuple(row[key] for key in ("event_id", "user_id", "event_time", "amount",
                     "merchant_category", "device_id", "label", "ingested_at"))),
                ("feature_snapshots", "SELECT * FROM feature_snapshots ORDER BY event_id",
                 """INSERT INTO ff_feature_snapshots
                 (event_id,user_id,event_time,feature_version,features,computed_at)
                 VALUES (%s,%s,%s,%s,%s::jsonb,%s)
                 ON CONFLICT(event_id) DO UPDATE SET features=EXCLUDED.features,
                 feature_version=EXCLUDED.feature_version,computed_at=EXCLUDED.computed_at""",
                 lambda row: tuple(row[key] for key in ("event_id", "user_id", "event_time",
                     "feature_version", "features_json", "computed_at"))),
                ("predictions", "SELECT * FROM predictions ORDER BY event_id",
                 """INSERT INTO ff_predictions(event_id,model_version,score,decision,scored_at)
                 VALUES (%s,%s,%s,%s,%s)
                 ON CONFLICT(event_id) DO NOTHING""",
                 lambda row: tuple(row[key] for key in ("event_id", "model_version", "score",
                     "decision", "scored_at"))),
            )
            for name, query, statement, transform in specs:
                source = connection.execute(query)
                while batch := source.fetchmany(batch_size):
                    cursor.executemany(statement, [transform(row) for row in batch])
                    counts[name] += len(batch)
    return counts


def verify(connection: sqlite3.Connection, dsn: str) -> dict:
    psycopg = _driver()
    tables = {"events": "ff_events", "feature_snapshots": "ff_feature_snapshots",
              "predictions": "ff_predictions"}
    result = {}
    with psycopg.connect(dsn) as pg:
        with pg.cursor() as cursor:
            for local, remote in tables.items():
                local_count = connection.execute(f"SELECT COUNT(*) FROM {local}").fetchone()[0]
                cursor.execute(f"SELECT COUNT(*) FROM {remote}")
                remote_count = cursor.fetchone()[0]
                result[local] = {"local": local_count, "postgres": remote_count,
                                 "in_sync": local_count == remote_count}
    result["in_sync"] = all(value["in_sync"] for value in result.values())
    return result
