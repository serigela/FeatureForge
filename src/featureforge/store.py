"""Validated ingestion and durable local event/prediction store."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  event_time TEXT NOT NULL,
  amount REAL NOT NULL CHECK(amount > 0),
  merchant_category TEXT NOT NULL,
  device_id TEXT NOT NULL,
  label INTEGER CHECK(label IN (0, 1) OR label IS NULL),
  ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_user_time ON events(user_id, event_time, event_id);
CREATE TABLE IF NOT EXISTS quarantine (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  raw_line TEXT NOT NULL,
  reason TEXT NOT NULL,
  recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  event_id TEXT PRIMARY KEY REFERENCES events(event_id),
  model_version TEXT NOT NULL,
  score REAL NOT NULL,
  decision TEXT NOT NULL,
  features_json TEXT NOT NULL,
  scored_at TEXT NOT NULL,
  processing_ms REAL CHECK(processing_ms >= 0)
);
CREATE TABLE IF NOT EXISTS prediction_shadows (
  event_id TEXT PRIMARY KEY REFERENCES predictions(event_id),
  model_version TEXT NOT NULL,
  score REAL NOT NULL,
  decision TEXT NOT NULL,
  scored_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_model_time ON predictions(model_version,scored_at);
CREATE TABLE IF NOT EXISTS event_queue (
  event_id TEXT PRIMARY KEY,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','inflight','retry','done','dead')),
  attempts INTEGER NOT NULL DEFAULT 0,
  available_at REAL NOT NULL,
  lease_until REAL,
  lease_token TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);
CREATE INDEX IF NOT EXISTS event_queue_ready ON event_queue(status,available_at);
CREATE TABLE IF NOT EXISTS feature_snapshots (
  event_id TEXT PRIMARY KEY REFERENCES events(event_id),
  user_id TEXT NOT NULL,
  event_time TEXT NOT NULL,
  feature_version TEXT NOT NULL,
  features_json TEXT NOT NULL,
  computed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS online_features (
  user_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES feature_snapshots(event_id),
  event_time TEXT NOT NULL,
  feature_version TEXT NOT NULL,
  features_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


class ManagedConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(path: str | Path) -> sqlite3.Connection:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination, timeout=30, factory=ManagedConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)")}
    if "processing_ms" not in columns:
        connection.execute("ALTER TABLE predictions ADD COLUMN processing_ms REAL")
    return connection


def validate(raw: dict) -> dict:
    required = ("event_id", "user_id", "event_time", "amount", "merchant_category", "device_id")
    if not isinstance(raw, dict):
        raise ValueError("event must be an object")
    for key in required:
        if key not in raw:
            raise ValueError(f"missing {key}")
    event = {key: raw[key] for key in required}
    for key in ("event_id", "user_id", "merchant_category", "device_id"):
        if not isinstance(event[key], str) or not event[key].strip():
            raise ValueError(f"invalid {key}")
    if len(event["event_id"]) > 128 or len(event["user_id"]) > 128:
        raise ValueError("identifier too long")
    try:
        stamp = datetime.fromisoformat(event["event_time"])
        if stamp.tzinfo is None:
            raise ValueError("event_time needs timezone")
        event["event_time"] = stamp.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid event_time; use timezone-aware ISO 8601") from exc
    if isinstance(event["amount"], bool) or not isinstance(event["amount"], (int, float)):
        raise ValueError("amount must be numeric")
    event["amount"] = float(event["amount"])
    if not 0 < event["amount"] <= 1_000_000:
        raise ValueError("amount out of range")
    label = raw.get("label")
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise ValueError("label must be 0 or 1")
    event["label"] = label
    return event


def insert_event(connection: sqlite3.Connection, event: dict) -> bool:
    existing = connection.execute("SELECT * FROM events WHERE event_id=?", (event["event_id"],)).fetchone()
    if existing:
        fields = ("user_id", "event_time", "amount", "merchant_category", "device_id", "label")
        if any(existing[key] != event[key] for key in fields):
            raise ValueError("event_id already exists with different data")
        return False
    now = datetime.now(timezone.utc).isoformat()
    cursor = connection.execute(
        """INSERT OR IGNORE INTO events
        (event_id,user_id,event_time,amount,merchant_category,device_id,label,ingested_at)
        VALUES (:event_id,:user_id,:event_time,:amount,:merchant_category,:device_id,:label,:ingested_at)""",
        {**event, "ingested_at": now},
    )
    return cursor.rowcount == 1


def ingest(connection: sqlite3.Connection, path: str | Path) -> dict:
    stats = {"accepted": 0, "duplicates": 0, "quarantined": 0}
    now = datetime.now(timezone.utc).isoformat()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = validate(json.loads(line))
                stats["accepted" if insert_event(connection, event) else "duplicates"] += 1
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                connection.execute(
                    "INSERT INTO quarantine(raw_line,reason,recorded_at) VALUES (?,?,?)",
                    (line.rstrip(), str(exc), now),
                )
                stats["quarantined"] += 1
    return stats


def record_feedback(connection: sqlite3.Connection, event_id: str, label: int) -> dict:
    if type(label) is not int or label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    row = connection.execute(
        """SELECT e.label FROM events e JOIN predictions p USING(event_id)
        WHERE e.event_id=?""", (event_id,)
    ).fetchone()
    if row is None:
        raise ValueError("prediction not found")
    if row["label"] is not None and row["label"] != label:
        raise ValueError("feedback already recorded with a different label")
    if row["label"] is None:
        connection.execute("UPDATE events SET label=? WHERE event_id=?", (label, event_id))
    return {"event_id": event_id, "label": label, "replayed": row["label"] is not None}
