"""A compact, read-only data quality snapshot for operators."""

import sqlite3
from datetime import datetime, timezone


def report(connection: sqlite3.Connection) -> dict:
    counts = {}
    for name in ("events", "quarantine", "predictions", "feature_snapshots", "online_features"):
        counts[name] = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    queue = {row["status"]: row["count"] for row in connection.execute(
        "SELECT status,COUNT(*) AS count FROM event_queue GROUP BY status")}
    labels = {"positive": 0, "negative": 0, "missing": 0}
    for row in connection.execute("SELECT label,COUNT(*) AS count FROM events GROUP BY label"):
        labels["missing" if row["label"] is None else
               "positive" if row["label"] == 1 else "negative"] = row["count"]
    newest = connection.execute("SELECT MAX(ingested_at) FROM events").fetchone()[0]
    age = None
    if newest:
        age = max(0, round((datetime.now(timezone.utc) - datetime.fromisoformat(newest)).total_seconds()))
    return {"counts": counts, "labels": labels, "queue": queue,
            "latest_ingest_at": newest, "latest_ingest_age_seconds": age,
            "quarantine_ratio": round(counts["quarantine"] /
                                      (counts["events"] + counts["quarantine"]), 4)
            if counts["events"] + counts["quarantine"] else None}
