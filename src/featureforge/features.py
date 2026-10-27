"""One feature definition for training and live inference, with past-only history."""

import sqlite3
from datetime import datetime, timedelta


FEATURE_NAMES = (
    "log_amount", "hour", "is_weekend", "count_24h", "amount_24h",
    "prior_mean_amount", "amount_ratio", "is_new_device", "category_travel",
)


def compute(connection: sqlite3.Connection, event: dict) -> dict[str, float]:
    import math

    stamp = datetime.fromisoformat(event["event_time"])
    since = (stamp - timedelta(hours=24)).isoformat()
    history = connection.execute(
        """SELECT amount, device_id, event_time FROM events
        WHERE user_id=? AND (event_time < ? OR (event_time = ? AND event_id < ?))
        ORDER BY event_time DESC, event_id DESC LIMIT 100""",
        (event["user_id"], event["event_time"], event["event_time"], event["event_id"]),
    ).fetchall()
    recent = connection.execute(
        """SELECT COUNT(*) AS count, COALESCE(SUM(amount),0) AS amount FROM events
        WHERE user_id=? AND event_time >= ?
          AND (event_time < ? OR (event_time = ? AND event_id < ?))""",
        (event["user_id"], since, event["event_time"], event["event_time"], event["event_id"]),
    ).fetchone()
    previous_mean = sum(row["amount"] for row in history) / len(history) if history else 0.0
    return {
        "log_amount": math.log1p(event["amount"]),
        "hour": float(stamp.hour),
        "is_weekend": float(stamp.weekday() >= 5),
        "count_24h": float(recent["count"]),
        "amount_24h": float(recent["amount"]),
        "prior_mean_amount": previous_mean,
        "amount_ratio": min(event["amount"] / previous_mean, 100.0) if previous_mean else 1.0,
        "is_new_device": float(not any(row["device_id"] == event["device_id"] for row in history)),
        "category_travel": float(event["merchant_category"].lower() == "travel"),
    }


def labeled_rows(connection: sqlite3.Connection) -> list[tuple[dict, int]]:
    rows = connection.execute(
        "SELECT * FROM events WHERE label IS NOT NULL ORDER BY event_time,event_id"
    ).fetchall()
    return [(compute(connection, dict(row)), row["label"]) for row in rows]
