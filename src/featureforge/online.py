"""Materialized online features and checks against offline recomputation."""

import json
import sqlite3
from datetime import datetime, timezone

from .features import FEATURE_NAMES, compute


FEATURE_VERSION = "v2"


def persist(connection: sqlite3.Connection, event: dict, features: dict[str, float]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(features, sort_keys=True, separators=(",", ":"))
    connection.execute(
        """INSERT INTO feature_snapshots
        (event_id,user_id,event_time,feature_version,features_json,computed_at)
        VALUES (?,?,?,?,?,?)""",
        (event["event_id"], event["user_id"], event["event_time"], FEATURE_VERSION, serialized, now),
    )
    connection.execute(
        """INSERT INTO online_features
        (user_id,event_id,event_time,feature_version,features_json,updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
        event_id=excluded.event_id,event_time=excluded.event_time,
        feature_version=excluded.feature_version,features_json=excluded.features_json,
        updated_at=excluded.updated_at
        WHERE excluded.event_time > online_features.event_time
           OR (excluded.event_time = online_features.event_time
               AND excluded.event_id > online_features.event_id)""",
        (event["user_id"], event["event_id"], event["event_time"], FEATURE_VERSION, serialized, now),
    )


def rebuild(connection: sqlite3.Connection) -> dict:
    """Recompute saved prediction features after a feature definition upgrade."""
    now = datetime.now(timezone.utc).isoformat()
    rows = connection.execute(
        """SELECT s.event_id,s.user_id,s.event_time,e.amount,e.merchant_category,e.device_id
        FROM feature_snapshots s JOIN events e USING(event_id)
        ORDER BY s.user_id,s.event_time,s.event_id"""
    ).fetchall()
    latest = {}
    for row in rows:
        event = dict(row)
        features = json.dumps(compute(connection, event), sort_keys=True, separators=(",", ":"))
        connection.execute(
            """UPDATE feature_snapshots SET feature_version=?,features_json=?,computed_at=?
            WHERE event_id=?""", (FEATURE_VERSION, features, now, row["event_id"])
        )
        latest[row["user_id"]] = (row["user_id"], row["event_id"], row["event_time"],
                                  FEATURE_VERSION, features, now)
    connection.execute("DELETE FROM online_features")
    connection.executemany(
        """INSERT INTO online_features
        (user_id,event_id,event_time,feature_version,features_json,updated_at)
        VALUES (?,?,?,?,?,?)""", latest.values()
    )
    return {"snapshots_rebuilt": len(rows), "users_rebuilt": len(latest),
            "feature_version": FEATURE_VERSION}


def get(connection: sqlite3.Connection, user_id: str, max_age_seconds: int = 300) -> dict | None:
    row = connection.execute("SELECT * FROM online_features WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        return None
    age = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(row["updated_at"])).total_seconds())
    return {"user_id": user_id, "event_id": row["event_id"], "event_time": row["event_time"],
            "feature_version": row["feature_version"], "features": json.loads(row["features_json"]),
            "age_seconds": round(age, 3), "stale": age > max_age_seconds}


def parity_report(connection: sqlite3.Connection, max_age_seconds: int = 300,
                  tolerance: float = 1e-8) -> dict:
    rows = connection.execute("SELECT * FROM online_features ORDER BY user_id").fetchall()
    mismatches = []
    stale = 0
    max_delta = 0.0
    for row in rows:
        online = get(connection, row["user_id"], max_age_seconds)
        stale += online["stale"]
        event = connection.execute("SELECT * FROM events WHERE event_id=?", (row["event_id"],)).fetchone()
        expected = compute(connection, dict(event))
        actual = online["features"]
        missing = [name for name in FEATURE_NAMES if name not in actual]
        if missing:
            mismatches.append({"event_id": row["event_id"], "missing_features": missing})
            continue
        delta = max(abs(expected[name] - actual[name]) for name in FEATURE_NAMES)
        max_delta = max(max_delta, delta)
        if delta > tolerance or online["feature_version"] != FEATURE_VERSION:
            mismatches.append({"event_id": row["event_id"], "max_delta": round(delta, 8)})
    return {"checked_users": len(rows), "mismatch_count": len(mismatches),
            "stale_count": stale, "max_delta": round(max_delta, 8),
            "mismatches": mismatches[:20], "max_age_seconds": max_age_seconds}
