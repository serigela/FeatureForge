"""Durable at-least-once event broker with leases, retries, and dead letters."""

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from .store import connect, validate


def publish(connection: sqlite3.Connection, raw: dict, now: float | None = None) -> dict:
    event = validate(raw)
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
    existing = connection.execute(
        "SELECT payload_json,status FROM event_queue WHERE event_id=?", (event["event_id"],)
    ).fetchone()
    if existing:
        if existing["payload_json"] != payload:
            raise ValueError("event_id already queued with a different payload")
        return {"event_id": event["event_id"], "status": existing["status"], "replayed": True}
    connection.execute(
        """INSERT INTO event_queue(event_id,payload_json,status,available_at,created_at)
        VALUES (?,?,?,?,?)""",
        (event["event_id"], payload, "pending", time.time() if now is None else now,
         datetime.now(timezone.utc).isoformat()),
    )
    return {"event_id": event["event_id"], "status": "pending", "replayed": False}


def claim(connection: sqlite3.Connection, now: float | None = None, lease_seconds: int = 30) -> dict | None:
    instant = time.time() if now is None else now
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """SELECT event_id,payload_json,attempts FROM event_queue
            WHERE (status IN ('pending','retry') AND available_at <= ?)
               OR (status='inflight' AND lease_until <= ?)
            ORDER BY available_at,created_at LIMIT 1""", (instant, instant),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        token = uuid.uuid4().hex
        connection.execute(
            """UPDATE event_queue SET status='inflight',attempts=attempts+1,
            lease_until=?,lease_token=? WHERE event_id=?""",
            (instant + lease_seconds, token, row["event_id"]),
        )
        connection.commit()
        return {"event_id": row["event_id"], "payload": json.loads(row["payload_json"]),
                "attempts": row["attempts"] + 1, "lease_token": token}
    except Exception:
        connection.rollback()
        raise


def ack(connection: sqlite3.Connection, message: dict) -> bool:
    cursor = connection.execute(
        """UPDATE event_queue SET status='done',lease_until=NULL,lease_token=NULL,
        last_error=NULL,completed_at=? WHERE event_id=? AND status='inflight' AND lease_token=?""",
        (datetime.now(timezone.utc).isoformat(), message["event_id"], message["lease_token"]),
    )
    return cursor.rowcount == 1


def fail(connection: sqlite3.Connection, message: dict, error: Exception,
         now: float | None = None, max_attempts: int = 5) -> str:
    instant = time.time() if now is None else now
    dead = message["attempts"] >= max_attempts
    status = "dead" if dead else "retry"
    delay = 0 if dead else min(60, 2 ** (message["attempts"] - 1))
    cursor = connection.execute(
        """UPDATE event_queue SET status=?,available_at=?,lease_until=NULL,
        lease_token=NULL,last_error=? WHERE event_id=? AND status='inflight' AND lease_token=?""",
        (status, instant + delay, str(error)[:500], message["event_id"], message["lease_token"]),
    )
    if cursor.rowcount != 1:
        raise ValueError("message lease has expired or changed owner")
    return status


def replay_dead(connection: sqlite3.Connection, event_id: str, now: float | None = None) -> bool:
    cursor = connection.execute(
        """UPDATE event_queue SET status='pending',attempts=0,available_at=?,
        lease_until=NULL,lease_token=NULL,last_error=NULL WHERE event_id=? AND status='dead'""",
        (time.time() if now is None else now, event_id),
    )
    return cursor.rowcount == 1


def stats(connection: sqlite3.Connection) -> dict:
    counts = connection.execute("SELECT status,COUNT(*) AS n FROM event_queue GROUP BY status").fetchall()
    return {status: next((row["n"] for row in counts if row["status"] == status), 0)
            for status in ("pending", "inflight", "retry", "done", "dead")}


def process_one(db_path: str, registry: str, now: float | None = None,
                handler=None, max_attempts: int = 5) -> dict | None:
    from .api import predict

    with connect(db_path) as connection:
        message = claim(connection, now=now)
    if message is None:
        return None
    operation = handler or (lambda payload: predict(db_path, registry, payload))
    try:
        outcome = operation(message["payload"])
    except Exception as exc:
        with connect(db_path) as connection:
            status = fail(connection, message, exc, now=now, max_attempts=max_attempts)
        return {"event_id": message["event_id"], "status": status, "attempts": message["attempts"],
                "error": str(exc)[:500]}
    with connect(db_path) as connection:
        acknowledged = ack(connection, message)
    return {"event_id": message["event_id"], "status": "done" if acknowledged else "lease_lost",
            "attempts": message["attempts"], "prediction": outcome}


def run_worker(db_path: str, registry: str, once: bool = False, poll_seconds: float = 1.0) -> None:
    while True:
        result = process_one(db_path, registry)
        if result is not None:
            print(json.dumps(result, sort_keys=True), flush=True)
        if once:
            return
        if result is None:
            time.sleep(poll_seconds)
