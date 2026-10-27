"""Small HTTP service using the Python standard library."""

import json
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .features import compute
from .model import list_models, load, score
from .monitor import report
from .rollout import select as select_model, status as rollout_status
from .online import get as get_online, parity_report, persist
from .store import connect, insert_event, record_feedback, validate
from .dashboard import HTML
from .broker import publish, replay_dead, stats as queue_stats


def predict(db_path: str, registry: str, raw: dict) -> dict:
    event = validate(raw)
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT model_version,score,decision FROM predictions WHERE event_id=?", (event["event_id"],)
        ).fetchone()
        if previous:
            original = connection.execute("SELECT * FROM events WHERE event_id=?", (event["event_id"],)).fetchone()
            if any(original[key] != event[key] for key in
                   ("user_id", "event_time", "amount", "merchant_category", "device_id")):
                raise ValueError("event_id already used for a different event")
            return {"event_id": event["event_id"], "model_version": previous["model_version"],
                    "score": previous["score"], "decision": previous["decision"], "replayed": True}
        if connection.execute("SELECT 1 FROM events WHERE event_id=?", (event["event_id"],)).fetchone():
            raise ValueError("event_id already exists in historical data")
        model = select_model(registry, event["user_id"])
        feature_values = compute(connection, event)
        probability = score(model, feature_values)
        decision = "review" if probability >= model["threshold"] else "approve"
        insert_event(connection, event)
        persist(connection, event, feature_values)
        connection.execute(
            """INSERT INTO predictions(event_id,model_version,score,decision,features_json,scored_at)
            VALUES (?,?,?,?,?,?)""",
            (event["event_id"], model["version"], probability, decision,
             json.dumps(feature_values, sort_keys=True), datetime.now(timezone.utc).isoformat()),
        )
        return {"event_id": event["event_id"], "model_version": model["version"],
                "score": round(probability, 6), "decision": decision, "replayed": False}


def make_server(db_path: str, registry: str, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    load(registry)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self.send_json(200, {"status": "ok", "model_version": load(registry)["version"]})
            elif self.path == "/models":
                self.send_json(200, {"models": list_models(registry)})
            elif self.path == "/rollout":
                with connect(db_path) as connection:
                    self.send_json(200, rollout_status(registry, connection))
            elif self.path == "/metrics":
                with connect(db_path) as connection:
                    counts = connection.execute(
                        "SELECT decision,COUNT(*) AS count FROM predictions GROUP BY decision"
                    ).fetchall()
                    queue = queue_stats(connection)
                self.send_json(200, {"predictions": {row["decision"]: row["count"] for row in counts},
                                     "queue": queue})
            elif self.path == "/monitor":
                with connect(db_path) as connection:
                    self.send_json(200, report(connection, registry))
            elif self.path == "/parity":
                with connect(db_path) as connection:
                    self.send_json(200, parity_report(connection))
            elif self.path.startswith("/features/"):
                user_id = self.path.removeprefix("/features/")
                with connect(db_path) as connection:
                    snapshot = get_online(connection, user_id)
                self.send_json(200, snapshot) if snapshot else self.send_json(404, {"error": "user not found"})
            elif self.path.startswith("/events/"):
                event_id = self.path.removeprefix("/events/")
                with connect(db_path) as connection:
                    row = connection.execute("SELECT status,attempts,last_error FROM event_queue WHERE event_id=?",
                                             (event_id,)).fetchone()
                self.send_json(200, dict(row)) if row else self.send_json(404, {"error": "event not found"})
            elif self.path == "/queue":
                with connect(db_path) as connection:
                    self.send_json(200, queue_stats(connection))
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/predict", "/feedback", "/events", "/replay"):
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16384:
                    raise ValueError("request body size must be 1–16384 bytes")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/predict":
                    self.send_json(200, predict(db_path, registry, payload))
                elif self.path == "/events":
                    with connect(db_path) as connection:
                        queued = publish(connection, payload)
                    self.send_json(202, queued)
                elif self.path == "/replay":
                    if not isinstance(payload, dict) or not isinstance(payload.get("event_id"), str):
                        raise ValueError("event_id is required")
                    with connect(db_path) as connection:
                        replayed = replay_dead(connection, payload["event_id"])
                    self.send_json(200, {"replayed": replayed})
                else:
                    if not isinstance(payload, dict) or not isinstance(payload.get("event_id"), str):
                        raise ValueError("event_id is required")
                    with connect(db_path) as connection:
                        outcome = record_feedback(connection, payload["event_id"], payload.get("label"))
                    self.send_json(200, outcome)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except sqlite3.Error:
                self.send_json(503, {"error": "storage unavailable"})

    return ThreadingHTTPServer((host, port), Handler)


def serve(db_path: str, registry: str, host: str = "127.0.0.1", port: int = 8080) -> None:
    server = make_server(db_path, registry, host, port)
    print(f"FeatureForge listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
