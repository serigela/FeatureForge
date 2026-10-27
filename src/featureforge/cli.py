"""Command line workflow for generating, ingesting, training, and serving."""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .api import serve
from .broker import process_one, publish, replay_dead, run_worker, stats as queue_stats
from .model import list_models, load, promote, train
from .monitor import report
from .rollout import rollback, start as start_canary, status as rollout_status
from .online import parity_report, rebuild as rebuild_online
from .performance import benchmark, report as latency_report
from .simulate import generate
from .store import connect, ingest
from .warehouse import sync as sync_postgres, verify as verify_postgres


def main() -> None:
    parser = argparse.ArgumentParser(prog="featureforge")
    parser.add_argument("--data-dir", type=Path, default=Path(".featureforge"))
    commands = parser.add_subparsers(dest="command", required=True)
    generate_cmd = commands.add_parser("generate", help="create deterministic synthetic events")
    generate_cmd.add_argument("--count", type=int, default=3000)
    generate_cmd.add_argument("--seed", type=int, default=42)
    ingest_cmd = commands.add_parser("ingest", help="validate and import JSONL events")
    ingest_cmd.add_argument("path", type=Path)
    train_cmd = commands.add_parser("train", help="train and evaluate a time-split model")
    train_cmd.add_argument("--epochs", type=int, default=350)
    train_cmd.add_argument("--stage", action="store_true", help="save candidate without making it active")
    serve_cmd = commands.add_parser("serve", help="start the prediction API")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)
    monitor_cmd = commands.add_parser("monitor", help="check live feature drift")
    monitor_cmd.add_argument("--window", type=int, default=500)
    promote_cmd = commands.add_parser("promote", help="select an existing model version")
    promote_cmd.add_argument("version")
    commands.add_parser("models", help="list experiments and evaluation metrics")
    canary_cmd = commands.add_parser("canary", help="route a stable percentage of users to a candidate")
    canary_cmd.add_argument("version")
    canary_cmd.add_argument("--percent", type=int, required=True)
    commands.add_parser("rollout", help="show rollout and labeled outcome counts by version")
    commands.add_parser("rollback", help="stop the canary and keep the baseline active")
    latency_cmd = commands.add_parser("latency", help="show measured prediction processing time")
    latency_cmd.add_argument("--window", type=int, default=500)
    benchmark_cmd = commands.add_parser("benchmark", help="run an isolated concurrent prediction load test")
    benchmark_cmd.add_argument("--count", type=int, default=200)
    benchmark_cmd.add_argument("--workers", type=int, default=4)
    publish_cmd = commands.add_parser("publish", help="enqueue one JSON event from a file")
    publish_cmd.add_argument("path", type=Path)
    worker_cmd = commands.add_parser("worker", help="process queued events with retries")
    worker_cmd.add_argument("--once", action="store_true")
    replay_cmd = commands.add_parser("replay", help="requeue a dead letter")
    replay_cmd.add_argument("event_id")
    commands.add_parser("queue-stats", help="show broker queue counts")
    commands.add_parser("recovery-demo", help="simulate crash after scoring and verify replay")
    parity_cmd = commands.add_parser("parity", help="compare online features with offline recomputation")
    parity_cmd.add_argument("--max-age", type=int, default=300)
    commands.add_parser("rebuild-online", help="recompute saved online features after an upgrade")
    commands.add_parser("sync-postgres", help="mirror local data to PostgreSQL")
    commands.add_parser("verify-postgres", help="compare local and PostgreSQL row counts")
    commands.add_parser("demo", help="generate, ingest, train, and print evaluation")
    args = parser.parse_args()
    data_dir = args.data_dir
    db_path = data_dir / "events.db"
    registry = data_dir / "models"
    if args.command == "generate":
        if args.count < 100:
            parser.error("--count must be at least 100")
        path = generate(data_dir / "synthetic_transactions.jsonl", args.count, args.seed)
        print(path)
    elif args.command == "ingest":
        with connect(db_path) as connection:
            print(json.dumps(ingest(connection, args.path), indent=2))
    elif args.command == "train":
        with connect(db_path) as connection:
            model = train(connection, registry, args.epochs, activate=not args.stage)
        print(json.dumps({"version": model["version"], "threshold": model["threshold"],
                          "dataset_sha256": model["dataset_sha256"],
                          "active": load(registry)["version"] == model["version"],
                          "metrics": model["metrics"]}, indent=2))
    elif args.command == "serve":
        serve(str(db_path), str(registry), args.host, args.port)
    elif args.command == "monitor":
        with connect(db_path) as connection:
            print(json.dumps(report(connection, registry, args.window), indent=2))
    elif args.command == "promote":
        promote(registry, args.version)
        print(json.dumps({"active_version": load(registry)["version"]}))
    elif args.command == "models":
        print(json.dumps({"models": list_models(registry)}, indent=2))
    elif args.command == "canary":
        print(json.dumps(start_canary(registry, args.version, args.percent), indent=2))
    elif args.command == "rollback":
        print(json.dumps(rollback(registry), indent=2))
    elif args.command == "rollout":
        with connect(db_path) as connection:
            print(json.dumps(rollout_status(registry, connection), indent=2))
    elif args.command == "latency":
        with connect(db_path) as connection:
            print(json.dumps(latency_report(connection, args.window), indent=2))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(db_path, registry, args.count, args.workers), indent=2))
    elif args.command == "publish":
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        with connect(db_path) as connection:
            print(json.dumps(publish(connection, payload), indent=2))
    elif args.command == "worker":
        run_worker(str(db_path), str(registry), once=args.once)
    elif args.command == "replay":
        with connect(db_path) as connection:
            print(json.dumps({"replayed": replay_dead(connection, args.event_id)}, indent=2))
    elif args.command == "queue-stats":
        with connect(db_path) as connection:
            print(json.dumps(queue_stats(connection), indent=2))
    elif args.command == "recovery-demo":
        from .broker import claim
        from .api import predict
        event = {"event_id": "recovery_" + uuid.uuid4().hex[:12], "user_id": "recovery_user",
                 "event_time": datetime.now(timezone.utc).isoformat(), "amount": 75,
                 "merchant_category": "retail", "device_id": "recovery_device"}
        with connect(db_path) as connection:
            publish(connection, event, now=100)
        with connect(db_path) as connection:
            leased = claim(connection, now=100, lease_seconds=10)
        first = predict(str(db_path), str(registry), leased["payload"])
        recovered = process_one(str(db_path), str(registry), now=111)
        with connect(db_path) as connection:
            prediction_count = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE event_id=?", (event["event_id"],)
            ).fetchone()[0]
        print(json.dumps({"event_id": event["event_id"], "first_replayed": first["replayed"],
                          "recovery_status": recovered["status"],
                          "recovery_replayed": recovered["prediction"]["replayed"],
                          "prediction_count": prediction_count}, indent=2))
    elif args.command == "parity":
        with connect(db_path) as connection:
            print(json.dumps(parity_report(connection, args.max_age), indent=2))
    elif args.command == "rebuild-online":
        with connect(db_path) as connection:
            print(json.dumps(rebuild_online(connection), indent=2))
    elif args.command in ("sync-postgres", "verify-postgres"):
        dsn = os.environ.get("FEATUREFORGE_POSTGRES_DSN")
        if not dsn:
            parser.error("set FEATUREFORGE_POSTGRES_DSN to connect to PostgreSQL")
        with connect(db_path) as connection:
            result = sync_postgres(connection, dsn) if args.command == "sync-postgres" else verify_postgres(connection, dsn)
        print(json.dumps(result, indent=2))
    elif args.command == "demo":
        path = generate(data_dir / "synthetic_transactions.jsonl")
        with connect(db_path) as connection:
            stats = ingest(connection, path)
            model = train(connection, registry)
        print(json.dumps({"ingestion": stats, "model_version": model["version"],
                          "threshold": model["threshold"], "metrics": model["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
