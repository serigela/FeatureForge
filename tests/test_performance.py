import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from featureforge.api import predict
from featureforge.model import load, train
from featureforge.performance import benchmark, report, summarize
from featureforge.simulate import generate
from featureforge.store import connect, ingest


class PerformanceTests(unittest.TestCase):
    def test_existing_database_is_migrated_and_timings_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.db"
            old = sqlite3.connect(database)
            old.execute("""CREATE TABLE predictions (event_id TEXT PRIMARY KEY, model_version TEXT NOT NULL,
                score REAL NOT NULL, decision TEXT NOT NULL, features_json TEXT NOT NULL,
                scored_at TEXT NOT NULL)""")
            old.commit()
            old.close()
            with connect(database) as connection:
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)")}
                self.assertIn("processing_ms", columns)
                ingest(connection, generate(root / "history.jsonl", count=120, seed=19))
                train(connection, root / "models", epochs=8)
            predict(str(database), str(root / "models"), {
                "event_id": "timed_1", "user_id": "timed_user", "event_time": "2025-03-01T00:00:00+00:00",
                "amount": 50, "merchant_category": "retail", "device_id": "timed_device"})
            with connect(database) as connection:
                timing = report(connection)
                self.assertEqual(timing["all"]["count"], 1)
                self.assertGreaterEqual(timing["all"]["p95_ms"], 0)

    def test_benchmark_is_isolated_and_reports_observed_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.db"
            registry = root / "models"
            with connect(database) as connection:
                ingest(connection, generate(root / "history.jsonl", count=120, seed=21))
                train(connection, registry, epochs=8)
            result = benchmark(database, registry, count=24, workers=3)
            self.assertEqual(result["requests"], 24)
            self.assertEqual(result["wall_latency"]["count"], 24)
            self.assertEqual(sum(result["model_assignments"].values()), 24)
            self.assertGreater(result["throughput_per_second"], 0)
            with connect(database) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)

    def test_percentiles_are_monotonic(self):
        values = summarize([1, 2, 3, 4, 5])
        self.assertEqual(values["p50_ms"], 3)
        self.assertLessEqual(values["p50_ms"], values["p95_ms"])
        self.assertLessEqual(values["p95_ms"], values["p99_ms"])

    def test_older_feature_definition_models_require_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.db"
            registry = root / "models"
            with connect(database) as connection:
                ingest(connection, generate(root / "history.jsonl", count=120, seed=23))
                model = train(connection, registry, epochs=8)
            path = registry / f"model-{model['version']}.json"
            old = json.loads(path.read_text())
            old.pop("feature_version")
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "retrain"):
                load(registry)


if __name__ == "__main__":
    unittest.main()
