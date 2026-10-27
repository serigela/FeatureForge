import json
import tempfile
import threading
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from featureforge.api import make_server, predict
from featureforge.broker import claim, process_one, publish, replay_dead, stats as queue_stats
from featureforge.features import compute
from featureforge.model import load, promote, train
from featureforge.monitor import report
from featureforge.online import get as get_online, parity_report, rebuild as rebuild_online
from featureforge.simulate import generate
from featureforge.store import connect, ingest, record_feedback, validate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "events.db"
        self.registry = self.root / "models"

    def test_ingestion_quarantines_invalid_rows_and_deduplicates(self):
        path = self.root / "input.jsonl"
        valid = {"event_id": "one", "user_id": "u", "event_time": "2025-01-01T00:00:00+00:00",
                 "amount": 10, "merchant_category": "retail", "device_id": "d", "label": 0}
        path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n" +
                        json.dumps({**valid, "amount": 12}) + "\n" +
                        json.dumps({**valid, "event_id": "bad", "amount": -2}) + "\n")
        with connect(self.database) as connection:
            self.assertEqual(ingest(connection, path), {"accepted": 1, "duplicates": 1, "quarantined": 2})
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0], 2)

    def test_features_use_only_prior_events(self):
        earlier = validate({"event_id": "a", "user_id": "u", "event_time": "2025-01-01T10:00:00+00:00",
                            "amount": 10, "merchant_category": "retail", "device_id": "old"})
        current = validate({"event_id": "b", "user_id": "u", "event_time": "2025-01-01T11:00:00+00:00",
                            "amount": 50, "merchant_category": "travel", "device_id": "new"})
        future = validate({"event_id": "c", "user_id": "u", "event_time": "2025-01-01T12:00:00+00:00",
                           "amount": 1000, "merchant_category": "retail", "device_id": "new"})
        with connect(self.database) as connection:
            from featureforge.store import insert_event
            for event in (earlier, current, future):
                insert_event(connection, event)
            features = compute(connection, current)
            self.assertEqual(features["count_24h"], 1)
            self.assertEqual(features["prior_mean_amount"], 10)
            self.assertEqual(features["is_new_device"], 1)
            self.assertEqual(features["amount_ratio"], 5)

    def test_24_hour_aggregate_is_not_limited_to_100_prior_events(self):
        from featureforge.store import insert_event
        with connect(self.database) as connection:
            for index in range(125):
                insert_event(connection, validate({"event_id": f"history_{index:03d}", "user_id": "busy",
                    "event_time": "2025-01-01T10:00:00+00:00", "amount": 2,
                    "merchant_category": "retail", "device_id": "known"}))
            current = validate({"event_id": "latest", "user_id": "busy",
                "event_time": "2025-01-01T11:00:00+00:00", "amount": 10,
                "merchant_category": "retail", "device_id": "known"})
            features = compute(connection, current)
            self.assertEqual(features["count_24h"], 125)
            self.assertEqual(features["amount_24h"], 250)

    def test_training_serving_replay_and_monitoring(self):
        path = generate(self.root / "events.jsonl", count=500, seed=7)
        with connect(self.database) as connection:
            self.assertEqual(ingest(connection, path)["accepted"], 500)
            model = train(connection, self.registry, epochs=75)
            self.assertEqual(model["metrics"]["test_count"], 100)
            self.assertEqual(load(self.registry)["version"], model["version"])
            promote(self.registry, model["version"])
        event = {"event_id": "live_1", "user_id": "user_0001",
                 "event_time": datetime(2025, 3, 1, tzinfo=timezone.utc).isoformat(),
                 "amount": 150, "merchant_category": "retail", "device_id": "new_device"}
        first = predict(str(self.database), str(self.registry), event)
        replay = predict(str(self.database), str(self.registry), event)
        self.assertFalse(first["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(first["score"], round(replay["score"], 6))
        with self.assertRaisesRegex(ValueError, "different event"):
            predict(str(self.database), str(self.registry), {**event, "amount": 200})
        with connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)
            self.assertEqual(report(connection, self.registry)["sample_count"], 1)

    def test_http_predict_and_health(self):
        path = generate(self.root / "events.jsonl", count=200, seed=3)
        with connect(self.database) as connection:
            ingest(connection, path)
            train(connection, self.registry, epochs=20)
        server = make_server(str(self.database), str(self.registry), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(base + "/health") as response:
                self.assertEqual(json.load(response)["status"], "ok")
            event = {"event_id": "api_1", "user_id": "user_0001",
                     "event_time": "2025-03-01T12:00:00+00:00", "amount": 200,
                     "merchant_category": "travel", "device_id": "unknown"}
            request = urllib.request.Request(base + "/predict", data=json.dumps(event).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request) as response:
                self.assertIn(json.load(response)["decision"], ("review", "approve"))
            with urllib.request.urlopen(base + "/metrics") as response:
                self.assertEqual(sum(json.load(response)["predictions"].values()), 1)
            with urllib.request.urlopen(base + "/latency") as response:
                self.assertEqual(json.load(response)["all"]["count"], 1)
            feedback = urllib.request.Request(base + "/feedback",
                data=json.dumps({"event_id": "api_1", "label": 1}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(feedback) as response:
                self.assertEqual(json.load(response)["label"], 1)
            with urllib.request.urlopen(base + "/monitor") as response:
                self.assertEqual(json.load(response)["outcomes"]["labeled_count"], 1)
            with urllib.request.urlopen(base + "/") as response:
                self.assertIn(b"FeatureForge", response.read())
            queued = {**event, "event_id": "api_queued_2", "user_id": "user_0002"}
            enqueue = urllib.request.Request(base + "/events", data=json.dumps(queued).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(enqueue) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response)["status"], "pending")
            self.assertEqual(process_one(str(self.database), str(self.registry))["status"], "done")
            with urllib.request.urlopen(base + "/events/api_queued_2") as response:
                self.assertEqual(json.load(response)["status"], "done")
            with urllib.request.urlopen(base + "/features/user_0002") as response:
                self.assertEqual(json.load(response)["event_id"], "api_queued_2")
            with urllib.request.urlopen(base + "/parity") as response:
                self.assertEqual(json.load(response)["mismatch_count"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_feedback_is_idempotent_and_cannot_change(self):
        path = generate(self.root / "events.jsonl", count=150, seed=4)
        with connect(self.database) as connection:
            ingest(connection, path)
            train(connection, self.registry, epochs=10)
        event = {"event_id": "fb_1", "user_id": "u", "event_time": "2025-03-01T00:00:00+00:00",
                 "amount": 20, "merchant_category": "retail", "device_id": "d"}
        predict(str(self.database), str(self.registry), event)
        with connect(self.database) as connection:
            self.assertFalse(record_feedback(connection, "fb_1", 1)["replayed"])
            self.assertTrue(record_feedback(connection, "fb_1", 1)["replayed"])
            with self.assertRaisesRegex(ValueError, "different label"):
                record_feedback(connection, "fb_1", 0)

    def test_rebuild_online_migrates_older_feature_snapshots(self):
        path = generate(self.root / "events.jsonl", count=150, seed=31)
        with connect(self.database) as connection:
            ingest(connection, path)
            train(connection, self.registry, epochs=10)
        event = {"event_id": "migration_1", "user_id": "migration_user",
                 "event_time": "2025-03-01T00:00:00+00:00", "amount": 50,
                 "merchant_category": "retail", "device_id": "migration_device"}
        predict(str(self.database), str(self.registry), event)
        with connect(self.database) as connection:
            connection.execute("UPDATE feature_snapshots SET feature_version='v1' WHERE event_id='migration_1'")
            connection.execute("UPDATE online_features SET feature_version='v1' WHERE user_id='migration_user'")
            self.assertEqual(parity_report(connection)["mismatch_count"], 1)
            self.assertEqual(rebuild_online(connection)["snapshots_rebuilt"], 1)
            self.assertEqual(parity_report(connection)["mismatch_count"], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)

    def test_broker_recovers_after_prediction_before_ack(self):
        path = generate(self.root / "events.jsonl", count=200, seed=6)
        with connect(self.database) as connection:
            ingest(connection, path)
            train(connection, self.registry, epochs=20)
        event = {"event_id": "queued_1", "user_id": "user_0001",
                 "event_time": "2025-03-01T12:00:00+00:00", "amount": 340,
                 "merchant_category": "travel", "device_id": "unknown"}
        with connect(self.database) as connection:
            self.assertFalse(publish(connection, event, now=100)["replayed"])
            self.assertTrue(publish(connection, event, now=100)["replayed"])
        with connect(self.database) as connection:
            message = claim(connection, now=100, lease_seconds=10)
            self.assertEqual(message["attempts"], 1)
        predict(str(self.database), str(self.registry), message["payload"])
        # Simulate a process crash: no acknowledgement. The expired lease is reclaimed.
        outcome = process_one(str(self.database), str(self.registry), now=111)
        self.assertEqual(outcome["status"], "done")
        self.assertTrue(outcome["prediction"]["replayed"])
        with connect(self.database) as connection:
            self.assertEqual(queue_stats(connection)["done"], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)
            self.assertEqual(parity_report(connection)["mismatch_count"], 0)
            self.assertEqual(get_online(connection, "user_0001")["event_id"], "queued_1")

    def test_dead_letter_replay_and_parity_alarm(self):
        path = generate(self.root / "events.jsonl", count=150, seed=8)
        with connect(self.database) as connection:
            ingest(connection, path)
            train(connection, self.registry, epochs=10)
        event = {"event_id": "queued_2", "user_id": "user_0002",
                 "event_time": "2025-03-01T12:00:00+00:00", "amount": 80,
                 "merchant_category": "retail", "device_id": "device_new"}
        with connect(self.database) as connection:
            publish(connection, event, now=100)
        broken = lambda _: (_ for _ in ()).throw(RuntimeError("simulated worker failure"))
        self.assertEqual(process_one(str(self.database), str(self.registry), now=100,
                                     handler=broken, max_attempts=2)["status"], "retry")
        self.assertEqual(process_one(str(self.database), str(self.registry), now=102,
                                     handler=broken, max_attempts=2)["status"], "dead")
        with connect(self.database) as connection:
            self.assertEqual(queue_stats(connection)["dead"], 1)
            self.assertTrue(replay_dead(connection, "queued_2", now=103))
        self.assertEqual(process_one(str(self.database), str(self.registry), now=103)["status"], "done")
        with connect(self.database) as connection:
            self.assertEqual(parity_report(connection)["mismatch_count"], 0)
            connection.execute("UPDATE online_features SET features_json='{}' WHERE user_id='user_0002'")
            self.assertEqual(parity_report(connection)["mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
