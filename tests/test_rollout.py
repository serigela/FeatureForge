import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from featureforge.api import make_server, predict
from featureforge.model import list_models, load, promote, train
from featureforge.monitor import report as monitor_report
from featureforge.rollout import rollback, routing_bucket, select, start, status
from featureforge.simulate import generate
from featureforge.store import connect, ingest, record_feedback


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.database = root / "events.db"
        self.registry = root / "models"
        with connect(self.database) as connection:
            ingest(connection, generate(root / "history.jsonl", count=150, seed=12))
            self.baseline = train(connection, self.registry, epochs=10)
            self.candidate = train(connection, self.registry, epochs=12, activate=False)

    def event(self, event_id, user_id):
        return {"event_id": event_id, "user_id": user_id,
                "event_time": "2025-03-01T12:00:00+00:00", "amount": 130,
                "merchant_category": "travel", "device_id": "new"}

    def test_staged_experiment_canary_routing_feedback_and_rollback(self):
        baseline, candidate = self.baseline["version"], self.candidate["version"]
        self.assertEqual(load(self.registry)["version"], baseline)
        experiments = list_models(self.registry)
        self.assertEqual(len(experiments), 2)
        self.assertEqual(experiments[0]["dataset_sha256"], experiments[1]["dataset_sha256"])
        self.assertEqual(sum(item["active"] for item in experiments), 1)
        with self.assertRaisesRegex(ValueError, "integer from 1 to 99"):
            start(self.registry, candidate, 100)
        predict(str(self.database), str(self.registry), self.event("before_rollout", "prior_user"))
        start(self.registry, candidate, 25)
        with self.assertRaisesRegex(ValueError, "already active"):
            start(self.registry, candidate, 25)
        canary_user = next(f"u{i}" for i in range(1000) if routing_bucket(f"u{i}") < 2500)
        baseline_user = next(f"u{i}" for i in range(1000) if routing_bucket(f"u{i}") >= 2500)
        self.assertEqual(select(self.registry, canary_user)["version"], candidate)
        self.assertEqual(select(self.registry, baseline_user)["version"], baseline)
        first = predict(str(self.database), str(self.registry), self.event("canary_1", canary_user))
        second = predict(str(self.database), str(self.registry), self.event("baseline_1", baseline_user))
        self.assertEqual((first["model_version"], second["model_version"]), (candidate, baseline))
        with connect(self.database) as connection:
            shadows = connection.execute(
                "SELECT event_id,model_version FROM prediction_shadows ORDER BY event_id"
            ).fetchall()
            self.assertEqual([(row["event_id"], row["model_version"]) for row in shadows],
                             [("baseline_1", candidate), ("canary_1", baseline)])
            record_feedback(connection, "canary_1", 1)
            view = status(self.registry, connection)
            self.assertEqual(monitor_report(connection, self.registry)["sample_count"], 2)
        self.assertEqual(view["versions"][candidate]["labeled"], 1)
        self.assertEqual(view["versions"][baseline]["predictions"], 1)
        self.assertEqual(view["paired"]["event_count"], 2)
        self.assertEqual(view["paired"]["labeled_count"], 1)
        self.assertEqual(set(view["paired"]["versions"]), {candidate, baseline})
        self.assertEqual(rollback(self.registry)["active_version"], baseline)
        self.assertEqual(select(self.registry, canary_user)["version"], baseline)
        self.assertTrue(predict(str(self.database), str(self.registry), self.event("canary_1", canary_user))["replayed"])
        with connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM prediction_shadows").fetchone()[0], 2)
        self.assertEqual(predict(str(self.database), str(self.registry), self.event("after_rollback", canary_user))["model_version"], baseline)
        with self.assertRaisesRegex(ValueError, "no canary"):
            rollback(self.registry)

    def test_promotion_clears_rollout_and_health_updates_without_restart(self):
        baseline, candidate = self.baseline["version"], self.candidate["version"]
        start(self.registry, candidate, 10)
        server = make_server(str(self.database), str(self.registry), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urllib.request.urlopen(base + "/rollout") as response:
                self.assertEqual(json.load(response)["rollout"]["percent"], 10)
            with urllib.request.urlopen(base + "/models") as response:
                self.assertEqual(len(json.load(response)["models"]), 2)
            promote(self.registry, candidate)
            with urllib.request.urlopen(base + "/health") as response:
                self.assertEqual(json.load(response)["model_version"], candidate)
            with urllib.request.urlopen(base + "/rollout") as response:
                self.assertIsNone(json.load(response)["rollout"])
            self.assertEqual(load(self.registry)["version"], candidate)
            self.assertNotEqual(baseline, candidate)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
