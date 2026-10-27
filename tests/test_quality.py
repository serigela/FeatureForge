import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from featureforge.api import make_server
from featureforge.quality import report
from featureforge.model import train
from featureforge.simulate import generate
from featureforge.store import connect, ingest


class QualityTests(unittest.TestCase):
    def test_quality_snapshot_and_http_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "events.db"
            payload = {"event_id": "a", "user_id": "u", "event_time": "2025-01-01T00:00:00+00:00",
                       "amount": 10, "merchant_category": "retail", "device_id": "d", "label": 1}
            source = root / "input.jsonl"
            source.write_text(json.dumps(payload) + "\n" + '{"amount": -1}\n')
            with connect(db) as connection:
                ingest(connection, source)
                result = report(connection)
            self.assertEqual(result["counts"]["events"], 1)
            self.assertEqual(result["labels"]["positive"], 1)
            self.assertEqual(result["counts"]["quarantine"], 1)
            self.assertEqual(result["quarantine_ratio"], 0.5)
            with connect(db) as connection:
                ingest(connection, generate(root / "training.jsonl", count=120, seed=3))
                train(connection, root / "models", epochs=20)
                current = report(connection)
            server = make_server(str(db), str(root / "models"), port=0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/quality") as response:
                    self.assertEqual(json.load(response)["labels"], current["labels"])
            finally:
                server.shutdown()
                worker.join()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
