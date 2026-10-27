"""Runs against the CI PostgreSQL service or a local compose instance."""

import os
import tempfile
import unittest
from pathlib import Path

from featureforge.api import predict
from featureforge.model import train
from featureforge.simulate import generate
from featureforge.store import connect, ingest, record_feedback
from featureforge.warehouse import sync, verify


@unittest.skipUnless(os.environ.get("FEATUREFORGE_POSTGRES_DSN"), "Postgres DSN not configured")
class PostgresWarehouseTests(unittest.TestCase):
    def test_sync_is_idempotent_and_updates_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "events.db"
            registry = root / "models"
            path = generate(root / "events.jsonl", count=160, seed=17)
            with connect(database) as connection:
                ingest(connection, path)
                train(connection, registry, epochs=10)
            event = {"event_id": "pg_demo_1", "user_id": "u_pg",
                     "event_time": "2025-03-01T00:00:00+00:00", "amount": 125,
                     "merchant_category": "travel", "device_id": "d_pg"}
            predict(str(database), str(registry), event)
            dsn = os.environ["FEATUREFORGE_POSTGRES_DSN"]
            with connect(database) as connection:
                self.assertEqual(sync(connection, dsn)["predictions"], 1)
                self.assertTrue(verify(connection, dsn)["in_sync"])
                self.assertEqual(sync(connection, dsn)["predictions"], 1)
                self.assertTrue(verify(connection, dsn)["in_sync"])
                record_feedback(connection, "pg_demo_1", 1)
            with connect(database) as connection:
                sync(connection, dsn)
                self.assertTrue(verify(connection, dsn)["in_sync"])
            import psycopg
            with psycopg.connect(dsn) as pg:
                with pg.cursor() as cursor:
                    cursor.execute("SELECT label FROM ff_events WHERE event_id=%s", ("pg_demo_1",))
                    self.assertEqual(cursor.fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
