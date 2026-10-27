"""Measured prediction timings and an isolated concurrent load benchmark."""

import shutil
import sqlite3
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)

    return {"count": len(values), "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99), "max_ms": round(ordered[-1], 3)}


def report(connection, window: int = 500) -> dict:
    if window < 1:
        raise ValueError("window must be positive")
    rows = connection.execute(
        """SELECT model_version,processing_ms FROM predictions
        WHERE processing_ms IS NOT NULL ORDER BY scored_at DESC LIMIT ?""", (window,)
    ).fetchall()
    versions = {}
    for row in rows:
        versions.setdefault(row["model_version"], []).append(row["processing_ms"])
    return {"window": window, "scope": "validation, model loading, and feature/model computation before storage write",
            "all": summarize([row["processing_ms"] for row in rows]),
            "versions": {version: summarize(values) for version, values in versions.items()}}


def benchmark(db_path: str | Path, registry: str | Path, count: int = 200, workers: int = 4) -> dict:
    """Copy the operational state and run concurrent synthetic predictions on the copy."""
    if not 1 <= count <= 5000:
        raise ValueError("count must be between 1 and 5000")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    from .api import predict

    with tempfile.TemporaryDirectory(prefix="featureforge-benchmark-") as directory:
        root = Path(directory)
        copy_db = root / "events.db"
        source = sqlite3.connect(db_path)
        try:
            target = sqlite3.connect(copy_db)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        copy_registry = root / "models"
        shutil.copytree(registry, copy_registry)
        stamp = datetime.now(timezone.utc).isoformat()
        run_id = uuid.uuid4().hex[:12]

        def one(index: int) -> tuple[float, str]:
            event = {"event_id": f"bench_{run_id}_{index}", "user_id": f"benchmark_user_{index}",
                     "event_time": stamp, "amount": float(10 + index % 900),
                     "merchant_category": "travel" if index % 5 == 0 else "retail",
                     "device_id": f"benchmark_device_{index}"}
            started = time.perf_counter_ns()
            result = predict(str(copy_db), str(copy_registry), event)
            return (time.perf_counter_ns() - started) / 1_000_000, result["model_version"]

        samples = []
        versions = {}
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, index) for index in range(count)]
            for future in as_completed(futures):
                duration, version = future.result()
                samples.append(duration)
                versions[version] = versions.get(version, 0) + 1
        elapsed = time.perf_counter() - started
        return {"requests": count, "workers": workers, "elapsed_seconds": round(elapsed, 3),
                "throughput_per_second": round(count / elapsed, 2), "wall_latency": summarize(samples),
                "model_assignments": versions, "isolated": True}
