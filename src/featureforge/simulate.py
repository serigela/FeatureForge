"""Deterministic synthetic events; no real financial data is required."""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(path: str | Path, count: int = 3000, seed: int = 42) -> Path:
    rng = random.Random(seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    users = [f"user_{i:04d}" for i in range(180)]
    trusted_device = {user: f"device_{user}" for user in users}
    normal_amount = {user: rng.uniform(20, 180) for user in users}
    categories = ["grocery", "travel", "retail", "dining", "utilities"]
    with destination.open("w", encoding="utf-8") as stream:
        for index in range(count):
            user = rng.choice(users)
            unusual_device = rng.random() < 0.085
            high_amount = rng.random() < 0.12
            amount = round(
                normal_amount[user] * (rng.uniform(5, 12) if high_amount else rng.uniform(0.3, 2.0)), 2
            )
            event_time = start + timedelta(minutes=index * 8, hours=rng.randrange(24))
            hour = event_time.hour
            category = rng.choice(categories)
            device = f"new_{index}" if unusual_device else trusted_device[user]
            risk = -4.0 + 1.25 * unusual_device + 1.0 * high_amount
            risk += 0.8 * (hour < 5) + 0.5 * (category == "travel")
            risk += 1.3 * (unusual_device and high_amount)
            fraud_probability = 1 / (1 + 2.718281828 ** -risk)
            event = {
                "event_id": f"evt_{index:07d}",
                "user_id": user,
                "event_time": event_time.isoformat(),
                "amount": amount,
                "merchant_category": category,
                "device_id": device,
                "label": int(rng.random() < fraud_probability),
            }
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    return destination
