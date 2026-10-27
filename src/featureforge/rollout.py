"""Stable user-level canary routing and observable model rollout state."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .model import load, load_version


def configuration(registry: str | Path) -> dict | None:
    path = Path(registry) / "rollout.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def start(registry: str | Path, version: str, percent: int) -> dict:
    if type(percent) is not int or not 1 <= percent <= 99:
        raise ValueError("canary percent must be an integer from 1 to 99")
    directory = Path(registry)
    baseline = load(directory)["version"]
    load_version(directory, version)
    if version == baseline:
        raise ValueError("canary must differ from the active model")
    if configuration(directory):
        raise ValueError("a canary rollout is already active; promote or rollback it first")
    rollout = {"baseline_version": baseline, "canary_version": version, "percent": percent,
               "started_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".rollout-", delete=False,
                                     encoding="utf-8") as stream:
        json.dump(rollout, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, directory / "rollout.json")
    return rollout


def rollback(registry: str | Path) -> dict:
    directory = Path(registry)
    rollout = configuration(directory)
    if not rollout:
        raise ValueError("no canary rollout is active")
    (directory / "rollout.json").unlink()
    return {"active_version": load(directory)["version"], "stopped_version": rollout["canary_version"]}


def routing_bucket(user_id: str) -> int:
    return int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:8], "big") % 10000


def select(registry: str | Path, user_id: str) -> dict:
    return select_pair(registry, user_id)[0]


def select_pair(registry: str | Path, user_id: str) -> tuple[dict, dict | None]:
    active = load(registry)
    rollout = configuration(registry)
    if rollout and rollout["baseline_version"] != active["version"]:
        raise ValueError("rollout baseline no longer matches active model")
    if not rollout:
        return active, None
    candidate = load_version(registry, rollout["canary_version"])
    return (candidate, active) if routing_bucket(user_id) < rollout["percent"] * 100 else (active, candidate)


def status(registry: str | Path, connection=None) -> dict:
    baseline = load(registry)["version"]
    rollout = configuration(registry)
    result = {"active_version": baseline, "rollout": rollout, "versions": {}, "paired": None}
    if connection is None:
        return result
    versions = [baseline]
    if rollout:
        versions.append(rollout["canary_version"])
    since = rollout["started_at"] if rollout else ""
    for version in versions:
        rows = connection.execute(
            """SELECT p.decision,e.label FROM predictions p JOIN events e USING(event_id)
            WHERE p.model_version=? AND p.scored_at>=?""", (version, since)
        ).fetchall()
        labeled = [row for row in rows if row["label"] is not None]
        tp = sum(row["label"] == 1 and row["decision"] == "review" for row in labeled)
        fp = sum(row["label"] == 0 and row["decision"] == "review" for row in labeled)
        fn = sum(row["label"] == 1 and row["decision"] == "approve" for row in labeled)
        result["versions"][version] = {
            "predictions": len(rows), "labeled": len(labeled),
            "review_rate": round(sum(row["decision"] == "review" for row in rows) / len(rows), 4)
            if rows else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    if rollout:
        paired_rows = connection.execute(
            """SELECT p.model_version AS served_version,p.decision AS served_decision,
            s.model_version AS shadow_version,s.decision AS shadow_decision,e.label
            FROM predictions p JOIN prediction_shadows s USING(event_id)
            JOIN events e USING(event_id) WHERE p.scored_at>=?""",
            (since,),
        ).fetchall()
        by_version = {version: [] for version in versions}
        disagreements = 0
        count = 0
        for row in paired_rows:
            if {row["served_version"], row["shadow_version"]} != set(versions):
                continue
            count += 1
            disagreements += row["served_decision"] != row["shadow_decision"]
            if row["label"] is not None:
                by_version[row["served_version"]].append((row["label"], row["served_decision"]))
                by_version[row["shadow_version"]].append((row["label"], row["shadow_decision"]))
        metrics = {}
        for version, labeled in by_version.items():
            tp = sum(label == 1 and decision == "review" for label, decision in labeled)
            fp = sum(label == 0 and decision == "review" for label, decision in labeled)
            fn = sum(label == 1 and decision == "approve" for label, decision in labeled)
            metrics[version] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                                "tp": tp, "fp": fp, "fn": fn}
        result["paired"] = {"event_count": count, "labeled_count": len(by_version[baseline]),
                            "disagreement_rate": round(disagreements / count, 4) if count else None,
                            "versions": metrics}
    return result
