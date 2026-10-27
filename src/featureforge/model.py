"""Dependency-free logistic baseline, chronological evaluation, and model registry."""

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .features import FEATURE_NAMES, labeled_rows


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _vectors(rows: list[tuple[dict, int]]) -> tuple[list[list[float]], list[int]]:
    return [[features[name] for name in FEATURE_NAMES] for features, _ in rows], [label for _, label in rows]


def _fit_scaler(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(FEATURE_NAMES)
    means = [sum(row[j] for row in vectors) / len(vectors) for j in range(width)]
    scales = [
        max(1e-6, math.sqrt(sum((row[j] - means[j]) ** 2 for row in vectors) / len(vectors)))
        for j in range(width)
    ]
    return means, scales


def _transform(vector: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - mean) / scale for value, mean, scale in zip(vector, means, scales)]


def score(model: dict, features: dict[str, float]) -> float:
    vector = [features[name] for name in model["feature_names"]]
    scaled = _transform(vector, model["means"], model["scales"])
    return _sigmoid(model["bias"] + sum(weight * value for weight, value in zip(model["weights"], scaled)))


def _classification(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = sum(label == 1 and value >= threshold for label, value in zip(labels, scores))
    fp = sum(label == 0 and value >= threshold for label, value in zip(labels, scores))
    fn = sum(label == 1 and value < threshold for label, value in zip(labels, scores))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "positive_rate": round(sum(labels) / len(labels), 4)}


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), reverse=True)
    correct = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            correct += 1
            total += correct / rank
    return round(total / positives, 4)


def _quantile_edges(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, int(len(ordered) * index / 10))] for index in range(1, 10)]


def train(connection, registry: str | Path, epochs: int = 350, activate: bool = True) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    rows = labeled_rows(connection)
    if len(rows) < 100:
        raise ValueError("at least 100 labeled events are required")
    first = int(len(rows) * 0.6)
    second = int(len(rows) * 0.8)
    training, validation, testing = rows[:first], rows[first:second], rows[second:]
    x_train, y_train = _vectors(training)
    means, scales = _fit_scaler(x_train)
    transformed = [_transform(row, means, scales) for row in x_train]
    weights = [0.0] * len(FEATURE_NAMES)
    prevalence = max(0.001, min(0.999, sum(y_train) / len(y_train)))
    bias = math.log(prevalence / (1 - prevalence))
    rate = 0.08
    for _ in range(epochs):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for vector, label in zip(transformed, y_train):
            difference = _sigmoid(bias + sum(a * b for a, b in zip(weights, vector))) - label
            bias_gradient += difference
            for index, value in enumerate(vector):
                gradients[index] += difference * value
        count = len(transformed)
        bias -= rate * bias_gradient / count
        weights = [weight - rate * (gradient / count + 0.001 * weight)
                   for weight, gradient in zip(weights, gradients)]
    model = {"feature_names": list(FEATURE_NAMES), "weights": weights, "bias": bias,
             "means": means, "scales": scales}
    val_scores = [score(model, feature) for feature, _ in validation]
    val_labels = [label for _, label in validation]
    candidates = [i / 100 for i in range(5, 81, 5)]
    threshold = max(candidates, key=lambda t: (_classification(val_labels, val_scores, t)["f1"], t))
    test_scores = [score(model, feature) for feature, _ in testing]
    test_labels = [label for _, label in testing]
    metrics = {
        "train_count": len(training), "validation_count": len(validation), "test_count": len(testing),
        "validation": _classification(val_labels, val_scores, threshold),
        "test": _classification(test_labels, test_scores, threshold),
        "test_average_precision": _average_precision(test_labels, test_scores),
    }
    model["threshold"] = threshold
    model["metrics"] = metrics
    model["epochs"] = epochs
    fingerprint_rows = connection.execute(
        """SELECT event_id,user_id,event_time,amount,merchant_category,device_id,label
        FROM events ORDER BY event_time,event_id"""
    ).fetchall()
    model["dataset_sha256"] = hashlib.sha256(json.dumps(
        [tuple(row) for row in fingerprint_rows],
        separators=(",", ":")
    ).encode()).hexdigest()
    model["trained_at"] = datetime.now(timezone.utc).isoformat()
    model["reference_bins"] = {}
    for name in FEATURE_NAMES:
        values = [features[name] for features, _ in training]
        edges = _quantile_edges(values)
        counts = [0] * 10
        for value in values:
            counts[next((i for i, edge in enumerate(edges) if value <= edge), 9)] += 1
        model["reference_bins"][name] = {
            "edges": edges,
            "proportions": [(count + 0.5) / (len(values) + 5) for count in counts],
        }
    raw = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    version = hashlib.sha256(raw).hexdigest()[:12]
    model["version"] = version
    directory = Path(registry)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"model-{version}.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    if activate:
        promote(directory, version)
    elif not (directory / "active.json").exists():
        _write_pointer(directory, version)
    return model


def load(registry: str | Path) -> dict:
    directory = Path(registry)
    pointer = json.loads((directory / "active.json").read_text(encoding="utf-8"))
    return load_version(directory, pointer["version"])


def load_version(registry: str | Path, version: str) -> dict:
    directory = Path(registry)
    if not isinstance(version, str) or not all(c in "0123456789abcdef" for c in version):
        raise ValueError("invalid model version")
    model = json.loads((directory / f"model-{version}.json").read_text(encoding="utf-8"))
    if model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("model feature schema does not match serving code")
    if model.get("version") != version:
        raise ValueError("model version does not match registry entry")
    return model


def list_models(registry: str | Path) -> list[dict]:
    directory = Path(registry)
    active = load(directory)["version"]
    models = []
    for path in directory.glob("model-*.json"):
        version = path.stem.removeprefix("model-")
        candidate = load_version(directory, version)
        models.append({"version": version, "active": version == active,
                       "trained_at": candidate["trained_at"],
                       "epochs": candidate.get("epochs"),
                       "dataset_sha256": candidate.get("dataset_sha256"),
                       "metrics": candidate["metrics"]})
    return sorted(models, key=lambda item: item["trained_at"], reverse=True)


def _write_pointer(directory: Path, version: str) -> None:
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".active-", delete=False,
                                     encoding="utf-8") as stream:
        json.dump({"version": version}, stream)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, directory / "active.json")


def promote(registry: str | Path, version: str) -> None:
    directory = Path(registry)
    load_version(directory, version)
    _write_pointer(directory, version)
    (directory / "rollout.json").unlink(missing_ok=True)
