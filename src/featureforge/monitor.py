"""Population stability monitoring against the active model's training reference."""

import math

from .model import load


def _bucket(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def report(connection, registry, window: int = 500) -> dict:
    model = load(registry)
    rows = connection.execute(
        "SELECT features_json,score FROM predictions ORDER BY scored_at DESC LIMIT ?", (window,)
    ).fetchall()
    import json
    features = [json.loads(row["features_json"]) for row in rows]
    if not features:
        return {"model_version": model["version"], "sample_count": 0, "status": "insufficient_data",
                "outcomes": {"labeled_count": 0}}
    results = {}
    for name, reference_bins in model["reference_bins"].items():
        edges = reference_bins["edges"]
        counts = [0] * 10
        for item in features:
            counts[_bucket(item[name], edges)] += 1
        observed = [(count + 0.5) / (len(features) + 5) for count in counts]
        reference = reference_bins["proportions"]
        psi = sum((value - baseline) * math.log(value / baseline)
                  for value, baseline in zip(observed, reference))
        results[name] = round(psi, 4)
    worst = max(results.values())
    outcomes = connection.execute(
        """SELECT e.label,p.decision FROM predictions p JOIN events e USING(event_id)
        WHERE e.label IS NOT NULL ORDER BY p.scored_at DESC LIMIT ?""", (window,)
    ).fetchall()
    tp = sum(row["label"] == 1 and row["decision"] == "review" for row in outcomes)
    fp = sum(row["label"] == 0 and row["decision"] == "review" for row in outcomes)
    fn = sum(row["label"] == 1 and row["decision"] == "approve" for row in outcomes)
    online = {"labeled_count": len(outcomes), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
              "recall": round(tp / (tp + fn), 4) if tp + fn else None,
              "tp": tp, "fp": fp, "fn": fn}
    return {"model_version": model["version"], "sample_count": len(features),
            "feature_psi": results, "max_psi": worst,
            "status": "alert" if len(features) >= 100 and worst >= 0.2 else "ok" if len(features) >= 100 else "insufficient_data",
            "outcomes": online}
