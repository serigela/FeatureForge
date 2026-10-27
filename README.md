# FeatureForge

An inspectable ML platform for transaction risk. It creates synthetic data, validates and stores events, computes past-only features, trains a model with a chronological split, serves predictions, and checks live feature drift. A durable broker handles retries and replay. Online feature snapshots can be compared with offline recomputation, and PostgreSQL can mirror the warehouse tables. Staged experiments and stable user-level canary routing make model changes observable. The core demo runs locally with Python's standard library and no credentials.

## Why this exists

Many ML demos stop at a notebook accuracy number. FeatureForge makes the surrounding engineering visible: event contracts, quarantine, at-least-once delivery, idempotency, training/serving feature parity, model versions, time-aware evaluation, replay-safe inference, and monitoring. The example is simulated fraud detection; it is **not** a production fraud decision system.

## Quick start

Requires Python 3.10+. The following path works without installing packages:

```bash
PYTHONPATH=src python3 -m featureforge.cli demo
PYTHONPATH=src python3 -m featureforge.cli serve
```

In another terminal:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"live_001","user_id":"user_0001","event_time":"2025-03-01T12:00:00+00:00","amount":750,"merchant_category":"travel","device_id":"unseen_device"}'
curl -s http://127.0.0.1:8080/monitor
```

Open `http://127.0.0.1:8080/` for the local operations dashboard.

`demo` saves generated data, the SQLite database, and versioned model files under `.featureforge/`. Rerunning it is safe: existing event IDs are counted as duplicates. Use `featureforge --data-dir /some/path demo` to change the data directory.

## System design

```mermaid
flowchart LR
  A[Synthetic JSONL events] --> B[Validation and quarantine]
  B --> C[(SQLite event store)]
  C --> D[Past-only feature definitions]
  D --> E[Chronological train / validation / test]
  E --> F[Versioned model registry]
  F --> G[Prediction API]
  F --> N[Staged experiment and canary routing]
  N --> G
  C --> G
  G --> H[(Prediction log)]
  H --> I[Drift and outcome monitoring]
  J[Durable event queue] --> K[Worker with retry and replay]
  K --> G
  G --> L[(Online feature snapshots)]
  C --> M[(Optional PostgreSQL warehouse)]
  H --> M
  L --> M
```

The same `features.compute` function runs in training and inference. It queries only events before the event being scored. Model threshold selection uses the validation period; the final test period is untouched until evaluation. The API stores each prediction by event ID and returns the original result on a replay.

The broker stores validated events before a worker scores them. A worker claims a time-limited lease; if it crashes, another worker can reclaim the event. A failure retries with backoff, then moves to a dead-letter state after five attempts. Replaying a delivered event returns its original prediction, so the side effect occurs once even though delivery can occur more than once. The `recovery-demo` command demonstrates this exact failure point.

## Commands

```bash
PYTHONPATH=src python3 -m featureforge.cli generate --count 3000 --seed 42
PYTHONPATH=src python3 -m featureforge.cli ingest .featureforge/synthetic_transactions.jsonl
PYTHONPATH=src python3 -m featureforge.cli train
PYTHONPATH=src python3 -m featureforge.cli serve --host 127.0.0.1 --port 8080
PYTHONPATH=src python3 -m featureforge.cli monitor --window 500
PYTHONPATH=src python3 -m featureforge.cli promote MODEL_VERSION
PYTHONPATH=src python3 -m featureforge.cli recovery-demo
PYTHONPATH=src python3 -m featureforge.cli queue-stats
PYTHONPATH=src python3 -m featureforge.cli parity
PYTHONPATH=src python3 -m featureforge.cli models
PYTHONPATH=src python3 -m featureforge.cli train --stage
PYTHONPATH=src python3 -m featureforge.cli canary MODEL_VERSION --percent 10
PYTHONPATH=src python3 -m featureforge.cli rollout
PYTHONPATH=src python3 -m featureforge.cli rollback
```

`train` prints validation and test precision, recall, F1, confusion counts, and test average precision. The test set is the newest 20% of labeled events. The baseline is regularized logistic regression implemented locally for transparent operation and zero runtime dependencies. Performance figures from synthetic data should never be presented as real-world fraud detection results.

## Experiments and staged rollout

Every training run saves its model, dataset fingerprint, epoch count, threshold, and time-split evaluation. `models` lists these experiments. By default, `train` activates the new version; use `train --stage` to keep the current model live. The first trained model becomes active even with `--stage` because serving needs a baseline.

After reviewing a staged version, run `canary MODEL_VERSION --percent 10`. The same user is always routed to the same version during that rollout, based on a stable hash of `user_id`. An event replay keeps its original prediction and model version. `rollout` and `GET /rollout` show prediction counts, labeled counts, review rate, observed precision, and recall by version since the current rollout began. `rollback` stops routing new users to the candidate while preserving existing prediction records; `promote MODEL_VERSION` makes a version active for all new predictions and ends the canary. `GET /health` reflects a promotion without restarting the API.

These outcome figures are descriptive. Small labeled samples and different user populations can make a canary comparison misleading. The project does not automatically promote or roll back a model, and the API is intended for local demonstration only.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Active model and health |
| `POST /predict` | Validate, score, and record a new event |
| `GET /metrics` | Prediction decision counts |
| `GET /monitor` | Population stability report for scored features |
| `POST /feedback` | Attach a delayed `0` or `1` outcome to a scored event |
| `POST /events` | Queue an event for asynchronous scoring |
| `GET /events/{event_id}` | Read queue state and attempt count |
| `GET /queue` | Queue counts by state |
| `POST /replay` | Requeue a dead event by `event_id` |
| `GET /features/{user_id}` | Latest online feature snapshot and age |
| `GET /parity` | Compare online features with offline recomputation |
| `GET /models` | List model experiments and offline evaluation |
| `GET /rollout` | Show traffic split and observed outcomes by model version |
| `GET /` | Local operations dashboard |

`POST /predict` requires `event_id`, `user_id`, timezone-aware ISO 8601 `event_time`, positive `amount`, `merchant_category`, and `device_id`. It returns a risk score and a `review` or `approve` decision. For the demo, use an event timestamp after the synthetic history.

After an outcome is known, submit `{"event_id":"live_001","label":1}` to `/feedback`. The monitor then reports observed precision and recall for labeled predictions. Repeating the same feedback is safe; changing an existing label is rejected.

For asynchronous scoring, send the same event object to `/events` and run `PYTHONPATH=src python3 -m featureforge.cli worker` in another terminal. The request returns `202` and an event ID. Check `/events/{event_id}` for `done`, `retry`, or `dead`. Run `worker --once` to process only one ready message. To retry a dead event, send `{"event_id":"..."}` to `/replay`.

## PostgreSQL warehouse

The operational store and online snapshot remain local SQLite. PostgreSQL is an optional warehouse mirror for event history, feature snapshots, and predictions. Sync is idempotent and propagates delayed outcome labels.

```bash
docker compose up -d postgres
python3 -m pip install -e '.[postgres]'
export FEATUREFORGE_POSTGRES_DSN='postgresql://featureforge:localdev@127.0.0.1:5433/featureforge'
featureforge sync-postgres
featureforge verify-postgres
```

The compose service binds only to local loopback. Its credentials are for local development; use a separate secret and restricted access for any deployment. `verify-postgres` compares row counts in a dedicated mirror database. `parity` checks each user's latest online snapshot against a fresh computation from event history and reports missing or stale records.

## Run checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

CI runs the tests on each push and pull request, including a PostgreSQL service for warehouse sync. To run the API in a container, build the Dockerfile and mount a prepared `.featureforge` directory at `/app/.featureforge`.

## Current limits and next phases

The broker and online store use SQLite so the core system runs on a laptop; PostgreSQL is a separate warehouse mirror. The HTTP API has no authentication and should be kept on localhost. Outcome monitoring depends on delayed ground-truth labels and is empty until feedback arrives. The synthetic generator is intentionally simplistic.

Next substantial phases:

1. Add automatic rollback guardrails backed by sufficient labeled outcomes and latency charts to the dashboard.
2. Add a real public dataset with documented licensing and a reproducible benchmark. Keep personal or payment data out of the repository.
3. Replace the local broker with a distributed event system and add a high-volume load and recovery benchmark.

## Repository structure

```text
src/featureforge/  generator, ingestion, broker, features, training, rollout, API, stores, monitoring, CLI
tests/             pipeline, recovery, parity, rollout, HTTP, and PostgreSQL checks
.github/workflows/ CI with PostgreSQL service
compose.yaml       local PostgreSQL warehouse
Dockerfile         containerized API
```
