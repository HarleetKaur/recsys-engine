# Additional integration work — 20 November 2026

This patch extends the existing Recommendation Engine. It **does not** include trained weights,
replace the user's newer evaluation results, or prove that a real-user experiment happened.
The stable Keras/Redis/FAISS API remains the default.

## 1. FastAPI → Triton inference (opt-in)

The Triton ranker now uses the standard-library HTTP V2 API rather than sharing a
`tritonclient.http`/gevent client across FastAPI worker threads. Both inputs use shape
`[batch,1]`, datatype `INT32`, and the requested output is `score` (`FP32`). HTTP
connections are created per inference request. The scorer obtains model names from metadata
and checks that each score batch matches the request length.

On the **existing Mac** with trained artifacts and the Triton container already running:

```bash
cd "$HOME/Downloads/recsys-trained-restored/recsys-engine"
TRITON_TAG=24.05-py3 docker compose -f docker-compose.yml -f docker-compose.triton.yml up -d --no-deps --build api
python3 scripts/verify_live_triton_api.py
```

Expected: `/health` includes `"backend":"triton"` and the verification prints
`PASS: FastAPI -> Triton DCN AND Wide & Deep inference verified` when both model
inference counters have increased. This **requires your running Mac stack**; the
repository patch alone cannot establish the runtime outcome. If the API fails, inspect
`TRITON_TAG=24.05-py3 docker compose logs --tail=80 api`.

Safe fallback to the already-verified Keras backend (keeps models/data intact):

```bash
TRITON_TAG=24.05-py3 docker compose -f docker-compose.yml up -d --no-deps --force-recreate api
curl -fsS http://localhost:8080/health
```

Expected: health contains `"backend":"keras"` and `"cache":"redis"`.
Do not claim the Triton path is verified until the live verification script passes.

## 2. Weekly Airflow retraining

The Airflow DAG has ten sequential tasks: download, split, features, train_retrieval,
build_index, train_ranker, evaluate, ab_replay, export_triton and flush_cache. The final
task clears **only `rec:*` Redis recommendation keys**, and only after upstream tasks succeed.
It runs weekly with `catchup=False` and limits concurrent DAG runs to one.

A separate Compose overlay provides an Airflow 2.10.5 `standalone` scheduler / UI at
`http://localhost:8081`. The image uses an isolated TensorFlow/TFRS/PySpark virtual
environment, plus a Java runtime. Building this image requires Internet access, sufficient
disk/RAM, and may take a long time. Use the existing project's real data/artifacts.

```bash
TRITON_TAG=24.05-py3 docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d --build airflow
TRITON_TAG=24.05-py3 docker compose -f docker-compose.yml -f docker-compose.airflow.yml logs --tail=60 airflow
```

Check the DAG via the Airflow UI at `http://localhost:8081`, unpause it and trigger a
manual run before relying on automatic scheduling. The scheduler starts with DAGs paused.
Airflow `standalone` prints its generated administrator password in its startup logs.
Weekly automatic operation **is not proven** until the Airflow scheduler stays up and a
complete DAG run finishes successfully. A full rerun may take several hours and overwrite
model artifacts. Set appropriate compute limits; do not trigger during a submission deadline.

After retraining, the serving API and Triton containers require a controlled model reload
or restart before they use the newly exported model versions. The DAG does **not** claim
zero-downtime automatic model promotion. Local Airflow Docker functionality has not been
verified in the assistant's runtime (Docker/GPU and full datasets are unavailable).

## 3. Live-request A/B instrumentation (not genuine traffic)

Requests are assigned deterministically by user ID to A (DCN) or B (Wide & Deep).
For eligible known-user requests, including cached repeat requests, the API records
impressions with a unique `request_id` in `logs/impressions.jsonl`. The endpoint:

```http
POST /feedback
Content-Type: application/json

{"request_id":"<ID_FROM_RECOMMEND>","movie_id":1028,"event":"click"}
```

validates that the request exists and the clicked movie was actually shown, deduplicates
repeat feedback, and records `logs/clicks.jsonl`. Use:

```bash
curl -fsS http://localhost:8080/ab/report
```

It returns per-arm request/impression click rates and statistical calculations with a
sample-sufficiency flag. For a one-arm or empty sample, the confidence interval is null
instead of invalid JSON NaN. `traffic_source: observed_api_logs` refers only to HTTP
calls and submitted events; it **does not authenticate** whether these were real people.

**A genuine online A/B experiment is not complete** until actual consented user traffic
and independently verified click events have been collected over a suitable period.
Never fabricate clicks or treat manual curl tests as a live experiment. `stage_ab_replay`
is explicitly an offline simulation and remains separate from the observed-requests report.

## Local verification performed in the patch environment

`PYTHONPATH=src:tests python3 -m pytest -q` ran all 25 tests successfully, including
HTTP metadata/inference protocol contract tests, multi-threaded scorer calls,
FastAPI feedback validation, and an Airflow DAG structural test. These tests use mocks
and do not establish successful live Triton inference, a running Airflow scheduler,
or real-user A/B traffic.

## How to merge without losing newer GitHub changes

Use the separate `recsys-advanced-integration-patch.zip` and its `apply_update.sh`
script. The script checks whether the patch applies cleanly and does not replace
results, notebook, presentation, README, config settings or existing model weights.
Review changes, run tests, then commit and push only after the live checks you need pass.
