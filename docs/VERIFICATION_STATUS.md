# Verification status — updated 16 November 2026

This record distinguishes the user's **Google Colab execution**, independent local source checks, and parts that remain unverified. The Colab results were supplied as `results_bundle.zip`; the API evidence was supplied as screenshots. We have not claimed a live production deployment.

## Verified in Google Colab (small mode)

- **Dataset:** `ml-latest-small`, 85,710 train, 8,067 validation, and 7,059 test interactions. Global time split; 17 users contributed to the offline ranking evaluation.
- **Runtime:** Linux x86_64, Python 3.13.16, TensorFlow 2.21.0, one available GPU (as recorded in `reports/metrics.json`).
- **Stages executed successfully (per Colab screenshots):** package installation, unit and synthetic smoke checks (including Spark), data preparation, TensorFlow two-tower training, FAISS HNSW index construction, both ranker models, offline evaluation, simulated A/B replay, and Triton export.
- **In-process API demo (per Colab screenshots):** `/health` returned `ok`, first `/recommend` returned personalised movie results, repeated call returned `cache` (Redis), and an unknown user returned `cold_content` recommendations. This is **FastAPI TestClient running in-process**, not a publicly deployed API.
- **Generated files checked in `results_bundle.zip`:** `reports/metrics.json`, `reports/ndcg.png`, training histories, `reports/ab_replay.json`, `docs/RESULTS.md`, and `triton/model_repository` configuration files plus `EXPORT_INFO.json`.

## Independent source checks

- 20 Python unit tests passed on the reviewed project code in the assistant's local environment (including one additional regression test). These unit tests do not establish that all external services work.
- The generated results JSON was checked for consistency with `docs/RESULTS.md` and the Colab screenshots.

## Not completed or not verified

- **Full-scale run:** `MODE = "full"` with MovieLens 25M and Spark has not been run. Results here are from the smaller dataset and must not be presented as 25M results.
- **Triton live inference:** a model repository was exported in Colab, but Triton Server was not started or tested. The submission ZIP includes lightweight configuration/metadata, **not SavedModel weights**. A separate compatible x86 Linux/Docker setup and matching TF backend are required.
- **Airflow:** DAG source exists; an Airflow scheduler has not run it.
- **Real A/B testing:** the replay uses simulated clicks and only 7 arm-A and 10 arm-B requests. Do not interpret its p-value as evidence of real user behaviour or product lift.
- **Deployed public endpoint, recording and GitHub publication:** not verified in this archive. A working in-process demonstration is verified by screenshots.

## Limits on reported performance

- NDCG/Recall are **offline test-set measures**, measured for only 17 users. Results may vary with data sampling, training random seeds and hardware.
- HNSW approximate-vs-exact recall is an offline index comparison, **not Triton serving accuracy**. `retrieval_ms_per_user` is an offline retrieval timing, **not end-to-end HTTP latency**.
- The project README documents known training feature leakage affecting training statistics. The temporal test split remains held out for evaluation.
