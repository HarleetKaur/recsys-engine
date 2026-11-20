# Verification status — updated 20 November 2026

## Additional verification — 20 November 2026

- Direct Triton inference: both DCN and Wide & Deep models loaded, returned numerical scores, and passed two-item batch requests.
- Local FastAPI serving: HTTP 200 for health, personalized recommendations, and cold-start requests.
- Redis-backed caching: verified identical cached recommendations and a Redis cache hit.
- FastAPI-to-Triton ranking: attempted but failed; the verified API configuration uses Keras.
- Full Airflow scheduler execution and real-world A/B testing remain unverified.


This file distinguishes **observed Colab run outputs**, separate local source checks, and infrastructure that remains unverified. The source for the 25M-subset numbers is the user's uploaded `results_bundle (4).zip`; all JSON values were cross-checked against its generated `docs/RESULTS.md` and `reports/metrics.json`.

## Completed: small-mode Colab run

- MovieLens `ml-latest-small`: **85,710 train**, **8,067 validation**, **7,059 test** ratings; **17 users evaluated** in offline rankings.
- TensorFlow Recommenders two-tower retrieval, FAISS HNSW, DCN and Wide & Deep, offline evaluation, simulated A/B replay, Spark smoke test and Triton model export were executed successfully, as supported by provided Colab screenshots.
- Best NDCG@20 was **0.1247 for retrieval + DCN**, from this small run alone.
- Colab in-process FastAPI TestClient demonstrated `/health`, personalised `/recommend`, cached repeat requests through Redis and new-user recommendations. A short **screen recording was shown in QuickTime** (the recording file itself is not bundled in this GitHub update).
- Original result files remain at `reports/` and `docs/RESULTS.md`.

## Completed: MovieLens 25M source, 2,000-user subset

- `MODE=full` was run in a new Colab notebook using **`data.max_users=2000`**, **Spark**, 2 retrieval training epochs and 1 ranker epoch.
- The real output reports **252,623 train**, **23,782 validation**, **20,953 test** ratings; **35 users evaluated**, 100 candidates per user, 1 GPU, TensorFlow **2.21.0**.
- FAISS HNSW approximate-vs-exact candidate overlap **99.5%**; offline FAISS retrieval time about **0.70 ms/user**. Neither is measured end-to-end API latency.
- Best NDCG@20 is **retrieval-only 0.0959**. DCN **0.0891** and Wide & Deep **0.0878** did **not** improve NDCG@20; their Recall@20 results were 0.1001 and 0.1040 versus 0.0880 for retrieval-only.
- The simulated A/B replay returned 18 DCN and 17 Wide & Deep requests, z-test p≈0.612; its synthetic clicks do not establish real-world lift.
- Final Colab stage (`export_triton`) and ZIP creation succeeded as shown in screenshot; the ZIP contains metadata/configuration, **not trained SavedModel weights**.
- Reported results and training histories preserved under `reports/ml25m_subset_2000/`, with `docs/RESULTS_ML25M_SUBSET.md` and `docs/RUN_PROVENANCE_ML25M_SUBSET.json`.

## Independent code verification

- The project sources plus the cold-start cache-isolation fix have **21 local Python unit tests**; these were reported as passing in the assistant's environment. This does not verify external infrastructure.
- The user pushed the corrected code, notebook, and regression test to GitHub in commit `004a84c`, and the presentation update in `9840b77` (per screenshots).

## Still not verified or not supplied

- **All 25 million MovieLens ratings:** the completed `ml-25m` run uses a 2,000-user sample and 297,358 total ratings, not the full corpus.
- **Running Airflow scheduler**, **publicly deployed web API**, or **real-user A/B testing**: not tested. The Airflow DAG and serving code are implementation components, not evidence of live infrastructure.
- **Individual intern reflection:** should be supplied truthfully by the project owner.

## Measurement caveats

- Ratings >= 4 are used as proxy positives; MovieLens is not click data.
- NDCG/Recall on 17 and 35 offline evaluation users cannot establish production performance or generalization to the full dataset.
- The known training feature-table issue is documented in the README; the test time window is held out for evaluation.
- The two Colab runs use different samples and budgets: **do not interpret their differences as a causal model improvement**.

## Latest integration verification — 20 November 2026

Live FastAPI-to-Triton inference: PASSED for DCN and Wide & Deep.
Inference count increases: 200 per model.
A/B reporting endpoint: PASSED; real-user experimentation not verified.
Airflow Docker configuration: PASSED; scheduled execution not verified.
