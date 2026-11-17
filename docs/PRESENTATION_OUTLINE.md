# Ten-minute presentation — two verified Colab runs (16 November 2026)

Use `docs/Recommendation_Engine_10min.pptx` or `.pdf` for a short technical walkthrough. The main results on slides 4-8 are from a **MovieLens 25M-source, 2,000-user subset**, not all 25 million ratings. The earlier small-run results are preserved in `docs/RESULTS.md`.

1. **Problem (45 sec)** — Fast retrieval narrows candidates, then a more expensive ranker reorders them.
2. **Architecture (60 sec)** — Time split, Spark features, TF Recommenders towers, FAISS HNSW, DCN/Wide & Deep, FastAPI/Redis. Triton config was exported; Airflow DAG supplied, neither deployed.
3. **Dataset (65 sec)** — ml-25m source, capped at 2,000 users: 252,623 train, 23,782 validation and 20,953 test ratings. Only 35 users in offline ranking metrics. State clearly this is **not** a full-25M training run.
4. **Retrieval (55 sec)** — 100 candidates/user, FAISS HNSW approximate-vs-exact overlap 99.5%, offline retrieval ~0.70 ms/user, not HTTP latency.
5. **Ranking (90 sec)** — NDCG@20: popularity 0.0739, retrieval-only **0.0959**, DCN 0.0891, Wide & Deep 0.0878. Deep rankers did not beat retrieval-only on this measure. Recall@20 DCN 0.1001 and Wide & Deep 0.1040 versus retrieval-only 0.0880.
6. **Cold start (55 sec)** — popularity NDCG@10 0.2885 and content-based 0.1404 on 119 cold-start evaluations. The earlier small-run in-process API demo validated Redis and cold start.
7. **Simulated A/B (55 sec)** — 18 DCN requests (16.7% request CTR) vs 17 Wide & Deep (23.5%), all clicks simulated; p≈0.612. No claim of real-user lift.
8. **Tests and limits (60 sec)** — 21 local unit tests, two executed Colab paths, recorded API demo. Do not claim Triton live serving, running Airflow or public deployment.
9. **Demo (45 sec)** — Show the saved six-second `/recommend` screen recording, including cached repeat and new-user response; disclose that it was recorded with the small-run model.
10. **Next work (45 sec)** — Larger-scale evaluation, ranker tuning, tested Triton/Airflow deployment, and real traffic experiments.

## Honest individual reflection

Write a few sentences in your own words about what you personally executed in Colab, what you observed from the repeated API demo and cache issue, what you learned about retrieval versus ranking, and what you would improve. Do not invent tasks you did not perform.
