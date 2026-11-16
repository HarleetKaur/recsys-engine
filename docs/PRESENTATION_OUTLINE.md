# Ten-minute presentation — verified small-mode run (16 November 2026)

A ready-to-present slide deck is at `docs/Recommendation_Engine_10min.pptx`. Use these notes to rehearse; edit your own personal reflection and demo recording rather than inventing an experience.

1. **The problem (45 sec):** A two-stage recommender retrieves candidates before applying more expensive ranking.
2. **System design (60 sec):** time split → features → TF Recommenders towers → FAISS → DCN/Wide & Deep → cached FastAPI recommender. Triton/Airflow are included as export/DAG infrastructure, not deployed services.
3. **Data and protocol (60 sec):** `ml-latest-small`, 85,710 train, 8,067 validation, 7,059 test interactions; test evaluation on 17 users. Ratings >= 4 are positives.
4. **Retrieval (50 sec):** 100 candidates; FAISS HNSW approximate-vs-exact recall ~98.4% on offline checks.
5. **Offline results (90 sec):** show `reports/ndcg.png`. NDCG@20: popularity 0.0647; retrieval-only 0.1178; retrieval+DCN 0.1247; retrieval+Wide & Deep 0.1143. Recall@20: 0.0556 / 0.0839 / 0.1098 / 0.0953.
6. **Cold-start and API (60 sec):** popularity and genre-based fallback. In-process FastAPI test returned personalised, cached and new-user results; Redis cache worked.
7. **Simulated A/B replay (60 sec):** A=DCN 7 requests and B=Wide & Deep 10 requests, with synthetic clicks. There is no real-user effect estimate.
8. **Testing and verification (45 sec):** 20 source unit tests independently; Colab small pipeline including Spark and FAISS; Triton export but no live Triton; DAG source only.
9. **Limitations (45 sec):** only 17 evaluation users, MovieLens ratings rather than clicks, training feature-table limitation, no production deployment.
10. **Demo and next steps (85 sec):** show a short screen recording of `/recommend`, a repeated cached response, and cold start. Say that 25M-scale run, Triton serving and real-user experiments are follow-ups.

## Individual reflection (write in your own words)

- Which parts did you execute and inspect yourself? What error or decision did you actually encounter?
- What do you now understand about two-stage retrieval vs ranking, or about evaluating a time split?
- What would you improve with more time? Do not add invented debugging stories or claim an unperformed deployment.
