# Results on ml-latest-small

Run on 2026-11-20 06:01 UTC, TensorFlow 2.21.0, 1 GPU(s), Python 3.13.16.
Split sizes (rows): {'train': 85710, 'val': 8067, 'test': 7059}.
17 users evaluated, 100 candidates per user, index: faiss-hnsw.

## Offline ranking quality (test period)

| System | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | NDCG@50 | Recall@50 |
|---|---|---|---|---|---|---|
| popularity | 0.0526 | 0.0199 | 0.0647 | 0.0556 | 0.0849 | 0.1035 |
| retrieval_only | 0.1049 | 0.0465 | 0.1154 | 0.0839 | 0.1282 | 0.1509 |
| retrieval+dcn | 0.0966 | 0.0554 | 0.1174 | 0.1046 | 0.1320 | 0.1710 |
| retrieval+wide_deep | 0.0941 | 0.0561 | 0.1060 | 0.0901 | 0.1257 | 0.1608 |

Share of a user's test positives that appear anywhere in the 100 retrieved candidates: 0.2393.
Retrieval time: 0.51 ms per user.
The approximate index returns 98.8% of the candidates exact search would.

## Users with no history (3 seed ratings)

| Method | NDCG@10 | Recall@10 |
|---|---|---|
| popularity | 0.2416 | 0.0659 |
| content_based | 0.1324 | 0.0264 |

## A/B test

Simulated traffic: clicks are sampled from test-period positives with a position-bias model.

| Arm | Ranker | Requests | Click rate per request | Click rate per impression |
|---|---|---|---|---|
| A | dcn | 7 | 0.1429 | 0.0143 |
| B | wide_deep | 10 | 0.5000 | 0.0500 |

B minus A: +0.3571 (z = 1.52, p = 0.1294); 95% bootstrap interval -0.0436 to +0.7000.

## Interpretation and verification limits

This is a small-dataset proof of execution, not a MovieLens 25M-scale benchmark. The ranking comparison includes only 17 evaluated users; differences between models should not be generalized.

The A/B replay uses simulated traffic, not real users or a live experiment. Its statistical results do not demonstrate a real-world improvement in click rate.

The working Docker API currently uses the local Keras SavedModel backend with FAISS and Redis. Both Triton ranking models passed direct inference tests separately, but FastAPI-to-Triton integration has not been successfully verified.

The Airflow DAG structure was verified; a complete scheduled Airflow execution has not been demonstrated.
