# Results on ml-latest-small

Run on 2026-11-16 08:30 UTC, TensorFlow 2.21.0, 1 GPU(s), Python 3.13.16.
Split sizes (rows): {'train': 85710, 'val': 8067, 'test': 7059}.
17 users evaluated, 100 candidates per user, index: faiss-hnsw.

## Offline ranking quality (test period)

| System | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | NDCG@50 | Recall@50 |
|---|---|---|---|---|---|---|
| popularity | 0.0526 | 0.0199 | 0.0647 | 0.0556 | 0.0849 | 0.1035 |
| retrieval_only | 0.1073 | 0.0465 | 0.1178 | 0.0839 | 0.1282 | 0.1493 |
| retrieval+dcn | 0.1029 | 0.0599 | 0.1247 | 0.1098 | 0.1349 | 0.1690 |
| retrieval+wide_deep | 0.1018 | 0.0606 | 0.1143 | 0.0953 | 0.1331 | 0.1657 |

Share of a user's test positives that appear anywhere in the 100 retrieved candidates: 0.2405.
Retrieval time: 0.50 ms per user.
The approximate index returns 98.4% of the candidates exact search would.

## Users with no history (3 seed ratings)

| Method | NDCG@10 | Recall@10 |
|---|---|---|
| popularity | 0.2416 | 0.0659 |
| content_based | 0.1324 | 0.0264 |

## A/B test

Simulated traffic: clicks are sampled from test-period positives with a position-bias model.

| Arm | Ranker | Requests | Click rate per request | Click rate per impression |
|---|---|---|---|---|
| A | dcn | 7 | 0.0000 | 0.0000 |
| B | wide_deep | 10 | 0.5000 | 0.0600 |

B minus A: +0.5000 (z = 2.23, p = 0.0260); 95% bootstrap interval +0.2000 to +0.8000.

## How to interpret these results

This is a **small-dataset proof of execution**, not a 25M-scale benchmark. Only **17 test-period users** were included in the ranking comparison; avoid generalizing the differences between rankers. The A/B replay has only 17 simulated requests in total. Its p-value and confidence interval are calculated by the project's demo statistics code and are **not evidence of real-world click improvement**. Retrieval timing refers to the offline search function, not a deployed API. Triton repository export is not a Triton inference test.
