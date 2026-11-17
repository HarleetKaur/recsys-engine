# MovieLens 25M source: 2,000-user subset experiment

**Scope:** This report was generated from a Google Colab run on 16 November 2026. The data source was MovieLens `ml-25m`, but the experiment limited the dataset to **up to 2,000 users** (`data.max_users=2000`). The time split contains **297,358 ratings**, not 25 million. Do not call this full-dataset training.

**Training budget:** Retrieval model trained for 2 epochs; each ranking model trained for 1 epoch. Results cover only **35 evaluated users** and may change with longer tuning, more users or different seeds. The rankings are evaluated offline using MovieLens ratings (4+ counted as positive), not real clicks.

**Outcome:** Retrieval-only has the highest NDCG@20 (0.0959); DCN and Wide & Deep do not improve this metric in this run. They slightly improve Recall@20 instead. The A/B replay samples simulated clicks; `p = 0.6119` is not evidence of real-user lift.

**Deployment status:** Model repository configuration was exported but live Triton was **not** tested. Airflow scheduling and a public endpoint were not verified. Large model weights are not included in GitHub.

**The earlier, independently successful small-mode run is preserved** in [`RESULTS.md`](RESULTS.md) and the root `reports/` files. These are separate runs with different training populations; do not merge their statistics.

---

# Results on ml-25m

Run on 2026-11-16 15:05 UTC, TensorFlow 2.21.0, 1 GPU(s), Python 3.13.16.
Split sizes (rows): {'train': 252623, 'val': 23782, 'test': 20953}.
35 users evaluated, 100 candidates per user, index: faiss-hnsw.

## Offline ranking quality (test period)

| System | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | NDCG@50 | Recall@50 |
|---|---|---|---|---|---|---|
| popularity | 0.0650 | 0.0256 | 0.0739 | 0.0653 | 0.0848 | 0.1175 |
| retrieval_only | 0.0924 | 0.0701 | 0.0959 | 0.0880 | 0.1270 | 0.1780 |
| retrieval+dcn | 0.0866 | 0.0731 | 0.0891 | 0.1001 | 0.1075 | 0.1611 |
| retrieval+wide_deep | 0.0820 | 0.0719 | 0.0878 | 0.1040 | 0.1024 | 0.1593 |

Share of a user's test positives that appear anywhere in the 100 retrieved candidates: 0.2484.
Retrieval time: 0.70 ms per user.
The approximate index returns 99.5% of the candidates exact search would.

## Users with no history (3 seed ratings)

| Method | NDCG@10 | Recall@10 |
|---|---|---|
| popularity | 0.2885 | 0.0544 |
| content_based | 0.1404 | 0.0230 |

## A/B test

Simulated traffic: clicks are sampled from test-period positives with a position-bias model.

| Arm | Ranker | Requests | Click rate per request | Click rate per impression |
|---|---|---|---|---|
| A | dcn | 18 | 0.1667 | 0.0167 |
| B | wide_deep | 17 | 0.2353 | 0.0412 |

B minus A: +0.0686 (z = 0.51, p = 0.6119); 95% bootstrap interval -0.2092 to +0.3530.
