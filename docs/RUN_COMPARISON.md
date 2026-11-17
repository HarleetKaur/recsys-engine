# Comparison of verified Colab experiments

Both runs executed the same project in different configurations. **They are not controlled experiments** because the data source, sample, evaluation users and training budgets differ; the figures should not be interpreted as a like-for-like improvement test.

| Measurement | MovieLens small | MovieLens 25M source, 2,000-user subset |
|---|---:|---:|
| Training ratings | 85,710 | 252,623 |
| Validation ratings | 8,067 | 23,782 |
| Test ratings | 7,059 | 20,953 |
| Users evaluated in offline ranking | 17 | 35 |
| Candidates per user | 100 | 100 |
| FAISS HNSW ANN overlap vs exact | 98.4% | 99.5% |
| Retrieval-only NDCG@20 | 0.1178 | 0.0959 |
| Retrieval + DCN NDCG@20 | 0.1247 | 0.0891 |
| Retrieval + Wide & Deep NDCG@20 | 0.1143 | 0.0878 |
| Best observed NDCG@20 | DCN, 0.1247 | Retrieval-only, 0.0959 |

## What we can conclude

- Both data pipelines completed and produced training, retrieval, ranking, and offline evaluation artifacts.
- On the **small** run, DCN obtained the highest NDCG@20 among the evaluated configurations.
- On the **25M-source subset**, retrieval-only obtained the highest NDCG@20; DCN and Wide & Deep had higher Recall@20 but lower NDCG@20.
- The 25M-source run used **only 2,000 users and 297,358 rated interactions**. Training on all 25M ratings has not been verified.
- The simulated A/B figures are not real-user click-through rates or a production A/B result.

Measured files: [`RESULTS.md`](RESULTS.md), [`RESULTS_ML25M_SUBSET.md`](RESULTS_ML25M_SUBSET.md), root [`reports/`](../reports/), and [`reports/ml25m_subset_2000/`](../reports/ml25m_subset_2000/).
