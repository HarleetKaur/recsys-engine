"""Ranking metrics with binary relevance."""
import numpy as np


def recall_at_k(recs, truth, k):
    if not truth:
        return 0.0
    return len(set(recs[:k]) & set(truth)) / len(truth)


def hit_rate_at_k(recs, truth, k):
    return float(len(set(recs[:k]) & set(truth)) > 0)


def precision_at_k(recs, truth, k):
    return len(set(recs[:k]) & set(truth)) / k


def ndcg_at_k(recs, truth, k):
    truth = set(truth)
    if not truth:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(recs[:k]) if item in truth)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(truth), k)))
    return float(dcg / idcg)


def evaluate_lists(rec_lists, truth_lists, ks=(10, 20, 50)):
    """rec_lists/truth_lists: parallel lists (one entry per user). Returns mean metrics."""
    out = {"n_users": len(rec_lists)}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(r, t, k) for r, t in zip(rec_lists, truth_lists)]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(r, t, k) for r, t in zip(rec_lists, truth_lists)]))
        out[f"hit_rate@{k}"] = float(np.mean([hit_rate_at_k(r, t, k) for r, t in zip(rec_lists, truth_lists)]))
    return out
