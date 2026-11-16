"""Offline evaluation on the test period: popularity baseline, retrieval alone, and retrieval + each ranker."""
import time

import numpy as np

from recsys.evaluation.metrics import evaluate_lists


def build_truth(test, fs, seen, pos_thr):
    """user_idx -> set of item_idx the user rated positively in the test period and had not seen before (known users only)."""
    t = test[test.rating >= pos_thr]
    t = t[t.userId.isin(fs.user_index.index) & t.movieId.isin(fs.item_index.index)]
    truth, seen_sets = {}, {}
    for u, i in zip(fs.user_index.reindex(t.userId).to_numpy(), fs.item_index.reindex(t.movieId).to_numpy()):
        u, i = int(u), int(i)
        if u not in seen_sets:
            seen_sets[u] = set(seen.get(u, ()))
        if i not in seen_sets[u]:
            truth.setdefault(u, set()).add(i)
    return truth


def retrieve_candidates(users, user_emb, index, seen, n_cand):
    C = np.zeros((len(users), n_cand), dtype="int64")
    for r, u in enumerate(users):
        ids, _ = index.search_unseen(user_emb[u], n_cand, seen.get(u))
        C[r, : len(ids)] = ids
        if len(ids) < n_cand:  # pad by repeating the last id (dedup'd later); keeps array rectangular
            C[r, len(ids):] = ids[-1] if len(ids) else 0
    return C


def rank_with(score_fn, users, C):
    """score_fn(user_array, item_array)->scores; returns reranked candidate matrix."""
    uu = np.repeat(np.asarray(users), C.shape[1])
    s = np.asarray(score_fn(uu, C.reshape(-1))).reshape(C.shape)
    order = np.argsort(-s, axis=1)
    return np.take_along_axis(C, order, 1)


def _dedup(row):
    seen, out = set(), []
    for x in row:
        if x > 0 and x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


def evaluate_pipeline(fs, user_emb, index, seen, truth, rankers, cfg_eval, n_cand, seed=0):
    """rankers: {name: score_fn(user_array, item_array)}. Returns metrics dict."""
    rng = np.random.default_rng(seed)
    users = sorted(truth)
    if len(users) > cfg_eval["max_eval_users"]:
        users = sorted(rng.choice(users, cfg_eval["max_eval_users"], replace=False).tolist())
    ks = cfg_eval["ks"]
    truths = [truth[u] for u in users]
    res = {"n_eval_users": len(users), "n_candidates": n_cand, "index_backend": index.backend}

    pop = []
    for u in users:
        s_u = set(seen.get(u, ()))
        pop.append([int(i) for i in fs.pop_order[: max(ks) + len(s_u)] if i not in s_u][: max(ks)])
    res["popularity"] = evaluate_lists(pop, truths, ks)

    t0 = time.perf_counter()
    C = retrieve_candidates(users, user_emb, index, seen, n_cand)
    res["retrieval_ms_per_user"] = (time.perf_counter() - t0) * 1e3 / max(len(users), 1)
    res["retrieval_only"] = evaluate_lists([_dedup(r) for r in C], truths, ks)
    res["retrieval_candidate_recall"] = float(np.mean([len(set(_dedup(r)) & t) / len(t) for r, t in zip(C, truths)]))

    for name, fn in rankers.items():
        R = rank_with(fn, users, C)
        res[f"retrieval+{name}"] = evaluate_lists([_dedup(r) for r in R], truths, ks)

    # ANN quality vs exact search
    sub = users[: cfg_eval.get("ann_check_users", 500)]
    if sub and index.backend != "numpy":
        q = user_emb[sub]
        _, approx = index.search(q, n_cand)
        _, exact = index.exact_search(q, n_cand)
        res["ann_recall_vs_exact"] = float(np.mean([len(set(a) & set(e)) / n_cand for a, e in zip(approx, exact)]))
    return res


def evaluate_cold_start(fs, cold, test, pos_thr, ks, n_seed=3, max_users=2000):
    """Users with no training history: their first n_seed test ratings act as seeds, later positives are the truth."""
    t = test[~test.userId.isin(fs.user_index.index) & test.movieId.isin(fs.item_index.index)].sort_values("timestamp")
    pops, conts, truths = [], [], []
    for _, g in list(t.groupby("userId"))[:max_users]:
        if len(g) <= n_seed:
            continue
        items = fs.item_index.reindex(g.movieId).to_numpy().astype(int)
        seeds, rest = items[:n_seed], g.iloc[n_seed:]
        truth = set(items[n_seed:][rest.rating.to_numpy() >= pos_thr].tolist()) - set(seeds.tolist())
        if not truth:
            continue
        truths.append(truth)
        pops.append(cold.popular(max(ks), exclude=seeds).tolist())
        conts.append(cold.content_based(max(ks), seed_items=seeds).tolist())
    if not truths:
        return {"n_users": 0}
    return {"popularity": evaluate_lists(pops, truths, ks), "content_based": evaluate_lists(conts, truths, ks)}
