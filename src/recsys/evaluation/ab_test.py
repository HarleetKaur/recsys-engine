"""A/B test helpers: user assignment, CTR statistics, and a replay simulator.

The simulator is not real traffic. It samples clicks from test-period positives with a position-bias model.
The API's /recommend and /feedback logs produce the same record format for real traffic.
"""
import hashlib
import math

import numpy as np


def assign_arm(user_id, split_b=0.5, salt="exp"):
    h = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:12], 16) % 10_000
    return "B" if h < split_b * 10_000 else "A"


def two_proportion_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1 if n1 else 0.0, x2 / n2 if n2 else 0.0
    pool = (x1 + x2) / (n1 + n2) if (n1 + n2) else 0.0
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if n1 and n2 else float("nan")
    z = (p2 - p1) / se if se and se == se and se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"p_A": p1, "p_B": p2, "diff_B_minus_A": p2 - p1, "z": z, "p_value": p}


def bootstrap_diff_ci(a, b, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)]
    return tuple(np.quantile(d, [alpha / 2, 1 - alpha / 2]).tolist())


def summarize(records):
    """records: dicts with request_id, user_id, arm, items (shown) and clicked. Returns CTR per arm and a significance test."""
    stats = {}
    for arm in ("A", "B"):
        rs = [r for r in records if r["arm"] == arm]
        shown = sum(len(r["items"]) for r in rs)
        clicks = sum(len(r["clicked"]) for r in rs)
        req_click = [1.0 if r["clicked"] else 0.0 for r in rs]
        stats[arm] = {
            "requests": len(rs), "impressions": shown, "clicks": clicks,
            "ctr_impression": clicks / shown if shown else 0.0,
            "ctr_request": float(np.mean(req_click)) if rs else 0.0,
            "_req_click": req_click,
        }
    a, b = stats["A"], stats["B"]
    test = two_proportion_ztest(sum(a["_req_click"]), a["requests"], sum(b["_req_click"]), b["requests"])
    lo, hi = bootstrap_diff_ci(a["_req_click"], b["_req_click"])
    for s in stats.values():
        s.pop("_req_click")
    return {"arms": stats, "request_level_ztest": test, "diff_ci95": [lo, hi]}


def simulate_replay(users, recommend_fn, truth, k=10, p_rel=0.6, p_irr=0.01, split_b=0.5, salt="exp", seed=0):
    """users: user_idx list; recommend_fn(user_idx, arm) -> list of item_idx; truth: user_idx -> set(item_idx)."""
    rng = np.random.default_rng(seed)
    out = []
    for n, u in enumerate(users):
        arm = assign_arm(u, split_b, salt)
        items = list(recommend_fn(u, arm))[:k]
        rel = truth.get(u, set())
        clicked = []
        for pos, it in enumerate(items):
            examine = 1.0 / math.log2(pos + 2)  # position bias
            if rng.random() < examine * (p_rel if it in rel else p_irr):
                clicked.append(int(it))
        out.append({"request_id": f"sim-{n}", "user_id": int(u), "arm": arm, "items": [int(i) for i in items], "clicked": clicked})
    return out
