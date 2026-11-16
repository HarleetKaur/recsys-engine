"""Turn reports/metrics.json and reports/ab_replay.json into docs/RESULTS.md and reports/ndcg.png.

Only reads files produced by a real run; it stops with an error if they are missing.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

with open(REPORTS / "metrics.json") as f:
    m = json.load(f)
ab = None
if (REPORTS / "ab_replay.json").exists():
    with open(REPORTS / "ab_replay.json") as f:
        ab = json.load(f)

systems = [k for k, v in m.items() if isinstance(v, dict) and "ndcg@10" in v]
ks = sorted({int(x.split("@")[1]) for x in m[systems[0]] if x.startswith("ndcg@")})
env = m.get("environment", {})

lines = [f"# Results on {m['dataset']}", "",
         f"Run on {env.get('time_utc', '?')} UTC, TensorFlow {env.get('tensorflow', '?')}, "
         f"{env.get('gpus', 0)} GPU(s), Python {env.get('python', '?')}.",
         f"Split sizes (rows): {m.get('split_rows')}.",
         f"{m['n_eval_users']} users evaluated, {m['n_candidates']} candidates per user, index: {m['index_backend']}.", "",
         "## Offline ranking quality (test period)", "",
         "| System | " + " | ".join(f"NDCG@{k} | Recall@{k}" for k in ks) + " |",
         "|---|" + "---|" * (2 * len(ks))]
for s in systems:
    lines.append(f"| {s} | " + " | ".join(f"{m[s][f'ndcg@{k}']:.4f} | {m[s][f'recall@{k}']:.4f}" for k in ks) + " |")
lines += ["", f"Share of a user's test positives that appear anywhere in the {m['n_candidates']} retrieved candidates: "
              f"{m['retrieval_candidate_recall']:.4f}.",
          f"Retrieval time: {m['retrieval_ms_per_user']:.2f} ms per user."]
if "ann_recall_vs_exact" in m:
    lines.append(f"The approximate index returns {m['ann_recall_vs_exact']:.1%} of the candidates exact search would.")

cold = m.get("cold_start", {})
if "popularity" in cold:
    lines += ["", "## Users with no history (3 seed ratings)", "", "| Method | NDCG@10 | Recall@10 |", "|---|---|---|"]
    for name in ("popularity", "content_based"):
        lines.append(f"| {name} | {cold[name]['ndcg@10']:.4f} | {cold[name]['recall@10']:.4f} |")

if ab:
    A, B, z = ab["arms"]["A"], ab["arms"]["B"], ab["request_level_ztest"]
    lines += ["", "## A/B test", "", ab["note"], "",
              "| Arm | Ranker | Requests | Click rate per request | Click rate per impression |", "|---|---|---|---|---|",
              f"| A | {ab['models']['A']} | {A['requests']} | {A['ctr_request']:.4f} | {A['ctr_impression']:.4f} |",
              f"| B | {ab['models']['B']} | {B['requests']} | {B['ctr_request']:.4f} | {B['ctr_impression']:.4f} |", "",
              f"B minus A: {z['diff_B_minus_A']:+.4f} (z = {z['z']:.2f}, p = {z['p_value']:.4f}); "
              f"95% bootstrap interval {ab['diff_ci95'][0]:+.4f} to {ab['diff_ci95'][1]:+.4f}."]

DOCS.mkdir(parents=True, exist_ok=True)
with open(DOCS / "RESULTS.md", "w") as f:
    f.write("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(7, 4))
width = 0.8 / len(systems)
for j, s in enumerate(systems):
    ax.bar([x + j * width for x in range(len(ks))], [m[s][f"ndcg@{k}"] for k in ks], width, label=s)
ax.set_xticks([x + 0.4 - width / 2 for x in range(len(ks))])
ax.set_xticklabels([f"@{k}" for k in ks])
ax.set_ylabel("NDCG")
ax.set_title("NDCG on the test period")
ax.legend()
fig.tight_layout()
REPORTS.mkdir(parents=True, exist_ok=True)
fig.savefig(REPORTS / "ndcg.png", dpi=150)
print("wrote docs/RESULTS.md and reports/ndcg.png")
