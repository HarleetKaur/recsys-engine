"""Compare a running Triton server against the local SavedModel on random (user, item) pairs.

    python scripts/check_triton.py --url localhost:8000 --model dcn
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recsys.config import load_config  # noqa: E402
from recsys.features import FeatureStore  # noqa: E402
from recsys.serving.scorers import SavedModelScorer, TritonScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="dcn", help="ranker variant: dcn or wide_deep")
    args = ap.parse_args()

    cfg = load_config()
    fs = FeatureStore.load(cfg["paths"]["artifacts"])
    rng = np.random.default_rng(0)
    users = rng.integers(1, fs.n_users + 1, 500, dtype="int32")
    items = rng.integers(1, fs.n_items + 1, 500, dtype="int32")

    local = SavedModelScorer(os.path.join(cfg["paths"]["artifacts"], "rankers", args.model)).score_batch(users, items)
    remote = TritonScorer(args.url, f"ranker_{args.model}").score_batch(users, items)
    diff = float(np.abs(local - remote).max())
    print(f"max abs difference between local and Triton scores: {diff:.2e}")
    sys.exit(0 if diff < 1e-4 else 1)


if __name__ == "__main__":
    main()
