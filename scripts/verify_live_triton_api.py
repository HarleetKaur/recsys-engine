"""End-to-end runtime check against YOUR running Docker API and Triton containers.

Requires running API with the optional Triton Compose overlay. This script never
submits fake clicks or mutates model weights. It verifies both arm counters.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "http://localhost:8080"
TRITON = "http://localhost:8000"
MODELS = {"A": "ranker_dcn", "B": "ranker_wide_deep"}


def read_json(url, payload=None):
    req = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=60) as r:
            return json.load(r)
    except HTTPError as exc:
        raise RuntimeError(f"{url}: HTTP {exc.code} {exc.read(2000).decode(errors='replace')}") from exc


def model_count(name):
    data = read_json(f"{TRITON}/v2/models/{name}/stats")
    return sum(int(m.get("inference_count", 0)) for m in data.get("model_stats", []))


def arm_of(user_id, salt, split=0.5):
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:12]
    return "B" if int(digest, 16) % 10_000 < split * 10_000 else "A"


def main():
    health = read_json(API + "/health")
    print("API health:", health)
    if health.get("backend") != "triton":
        raise RuntimeError("API is not in Triton mode. Use docker-compose.triton.yml (see docs).")
    import re
    from pathlib import Path
    raw = Path("configs/config.yaml").read_text()
    salt = re.search(r"(?m)^\s*ab_salt:\s*(\S+)", raw).group(1)
    split = float(re.search(r"(?m)^\s*ab_split:\s*([0-9.]+)", raw).group(1))
    ratings = Counter()
    with open("data/raw/ml-latest-small/ratings.csv", newline="") as f:
        for row in csv.DictReader(f):
            ratings[int(row["userId"])] += 1
    users = {}
    for user_id, count in sorted(ratings.items()):
        if count < 40:
            continue
        arm = arm_of(user_id, salt, split)
        users.setdefault(arm, user_id)
        if len(users) == 2:
            break
    if len(users) != 2:
        raise RuntimeError("Could not discover both A/B arms from configured assignment")
    before = {arm: model_count(model) for arm, model in MODELS.items()}
    print("Before:", before)
    for idx, arm in enumerate(("A", "B")):
        # Unique seed IDs keep the cache key different, even for a known user.
        # For personalized users, these seeds do not alter ranking logic.
        payload = {"user_id": users[arm], "k": 25 + idx,
                   "seed_movie_ids": [100000000 + idx]}
        result = read_json(API + "/recommend", payload)
        print(f"Arm {arm}: user={users[arm]}, source={result.get('source')}, "
              f"returned_arm={result.get('arm')}, items={len(result.get('items', []))}")
        if result.get("arm") != arm or result.get("source") != "personalized":
            raise RuntimeError("Unexpected arm or cached/cold response; check model/user IDs")
    after = {arm: model_count(model) for arm, model in MODELS.items()}
    print("After:", after)
    delta = {arm: after[arm] - before[arm] for arm in MODELS}
    print("Triton inference deltas:", delta)
    if not all(count > 0 for count in delta.values()):
        raise RuntimeError("No confirmed inference for one or both arms")
    print("PASS: FastAPI -> Triton DCN AND Wide & Deep inference verified")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, URLError) as exc:
        print("FAILED:", exc, file=sys.stderr)
        sys.exit(1)
