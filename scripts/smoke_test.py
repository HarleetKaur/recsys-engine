"""Run every pipeline stage on a small synthetic dataset to check that the whole thing works end to end.

    python scripts/smoke_test.py                  # pandas engine
    python scripts/smoke_test.py --engine spark   # also exercises the Spark job

It needs the full requirements (TensorFlow, TFRS, FAISS; PySpark for --engine spark) and takes a few
minutes on CPU. Everything is written to smoke_run/, so real artifacts are never touched.
The numbers it prints are meaningless (synthetic data, one epoch) - it only checks that nothing breaks.
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from recsys.config import load_config  # noqa: E402
from recsys.data.synthetic import make_synthetic  # noqa: E402
from recsys.pipeline import stages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pandas", "spark"], default="pandas")
    args = ap.parse_args()

    run = os.path.join(ROOT, "smoke_run")
    shutil.rmtree(run, ignore_errors=True)
    cfg = load_config()
    cfg["data"].update(dataset="synthetic", raw_dir=f"{run}/raw", processed_dir=f"{run}/processed",
                       engine=args.engine, min_user_interactions=20, max_users=None)
    cfg["paths"].update(artifacts=f"{run}/artifacts", reports=f"{run}/reports", triton_repo=f"{run}/triton")
    cfg["serving"].update(log_dir=f"{run}/logs", redis_host="localhost")
    cfg["data"]["spark_driver_memory"] = "2g"  # small smoke data does not need an 8GB Spark driver
    cfg["retrieval"].update(epochs=1, batch_size=256, n_candidates=50, hnsw_m=16)
    cfg["ranker"].update(epochs=1, batch_size=512, neg_per_pos=1)
    cfg["eval"].update(ks=[10], max_eval_users=100, ann_check_users=50)
    cfg["ab_replay"].update(n_users=200)

    os.makedirs(f"{run}/raw/synthetic")
    ratings, movies = make_synthetic()
    ratings.to_csv(f"{run}/raw/synthetic/ratings.csv", index=False)
    movies.to_csv(f"{run}/raw/synthetic/movies.csv", index=False)

    for name in ["split", "features", "train_retrieval", "build_index", "train_ranker",
                 "evaluate", "ab_replay", "export_triton"]:
        print(f"\n===== smoke: {name} =====")
        getattr(stages, f"stage_{name}")(cfg)

    print("\n===== smoke: online recommender =====")
    rec = stages.build_recommender(cfg)
    user = int(rec.fs.user_ids[0])
    first, second = rec.recommend(user, 5), rec.recommend(user, 5)
    assert first["source"] == "personalized" and second["source"] == "cache", (first["source"], second["source"])
    assert rec.recommend(10**9, 5)["source"] == "cold_popularity"
    assert rec.recommend(10**9, 5, liked_genres=[rec.fs.genres[0]])["source"] == "cold_content"
    from recsys.retrieval.faiss_index import HAS_FAISS
    if HAS_FAISS:
        assert rec.index.backend.startswith("faiss-"), f"FAISS installed but backend was {rec.index.backend}"
    print(f"cache backend: {rec.cache.kind}; index: {rec.index.backend}")

    for path in [f"{run}/reports/metrics.json", f"{run}/reports/ab_replay.json",
                 f"{run}/triton/ranker_dcn/config.pbtxt"]:
        assert os.path.exists(path), f"missing {path}"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
