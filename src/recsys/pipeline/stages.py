"""Pipeline stages. Each one reads and writes files, so Airflow can run them as separate tasks."""
import hashlib
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from recsys.io import read_table, write_table

ALL_TABLES = ("train", "val", "test", "user_stats", "item_stats", "user_genre_counts")


def _art(cfg, *parts):
    return os.path.join(cfg["paths"]["artifacts"], *parts)


def _report(cfg, name):
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    return os.path.join(cfg["paths"]["reports"], name)


def _movies(cfg):
    d = cfg["data"]
    return pd.read_csv(os.path.join(d["raw_dir"], d["dataset"], "movies.csv"))


def _tables(cfg, names=ALL_TABLES):
    return {n: read_table(cfg["data"]["processed_dir"], n) for n in names}


def environment_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = len(tf.config.list_physical_devices("GPU"))
    except ImportError:
        pass
    return info


# ----------------------------------------------------------------------------- data

def stage_download(cfg):
    from recsys.data.download import download_movielens
    download_movielens(cfg["data"]["raw_dir"], cfg["data"]["dataset"])


def stage_split(cfg):
    d = cfg["data"]
    if d["engine"] == "spark":
        from recsys.data.spark_job import run_spark
        run_spark(d["raw_dir"], d["processed_dir"], d)
        return
    from recsys.data import split
    ratings, movies = split.load_raw(d["raw_dir"], d["dataset"])
    df = split.clean(ratings, d["min_user_interactions"], d["max_users"], d["seed"])
    train, val, test = split.time_split(df, d["val_quantile"], d["test_quantile"])
    thr = d["positive_threshold"]
    user_stats, item_stats = split.compute_stats_pandas(train, thr)
    tables = {"train": train, "val": val, "test": test, "user_stats": user_stats, "item_stats": item_stats,
              "user_genre_counts": split.compute_genre_counts_pandas(train, movies, thr)}
    for name, table in tables.items():
        write_table(table, d["processed_dir"], name)
        print(f"[pandas] wrote {name}: {len(table):,} rows")


def stage_features(cfg):
    from recsys.features import FeatureStore
    t = _tables(cfg, ("user_stats", "item_stats", "user_genre_counts"))
    fs = FeatureStore.build(_movies(cfg), t["user_stats"], t["item_stats"], t["user_genre_counts"])
    fs.save(_art(cfg))
    print(f"[features] users={fs.n_users:,} items={fs.n_items:,} genres={len(fs.genres)}")


# ----------------------------------------------------------------------------- stage 1: retrieval

def stage_train_retrieval(cfg):
    import tensorflow as tf
    from recsys.features import FeatureStore
    from recsys.models.towers import TwoTowerModel, build_tower, embed_all, load_tables

    r, thr, seed = cfg["retrieval"], cfg["data"]["positive_threshold"], cfg["data"]["seed"]
    fs = FeatureStore.load(_art(cfg))
    t = _tables(cfg, ("train", "val"))
    pos = t["train"][t["train"].rating >= thr]
    u = fs.user_index.reindex(pos.userId).to_numpy().astype(float)
    i = fs.item_index.reindex(pos.movieId).to_numpy().astype(float)
    ok = ~(np.isnan(u) | np.isnan(i))
    u, i = u[ok].astype("int32"), i[ok].astype("int32")
    batch = max(2, min(r["batch_size"], len(u) // 2))
    ds = tf.data.Dataset.from_tensor_slices({"user_idx": u, "item_idx": i})
    ds = ds.shuffle(min(len(u), 1_000_000), seed=seed).batch(batch, drop_remainder=True).prefetch(2)

    tf.keras.utils.set_random_seed(seed)
    user_tower = build_tower("user", fs.user_stats, fs.user_genre, r["embedding_dim"], r["hidden_units"])
    item_tower = build_tower("item", fs.item_stats, fs.item_genre, r["embedding_dim"], r["hidden_units"])
    model = TwoTowerModel(user_tower, item_tower, fs.item_prob, r["temperature"])
    model.compile(optimizer=tf.keras.optimizers.Adam(r["learning_rate"]))
    history = model.fit(ds, epochs=r["epochs"], verbose=2)

    # For serving and evaluation, user features come from everything before the test period (train + val).
    seen_history = pd.concat([t["train"], t["val"]])
    stats, genre = fs.user_tables_from_history(seen_history, thr)
    load_tables(user_tower, {"user_stats_tbl": stats, "user_genre_tbl": genre})
    os.makedirs(_art(cfg, "retrieval"), exist_ok=True)
    np.save(_art(cfg, "retrieval", "user_emb.npy"), embed_all(user_tower, fs.n_users + 1))
    np.save(_art(cfg, "retrieval", "item_emb.npy"), embed_all(item_tower, fs.n_items + 1))
    with open(_art(cfg, "seen.pkl"), "wb") as f:
        pickle.dump(fs.seen_items(seen_history), f)
    with open(_report(cfg, "retrieval_training.json"), "w") as f:
        json.dump({"loss": [float(x) for x in history.history["loss"]]}, f)
    print("[retrieval] saved user/item embeddings")


def stage_build_index(cfg):
    from recsys.retrieval.faiss_index import ItemIndex
    r = cfg["retrieval"]
    index = ItemIndex(np.load(_art(cfg, "retrieval", "item_emb.npy")), r["index_type"], r["hnsw_m"], r["hnsw_ef_search"])
    index.save(_art(cfg, "index"))
    print(f"[index] built {index.backend} over {len(index.emb):,} items")


# ----------------------------------------------------------------------------- stage 2: rankers

def stage_train_ranker(cfg):
    import tensorflow as tf
    from recsys.features import FeatureStore
    from recsys.models.ranker import build_ranker, make_ranker_data
    from recsys.models.towers import load_tables, save_savedmodel

    rk, thr, seed = cfg["ranker"], cfg["data"]["positive_threshold"], cfg["data"]["seed"]
    fs = FeatureStore.load(_art(cfg))
    t = _tables(cfg, ("train", "val"))
    U, I, Y = make_ranker_data(t["train"], fs, thr, rk["neg_per_pos"], seed)
    Uv, Iv, Yv = make_ranker_data(t["val"], fs, thr, rk["neg_per_pos"], seed + 1)  # validation = the val period
    print(f"[ranker] train rows={len(U):,} (positive share {Y.mean():.2f}), validation rows={len(Uv):,}")
    validation = ({"user_idx": Uv, "item_idx": Iv}, Yv) if len(Uv) else None
    stats, genre = fs.user_tables_from_history(pd.concat([t["train"], t["val"]]), thr)

    curves = {}
    for kind in rk["variants"]:
        tf.keras.utils.set_random_seed(seed)
        model = build_ranker(kind, fs, rk["embedding_dim"], tuple(rk["hidden_units"]), rk["cross_layers"])
        model.compile(optimizer=tf.keras.optimizers.Adam(rk["learning_rate"]), loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc")])
        h = model.fit({"user_idx": U, "item_idx": I}, Y, batch_size=rk["batch_size"], epochs=rk["epochs"],
                      validation_data=validation, verbose=2)
        load_tables(model, {"user_stats_tbl": stats, "user_genre_tbl": genre})  # features up to the test period
        save_savedmodel(model, _art(cfg, "rankers", kind))
        curves[kind] = {k: [float(x) for x in v] for k, v in h.history.items()}
    with open(_report(cfg, "ranker_training.json"), "w") as f:
        json.dump(curves, f)


# ----------------------------------------------------------------------------- evaluation and serving

def _load_serving_parts(cfg):
    from recsys.features import FeatureStore
    from recsys.retrieval.faiss_index import ItemIndex
    with open(_art(cfg, "seen.pkl"), "rb") as f:
        seen = pickle.load(f)
    return (FeatureStore.load(_art(cfg)), np.load(_art(cfg, "retrieval", "user_emb.npy")),
            ItemIndex.load(_art(cfg, "index")), seen)


def stage_evaluate(cfg):
    from recsys.evaluation.offline import build_truth, evaluate_cold_start, evaluate_pipeline
    from recsys.serving.cold_start import ColdStart
    from recsys.serving.scorers import SavedModelScorer

    fs, user_emb, index, seen = _load_serving_parts(cfg)
    t = _tables(cfg, ("train", "val", "test"))
    thr = cfg["data"]["positive_threshold"]
    truth = build_truth(t["test"], fs, seen, thr)
    rankers = {k: SavedModelScorer(_art(cfg, "rankers", k)).score_batch for k in cfg["ranker"]["variants"]}
    res = evaluate_pipeline(fs, user_emb, index, seen, truth, rankers, cfg["eval"], cfg["retrieval"]["n_candidates"])
    res["cold_start"] = evaluate_cold_start(fs, ColdStart(fs), t["test"], thr, cfg["eval"]["ks"])
    res["dataset"] = cfg["data"]["dataset"]
    res["split_rows"] = {k: len(v) for k, v in t.items()}
    res["settings"] = {"data": {k: cfg["data"][k] for k in ("engine", "positive_threshold", "min_user_interactions",
                                                           "max_users", "val_quantile", "test_quantile", "seed")},
                       "retrieval": cfg["retrieval"], "ranker": cfg["ranker"]}
    res["environment"] = environment_info()
    with open(_report(cfg, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("settings",)}, indent=2))


def build_recommender(cfg):
    from recsys.serving.cache import make_cache
    from recsys.serving.cold_start import ColdStart
    from recsys.serving.recommender import Recommender
    from recsys.serving.scorers import SavedModelScorer, TritonScorer

    s = cfg["serving"]
    fs, user_emb, index, seen = _load_serving_parts(cfg)
    movies = _movies(cfg)
    a, b = cfg["ranker"]["variants"][:2]
    backend = os.environ.get("RECSYS_SERVING_BACKEND", s["backend"]).strip().lower()
    if backend == "triton":
        url = os.environ.get("RECSYS_TRITON_URL", s["triton_url"])
        scorers = {"A": TritonScorer(url, f"ranker_{a}"), "B": TritonScorer(url, f"ranker_{b}")}
    elif backend == "keras":
        scorers = {"A": SavedModelScorer(_art(cfg, "rankers", a)), "B": SavedModelScorer(_art(cfg, "rankers", b))}
    else:
        raise ValueError(f"Unrecognized serving backend: {backend}")
    # The cache key includes the model version, so retrained models never serve old cached lists.
    # Hash file metadata from both stages, so replacing a ranker invalidates cached recommendations too.
    fingerprint = hashlib.sha256()
    for path in [_art(cfg, "retrieval", "user_emb.npy"),
                 *[os.path.join(folder, name)
                   for variant in (a, b)
                   for folder, _, names in os.walk(_art(cfg, "rankers", variant))
                   for name in sorted(names)]]:
        st = os.stat(path)
        fingerprint.update(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode())
    version = "m" + fingerprint.hexdigest()[:16]
    rec = Recommender(fs, user_emb, index, seen, scorers, make_cache(s), ColdStart(fs), movies,
                      cfg["retrieval"]["n_candidates"], s["cache_ttl_seconds"], s["ab_split"], s["ab_salt"], version)
    rec.backend = backend
    return rec


def stage_ab_replay(cfg):
    from recsys.evaluation.ab_test import simulate_replay, summarize
    from recsys.evaluation.offline import build_truth, rank_with, retrieve_candidates
    from recsys.serving.scorers import SavedModelScorer

    fs, user_emb, index, seen = _load_serving_parts(cfg)
    test = _tables(cfg, ("test",))["test"]
    ab, k = cfg["ab_replay"], cfg["serving"]["default_k"]
    truth = build_truth(test, fs, seen, cfg["data"]["positive_threshold"])
    users = sorted(truth)[: ab["n_users"]]
    a, b = cfg["ranker"]["variants"][:2]
    scorers = {"A": SavedModelScorer(_art(cfg, "rankers", a)), "B": SavedModelScorer(_art(cfg, "rankers", b))}
    candidates = {}

    def recommend(user, arm):
        if user not in candidates:
            candidates[user] = retrieve_candidates([user], user_emb, index, seen, cfg["retrieval"]["n_candidates"])
        ranked = rank_with(scorers[arm].score_batch, [user], candidates[user])
        return [int(x) for x in ranked[0][:k]]

    records = simulate_replay(users, recommend, truth, k, ab["p_click_relevant"], ab["p_click_irrelevant"],
                              cfg["serving"]["ab_split"], cfg["serving"]["ab_salt"], cfg["data"]["seed"])
    report = summarize(records)
    report["models"] = {"A": a, "B": b}
    report["note"] = "Simulated traffic: clicks are sampled from test-period positives with a position-bias model."
    with open(_report(cfg, "ab_replay.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def stage_export_triton(cfg):
    from recsys.pipeline.triton_export import export_model_repository
    export_model_repository(_art(cfg, "rankers"), cfg["paths"]["triton_repo"], cfg["ranker"]["variants"])
