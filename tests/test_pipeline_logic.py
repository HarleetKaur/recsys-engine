import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from helpers import build_fixture
from recsys.config import apply_overrides, load_config
from recsys.features import FeatureStore
from recsys.io import read_table, write_table
from recsys.pipeline.triton_export import export_model_repository
from recsys.evaluation import metrics as M
from recsys.evaluation.ab_test import assign_arm, simulate_replay, summarize, two_proportion_ztest
from recsys.evaluation.offline import build_truth, evaluate_cold_start, evaluate_pipeline
from recsys.retrieval.faiss_index import ItemIndex
from recsys.serving.cache import InMemoryCache
from recsys.serving.cold_start import ColdStart
from recsys.serving.recommender import Recommender

R, MOV, DF, TRAIN, VAL, TEST, US, ITS, FS = build_fixture()


class TestMetrics(unittest.TestCase):
    def test_values(self):
        self.assertEqual(M.recall_at_k([1, 2, 3], {3, 9}, 3), 0.5)
        self.assertAlmostEqual(M.ndcg_at_k([1, 2], {1}, 2), 1.0)
        self.assertAlmostEqual(M.ndcg_at_k([2, 1], {1}, 2), 1 / np.log2(3))
        self.assertEqual(M.ndcg_at_k([1], set(), 1), 0.0)
        self.assertEqual(M.hit_rate_at_k([5, 6], {6}, 1), 0.0)


class TestSplit(unittest.TestCase):
    def test_time_ordered_no_leak(self):
        self.assertLessEqual(TRAIN.timestamp.max(), VAL.timestamp.min())
        self.assertLessEqual(VAL.timestamp.max(), TEST.timestamp.min())
        self.assertEqual(len(TRAIN) + len(VAL) + len(TEST), len(DF))

    def test_stats_from_train_only(self):
        self.assertEqual(US.n_ratings.sum(), len(TRAIN))


class TestFeatures(unittest.TestCase):
    def test_shapes_and_oov_row(self):
        self.assertEqual(FS.user_stats.shape, (FS.n_users + 1, 3))
        self.assertEqual(FS.item_genre.shape[0], FS.n_items + 1)
        self.assertTrue((FS.item_genre[0] == 0).all() and (FS.user_genre[0] == 0).all())
        self.assertAlmostEqual(float(FS.user_genre[1:].sum(1).max()), 1.0, places=4)

    def test_history_tables_match_build(self):
        us, ug = FS.user_tables_from_history(TRAIN, 4.0)
        np.testing.assert_allclose(us, FS.user_stats, atol=1e-5)
        np.testing.assert_allclose(ug, FS.user_genre, atol=1e-5)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            FS.save(d)
            fs2 = FeatureStore.load(d)
            np.testing.assert_array_equal(fs2.item_stats, FS.item_stats)
            self.assertEqual(fs2.genres, FS.genres)

    def test_popularity_order_matches_positive_counts(self):
        top = FS.pop_order[0]
        self.assertEqual(FS.item_prob[top], FS.item_prob[1:].max())


class TestConfigAndIO(unittest.TestCase):
    def test_overrides_and_absolute_paths(self):
        cfg = load_config(overrides=["retrieval.epochs=2", "data.dataset=ml-latest-small"])
        self.assertEqual(cfg["retrieval"]["epochs"], 2)
        self.assertEqual(cfg["data"]["dataset"], "ml-latest-small")
        self.assertTrue(os.path.isabs(cfg["paths"]["artifacts"]))
        with self.assertRaises(KeyError):
            apply_overrides(cfg, ["retrieval.no_such_key=1"])

    def test_table_roundtrip_and_spark_style_directory(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as d:
            write_table(df, d, "t")
            pd.testing.assert_frame_equal(read_table(d, "t"), df)
            with self.assertRaises(FileNotFoundError):
                read_table(d, "missing")


class TestTritonExport(unittest.TestCase):
    def test_repository_layout(self):
        with tempfile.TemporaryDirectory() as d:
            src, repo = os.path.join(d, "rankers"), os.path.join(d, "repo")
            for v in ("dcn", "wide_deep"):
                os.makedirs(os.path.join(src, v, "variables"))
                open(os.path.join(src, v, "saved_model.pb"), "wb").close()
            export_model_repository(src, repo, ["dcn", "wide_deep"])
            self.assertTrue(os.path.exists(os.path.join(repo, "ranker_dcn", "1", "model.savedmodel", "saved_model.pb")))
            with open(os.path.join(repo, "ranker_wide_deep", "config.pbtxt")) as f:
                cfg = f.read()
            self.assertIn('name: "ranker_wide_deep"', cfg)
            self.assertIn("max_batch_size: 0", cfg)
            self.assertTrue(os.path.exists(os.path.join(repo, "EXPORT_INFO.json")))


class TestIndex(unittest.TestCase):
    def test_exact_matches_bruteforce_and_filters_seen(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(101, 16)).astype("float32"); emb[0] = 0
        idx = ItemIndex(emb, force_numpy=True)
        q = rng.normal(size=(1, 16)).astype("float32")
        best = int(np.argmax(emb[1:] @ q[0])) + 1
        _, ids = idx.search(q, 5)
        self.assertEqual(int(ids[0][0]), best)
        ids2, _ = idx.search_unseen(q[0], 5, [best])
        self.assertNotIn(best, ids2.tolist()); self.assertEqual(len(ids2), 5)


class TestColdStart(unittest.TestCase):
    def test_popularity_and_content(self):
        cs = ColdStart(FS)
        pop = cs.popular(10); self.assertEqual(len(pop), 10)
        g = FS.genres[0]
        pg = cs.popular(10, genres=[g]); self.assertTrue((FS.item_genre[pg, 0] == 1).all())
        seed = np.where(FS.item_genre[:, 1] == 1)[0][:3]
        rec = cs.content_based(10, seed_items=seed)
        self.assertFalse(set(rec) & set(seed)); self.assertGreater(FS.item_genre[rec, 1].mean(), FS.item_genre[1:, 1].mean())
        self.assertEqual(len(cs.content_based(5)), 5)  # nothing given -> popularity


class FakeScorer:
    def __init__(self, sign): self.sign = sign
    def score(self, u, items): return self.sign * np.asarray(items, float) / 1000


class TestRecommender(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.ue = rng.normal(size=(FS.n_users + 1, 8)).astype("float32")
        self.ie = rng.normal(size=(FS.n_items + 1, 8)).astype("float32")
        self.idx = ItemIndex(self.ie, force_numpy=True)
        self.seen = FS.seen_items(pd.concat([TRAIN, VAL]))
        self.rec = Recommender(FS, self.ue, self.idx, self.seen, {"A": FakeScorer(1), "B": FakeScorer(-1)},
                               InMemoryCache(), ColdStart(FS), MOV, n_candidates=50)

    def test_personalized_cache_and_seen(self):
        uid = int(FS.user_ids[0]); u = int(FS.user_index[uid])
        r1 = self.rec.recommend(uid, 10); r2 = self.rec.recommend(uid, 10)
        self.assertEqual(r1["source"], "personalized"); self.assertEqual(r2["source"], "cache")
        got = FS.item_index.reindex([i["movie_id"] for i in r1["items"]]).to_numpy()
        self.assertFalse(set(got) & set(self.seen[u]))
        self.assertEqual([i["movie_id"] for i in r1["items"]], [i["movie_id"] for i in r2["items"]])

    def test_arms_rank_differently(self):
        uid = int(FS.user_ids[3])
        a = self.rec.recommend(uid, 10, arm="A")["items"]; b = self.rec.recommend(uid, 10, arm="B")["items"]
        self.assertNotEqual([i["movie_id"] for i in a], [i["movie_id"] for i in b])

    def test_cold_paths(self):
        r = self.rec.recommend(10**9, 5); self.assertEqual(r["source"], "cold_popularity")
        r = self.rec.recommend(10**9, 5, liked_genres=[FS.genres[0]]); self.assertEqual(r["source"], "cold_content")
        r = self.rec.recommend(10**9, 5, seed_movie_ids=[int(FS.item_ids[0])]); self.assertEqual(r["source"], "cold_content")


class TestOffline(unittest.TestCase):
    def test_pipeline_eval_runs_and_good_ranker_beats_bad(self):
        rng = np.random.default_rng(2)
        ue = rng.normal(size=(FS.n_users + 1, 8)).astype("float32"); ie = rng.normal(size=(FS.n_items + 1, 8)).astype("float32")
        idx = ItemIndex(ie, force_numpy=True)
        seen = FS.seen_items(pd.concat([TRAIN, VAL]))
        truth = build_truth(TEST, FS, seen, 4.0)
        self.assertGreater(len(truth), 10)
        oracle = lambda u, i: np.array([1.0 if int(ii) in truth.get(int(uu), ()) else 0.0 for uu, ii in zip(u, i)])
        bad = lambda u, i: -oracle(u, i)
        res = evaluate_pipeline(FS, ue, idx, seen, truth, {"oracle": oracle, "bad": bad},
                                {"ks": [10], "max_eval_users": 100}, 100)
        self.assertGreaterEqual(res["retrieval+oracle"]["ndcg@10"], res["retrieval_only"]["ndcg@10"])
        self.assertLessEqual(res["retrieval+bad"]["ndcg@10"], res["retrieval_only"]["ndcg@10"])
        cs = evaluate_cold_start(FS, ColdStart(FS), TEST, 4.0, [10])
        self.assertIn("n_users", cs) if cs.get("n_users") == 0 else self.assertIn("content_based", cs)


class TestAB(unittest.TestCase):
    def test_assignment_balanced_deterministic(self):
        arms = [assign_arm(u, 0.5, "s") for u in range(4000)]
        self.assertAlmostEqual(arms.count("B") / 4000, 0.5, delta=0.03)
        self.assertEqual(assign_arm(7, 0.5, "s"), assign_arm(7, 0.5, "s"))

    def test_ztest(self):
        r = two_proportion_ztest(100, 1000, 100, 1000); self.assertAlmostEqual(r["p_value"], 1.0)
        r = two_proportion_ztest(100, 1000, 160, 1000); self.assertLess(r["p_value"], 0.01)

    def test_simulation_detects_better_arm(self):
        users = list(range(4000)); truth = {u: set(range(0, 10)) for u in users}
        def rec(u, arm):  # B surfaces relevant items first, A buries them
            return list(range(0, 10)) if arm == "B" else list(range(100, 110))
        rep = summarize(simulate_replay(users, rec, truth, 10, 0.6, 0.01, 0.5, "s", 0))
        self.assertGreater(rep["arms"]["B"]["ctr_request"], rep["arms"]["A"]["ctr_request"])
        self.assertLess(rep["request_level_ztest"]["p_value"], 0.001)
        self.assertGreater(rep["diff_ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
