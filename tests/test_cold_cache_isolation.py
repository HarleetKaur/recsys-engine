"""Cold-start recommendation responses must not claim to belong to a different user."""
import unittest
import numpy as np
from recsys.serving.cache import InMemoryCache
from recsys.serving.recommender import Recommender


class _FeatureStore:
    user_index = {}
    item_ids = np.array([101, 102, 103])


class _ColdStart:
    def content_based(self, k, seed_items=None, liked_genres=None):
        return np.array([1, 2, 3])[:k]


class TestColdCacheIsolation(unittest.TestCase):
    def test_shared_cold_cache_preserves_current_request_user_id(self):
        rec = Recommender(
            _FeatureStore(), None, None, {}, {}, InMemoryCache(), _ColdStart(),
            n_candidates=3,
        )
        first = rec.recommend(900001, 2, liked_genres=["Thriller"], arm="A")
        second = rec.recommend(900002, 2, liked_genres=["Thriller"], arm="A")
        self.assertEqual(first["source"], "cold_content")
        self.assertEqual(second["source"], "cache")
        self.assertEqual(first["user_id"], 900001)
        self.assertEqual(second["user_id"], 900002)
        self.assertEqual(first["items"], second["items"])
        self.assertNotEqual(first["request_id"], second["request_id"])


if __name__ == "__main__":
    unittest.main()
