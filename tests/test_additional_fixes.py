"""Regression checks for problems found during the project handover."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from recsys.pipeline.stages import build_recommender


class _FS:
    n_users = 1
    n_items = 1
    user_ids = np.array([123])
    item_ids = np.array([456])


class TestCacheVersion(unittest.TestCase):
    def test_ranker_file_update_changes_cache_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("retrieval", "rankers/dcn", "rankers/wide_deep"):
                (root / name).mkdir(parents=True)
            (root / "retrieval/user_emb.npy").write_bytes(b"retrieval")
            (root / "rankers/dcn/saved_model.pb").write_bytes(b"ranker-a")
            (root / "rankers/wide_deep/saved_model.pb").write_bytes(b"ranker-b")
            cfg = {
                "paths": {"artifacts": str(root)},
                "ranker": {"variants": ["dcn", "wide_deep"]},
                "serving": {"backend": "keras", "cache_ttl_seconds": 30,
                            "ab_split": 0.5, "ab_salt": "test"},
                "retrieval": {"n_candidates": 10},
            }
            # Avoid imports/loading for external serving dependencies in this unit test.
            with patch("recsys.pipeline.stages._load_serving_parts", return_value=(_FS(), np.zeros((2, 2)), None, {})), \
                 patch("recsys.pipeline.stages._movies", return_value=None), \
                 patch("recsys.serving.scorers.SavedModelScorer", return_value=object()), \
                 patch("recsys.serving.cache.make_cache", return_value=object()), \
                 patch("recsys.serving.cold_start.ColdStart", return_value=object()):
                first = build_recommender(cfg).version
                (root / "rankers/dcn/saved_model.pb").write_bytes(b"ranker-a-UPDATED")
                second = build_recommender(cfg).version
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
