"""Online recommendation logic: check the cache, then either a cold-start fallback or retrieval + ranking."""
import time
import uuid

import numpy as np

from recsys.evaluation.ab_test import assign_arm


class Recommender:
    def __init__(self, fs, user_emb, index, seen, scorers, cache, cold, movies=None,
                 n_candidates=200, ttl=600, ab_split=0.5, ab_salt="exp", model_version="v1"):
        self.fs, self.user_emb, self.index, self.seen = fs, user_emb, index, seen
        self.scorers, self.cache, self.cold = scorers, cache, cold  # scorers: {"A": scorer, "B": scorer}
        self.n_cand, self.ttl, self.split, self.salt, self.version = n_candidates, ttl, ab_split, ab_salt, model_version
        self.titles = {}
        if movies is not None:
            self.titles = dict(zip(movies.movieId, movies.title))

    def _present(self, item_idx, scores=None):
        out = []
        for n, i in enumerate(item_idx):
            mid = int(self.fs.item_ids[int(i) - 1])
            d = {"movie_id": mid, "title": self.titles.get(mid, str(mid))}
            if scores is not None:
                d["score"] = round(float(scores[n]), 5)
            out.append(d)
        return out

    def recommend(self, user_id, k=10, seed_movie_ids=None, liked_genres=None, arm=None):
        t0 = time.perf_counter()
        arm = arm or assign_arm(user_id, self.split, self.salt)
        seeds = None
        if seed_movie_ids:
            seeds = self.fs.item_index.reindex(seed_movie_ids).dropna().astype(int).to_numpy()
        uidx = self.fs.user_index.get(user_id)
        known = uidx is not None and not (isinstance(uidx, float) and np.isnan(uidx))

        ck = f"rec:{self.version}:{arm}:{user_id if known else 'cold'}:{k}:{sorted(seed_movie_ids or [])}:{sorted(liked_genres or [])}"
        hit = self.cache.get(ck)
        if hit is not None:
            hit = dict(hit, user_id=user_id, source="cache", latency_ms=round((time.perf_counter() - t0) * 1e3, 3))
            hit["request_id"] = uuid.uuid4().hex
            return hit

        if known:
            uidx = int(uidx)
            seen = self.seen.get(uidx, np.array([], int))
            cand, _ = self.index.search_unseen(self.user_emb[uidx], self.n_cand, seen)
            scores = self.scorers[arm].score(uidx, cand)
            order = np.argsort(-scores)[:k]
            items, source = self._present(cand[order], scores[order]), "personalized"
        elif (seeds is not None and len(seeds) > 0) or liked_genres:
            items, source = self._present(self.cold.content_based(k, seeds, liked_genres)), "cold_content"
        else:
            items, source = self._present(self.cold.popular(k)), "cold_popularity"

        res = {"user_id": user_id, "arm": arm, "source": source, "items": items, "model_version": self.version}
        self.cache.set(ck, res, self.ttl)
        res = dict(res, request_id=uuid.uuid4().hex, latency_ms=round((time.perf_counter() - t0) * 1e3, 3))
        return res
