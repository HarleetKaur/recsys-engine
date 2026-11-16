"""Feature store: id maps plus user/item feature tables, all as numpy arrays.

Row 0 of every table is reserved for unknown ids, so real users/items are numbered from 1.
The tables are built from the aggregate tables of the split stage (Spark or pandas), which only
use the training period.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse

from recsys.data.split import NO_GENRE


def _finish_user_arrays(stats, genre_counts, n_users, n_max):
    """stats: df[user_idx, n_ratings, mean_rating, pos_rate]; genre_counts: (n_users+1, G) positive counts."""
    us = np.zeros((n_users + 1, 3), np.float32)
    if len(stats):
        r = stats.user_idx.to_numpy().astype(int)
        us[r, 0] = np.clip(np.log1p(stats.n_ratings.to_numpy()) / np.log1p(n_max), 0, 1)
        us[r, 1] = stats.mean_rating.to_numpy() / 5.0
        us[r, 2] = stats.pos_rate.to_numpy()
    total = genre_counts.sum(1, keepdims=True)
    ug = np.divide(genre_counts, total, out=np.zeros_like(genre_counts, dtype=np.float64), where=total > 0)
    return us, ug.astype(np.float32)


class FeatureStore:
    def __init__(self, user_ids, item_ids, genres, item_genre, item_stats, user_stats, user_genre,
                 item_prob, pop_order, user_n_max):
        self.user_ids, self.item_ids, self.genres = np.asarray(user_ids), np.asarray(item_ids), list(genres)
        self.item_genre, self.item_stats = item_genre, item_stats
        self.user_stats, self.user_genre = user_stats, user_genre
        self.item_prob, self.pop_order, self.user_n_max = item_prob, pop_order, float(user_n_max)
        self.user_index = pd.Series(np.arange(1, len(self.user_ids) + 1), index=self.user_ids)
        self.item_index = pd.Series(np.arange(1, len(self.item_ids) + 1), index=self.item_ids)

    @property
    def n_users(self):
        return len(self.user_ids)

    @property
    def n_items(self):
        return len(self.item_ids)

    @classmethod
    def build(cls, movies, user_stats_df, item_stats_df, user_genre_df, bayes_m=20):
        user_ids = np.sort(user_stats_df.userId.unique())
        item_ids = np.sort(item_stats_df.movieId.unique())
        genres = sorted({g for s in movies.genres.astype(str) for g in s.split("|")} - {NO_GENRE})
        gpos = pd.Series(np.arange(len(genres)), index=genres)
        uidx = pd.Series(np.arange(1, len(user_ids) + 1), index=user_ids)
        iidx = pd.Series(np.arange(1, len(item_ids) + 1), index=item_ids)

        item_genre = np.zeros((len(item_ids) + 1, len(genres)), np.float32)
        for mid, gs in zip(movies.movieId, movies.genres.astype(str)):
            if mid in iidx.index:
                for g in gs.split("|"):
                    if g in gpos.index:
                        item_genre[iidx[mid], gpos[g]] = 1.0

        # item side: popularity (log count), smoothed mean rating, share of positive ratings
        ist = item_stats_df.set_index("movieId").reindex(item_ids)
        n = ist.n_ratings.to_numpy(float)
        global_mean = float((ist.mean_rating * ist.n_ratings).sum() / ist.n_ratings.sum())
        smoothed = (ist.mean_rating.to_numpy() * n + bayes_m * global_mean) / (n + bayes_m)
        item_stats = np.zeros((len(item_ids) + 1, 3), np.float32)
        item_stats[1:, 0] = np.log1p(n) / np.log1p(n.max())
        item_stats[1:, 1] = smoothed / 5.0
        item_stats[1:, 2] = ist.pos_rate.to_numpy()

        pos_count = n * ist.pos_rate.to_numpy()
        item_prob = np.ones(len(item_ids) + 1, np.float32)  # chance an item shows up as a positive
        item_prob[1:] = np.clip(pos_count / max(pos_count.sum(), 1), 1e-8, 1.0)
        pop_order = np.argsort(-pos_count, kind="stable") + 1

        # user side: rating stats and genre profile (share of positives per genre)
        stats = user_stats_df.assign(user_idx=uidx.reindex(user_stats_df.userId).to_numpy())
        counts = np.zeros((len(user_ids) + 1, len(genres)))
        rows = uidx.reindex(user_genre_df.userId).to_numpy()
        cols = gpos.reindex(user_genre_df.genre).to_numpy()
        ok = ~(np.isnan(rows.astype(float)) | np.isnan(cols.astype(float)))
        np.add.at(counts, (rows[ok].astype(int), cols[ok].astype(int)), user_genre_df.n_pos.to_numpy()[ok])
        n_max = float(user_stats_df.n_ratings.max())
        us, ug = _finish_user_arrays(stats, counts, len(user_ids), n_max)
        return cls(user_ids, item_ids, genres, item_genre, item_stats, us, ug, item_prob, pop_order, n_max)

    def user_tables_from_history(self, hist, pos_thr=4.0):
        """User feature tables recomputed from any history (for example train + val), same definitions as build()."""
        h = hist[hist.userId.isin(self.user_index.index) & hist.movieId.isin(self.item_index.index)]
        df = pd.DataFrame({
            "user_idx": self.user_index.reindex(h.userId).to_numpy(),
            "item_idx": self.item_index.reindex(h.movieId).to_numpy(),
            "rating": h.rating.to_numpy(),
        })
        df["pos"] = (df.rating >= pos_thr).astype(float)
        stats = df.groupby("user_idx").agg(
            n_ratings=("rating", "size"), mean_rating=("rating", "mean"), pos_rate=("pos", "mean")).reset_index()
        pos = df[df.pos == 1]
        m = sparse.csr_matrix(
            (np.ones(len(pos)), (pos.user_idx.to_numpy(), pos.item_idx.to_numpy())),
            shape=(self.n_users + 1, self.n_items + 1))
        counts = np.asarray((m @ sparse.csr_matrix(self.item_genre)).todense())
        return _finish_user_arrays(stats, counts, self.n_users, self.user_n_max)

    def seen_items(self, hist):
        """user_idx -> sorted array of every item_idx the user has interacted with."""
        h = hist[hist.userId.isin(self.user_index.index) & hist.movieId.isin(self.item_index.index)]
        df = pd.DataFrame({"u": self.user_index.reindex(h.userId).to_numpy(),
                           "i": self.item_index.reindex(h.movieId).to_numpy()})
        return {int(u): np.sort(g.i.to_numpy()) for u, g in df.groupby("u")}

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(
            os.path.join(directory, "features.npz"), user_ids=self.user_ids, item_ids=self.item_ids,
            item_genre=self.item_genre, item_stats=self.item_stats, user_stats=self.user_stats,
            user_genre=self.user_genre, item_prob=self.item_prob, pop_order=self.pop_order)
        with open(os.path.join(directory, "features_meta.json"), "w") as f:
            json.dump({"genres": self.genres, "user_n_max": self.user_n_max}, f)

    @classmethod
    def load(cls, directory):
        z = np.load(os.path.join(directory, "features.npz"))
        with open(os.path.join(directory, "features_meta.json")) as f:
            meta = json.load(f)
        return cls(z["user_ids"], z["item_ids"], meta["genres"], z["item_genre"], z["item_stats"], z["user_stats"],
                   z["user_genre"], z["item_prob"], z["pop_order"], meta["user_n_max"])
