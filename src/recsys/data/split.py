"""Cleaning, time-based splitting and the aggregate tables (pandas version of the Spark job)."""
import os

import numpy as np
import pandas as pd

NO_GENRE = "(no genres listed)"


def load_raw(raw_dir, dataset):
    base = os.path.join(raw_dir, dataset)
    ratings = pd.read_csv(
        os.path.join(base, "ratings.csv"),
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(os.path.join(base, "movies.csv"))
    return ratings, movies


def clean(ratings, min_user_interactions=20, max_users=None, seed=42):
    """Drop duplicates and invalid ratings, drop users with too few ratings, optionally subsample users."""
    df = ratings.dropna().drop_duplicates(["userId", "movieId"], keep="last")
    df = df[(df.rating >= 0.5) & (df.rating <= 5.0)]
    counts = df.groupby("userId").size()
    keep = counts[counts >= min_user_interactions].index
    if max_users and len(keep) > max_users:
        keep = np.random.default_rng(seed).choice(keep.to_numpy(), size=max_users, replace=False)
    return df[df.userId.isin(keep)].reset_index(drop=True)


def time_split(df, val_q=0.85, test_q=0.93):
    """Split on the global timeline: train <= t_val < val <= t_test < test."""
    t_val, t_test = df.timestamp.quantile([val_q, test_q]).to_numpy()
    train = df[df.timestamp <= t_val]
    val = df[(df.timestamp > t_val) & (df.timestamp <= t_test)]
    test = df[df.timestamp > t_test]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def compute_stats_pandas(train, pos_thr):
    """Per-user and per-item rating count, mean rating and share of positive ratings (training period only)."""
    t = train.assign(pos=(train.rating >= pos_thr).astype("float64"))
    agg = dict(n_ratings=("rating", "size"), mean_rating=("rating", "mean"), pos_rate=("pos", "mean"))
    return t.groupby("userId").agg(**agg).reset_index(), t.groupby("movieId").agg(**agg).reset_index()


def compute_genre_counts_pandas(train, movies, pos_thr):
    """How many positive ratings each user gave to each genre (training period only)."""
    g = movies[["movieId", "genres"]].assign(genre=movies.genres.astype(str).str.split("|")).explode("genre")
    g = g[g.genre != NO_GENRE][["movieId", "genre"]]
    pos = train[train.rating >= pos_thr][["userId", "movieId"]]
    return pos.merge(g, on="movieId").groupby(["userId", "genre"]).size().rename("n_pos").reset_index()
