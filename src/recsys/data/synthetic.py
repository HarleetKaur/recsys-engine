"""Small made-up ratings in MovieLens format, for tests and the smoke run only (never for reported results)."""
import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]


def make_synthetic(n_users=300, n_items=400, per_user=60, seed=0):
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_items + 1) * 3,
        "title": [f"Movie {i} ({1980 + i % 40})" for i in range(n_items)],
        "genres": ["|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False)) for _ in range(n_items)],
    })
    G = np.array([[g in s.split("|") for g in GENRES] for s in movies.genres], dtype=float)
    pop = rng.pareto(1.2, n_items) + 1
    pop /= pop.sum()
    rows = []
    for u in range(1, n_users + 1):
        taste = rng.dirichlet(np.ones(len(GENRES)) * 0.3)
        affinity = G @ taste
        p = pop * (0.2 + affinity)
        p /= p.sum()
        items = rng.choice(n_items, size=min(per_user, n_items), replace=False, p=p)
        ts = np.sort(rng.integers(1_000_000_000, 1_600_000_000, size=len(items)))
        for it, t in zip(items, ts):
            r = np.clip(np.round(2.5 + 4 * affinity[it] + rng.normal(0, 0.7)), 1, 5)
            rows.append((u, int(movies.movieId[it]), float(r), int(t)))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    ratings["userId"] = ratings.userId.astype("int32")
    ratings["movieId"] = ratings.movieId.astype("int32")
    return ratings, movies
