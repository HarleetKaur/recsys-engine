import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recsys.data.split import clean, compute_genre_counts_pandas, compute_stats_pandas, time_split  # noqa: E402
from recsys.data.synthetic import make_synthetic  # noqa: E402
from recsys.features import FeatureStore  # noqa: E402


def build_fixture():
    ratings, movies = make_synthetic(n_users=200, n_items=300, per_user=50)
    df = clean(ratings, 20)
    train, val, test = time_split(df, 0.8, 0.9)
    user_stats, item_stats = compute_stats_pandas(train, 4.0)
    genre_counts = compute_genre_counts_pandas(train, movies, 4.0)
    fs = FeatureStore.build(movies, user_stats, item_stats, genre_counts)
    return ratings, movies, df, train, val, test, user_stats, item_stats, fs
