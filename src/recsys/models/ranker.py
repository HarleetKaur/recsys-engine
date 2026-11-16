"""Second-stage rankers: Deep & Cross Network (DCN) and Wide & Deep.

Like the towers, the models take only (user_idx, item_idx); feature tables are frozen layers inside.
"""
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recsys.models.towers import frozen_table, load_tables

keras = tf.keras
layers = keras.layers


@keras.utils.register_keras_serializable(package="recsys")
class CrossStack(layers.Layer):
    """A few TFRS cross layers applied in a row (wrapped so it takes a single input)."""

    def __init__(self, n_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.cross = [tfrs.layers.dcn.Cross() for _ in range(n_layers)]

    def call(self, x0):
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return x

    def get_config(self):
        return {**super().get_config(), "n_layers": self.n_layers}


def _mlp(x, units):
    for u in units:
        x = layers.Dense(u, activation="relu")(x)
    return x


def build_ranker(kind, fs, dim=32, hidden=(256, 128), cross_layers=2):
    """kind: 'dcn' or 'wide_deep'. Output is the predicted probability that the user rates the item >= threshold."""
    u = keras.Input(shape=(), dtype="int32", name="user_idx")
    i = keras.Input(shape=(), dtype="int32", name="item_idx")
    ue = layers.Embedding(fs.n_users + 1, dim, name="rk_user_emb")(u)
    ie = layers.Embedding(fs.n_items + 1, dim, name="rk_item_emb")(i)
    us = frozen_table(u, *fs.user_stats.shape, "user_stats_tbl")
    ug = frozen_table(u, *fs.user_genre.shape, "user_genre_tbl")
    ist = frozen_table(i, *fs.item_stats.shape, "item_stats_tbl")
    ig = frozen_table(i, *fs.item_genre.shape, "item_genre_tbl")
    affinity = layers.Dot(axes=1, name="genre_affinity")([ug, ig])  # overlap of user genre profile and item genres
    features = layers.Concatenate(name="features")([ue, ie, us, ist, ug, ig, affinity])

    if kind == "dcn":
        crossed = CrossStack(cross_layers, name="cross")(features)
        logit = layers.Dense(1, name="logit")(layers.Concatenate()([crossed, _mlp(features, hidden)]))
    elif kind == "wide_deep":
        wide = layers.Concatenate()([us, ist, affinity, layers.Multiply()([ug, ig])])
        logit = layers.Add(name="logit")([layers.Dense(1)(wide), layers.Dense(1)(_mlp(features, hidden))])
    else:
        raise ValueError(f"unknown ranker kind: {kind}")
    model = keras.Model([u, i], layers.Activation("sigmoid", name="score")(logit), name=f"ranker_{kind}")
    load_tables(model, {"user_stats_tbl": fs.user_stats, "user_genre_tbl": fs.user_genre,
                        "item_stats_tbl": fs.item_stats, "item_genre_tbl": fs.item_genre})
    return model


def make_ranker_data(df, fs, pos_thr, neg_per_pos, seed=0):
    """Observed ratings (label 1 if rating >= threshold) plus unseen items sampled by popularity as negatives."""
    rng = np.random.default_rng(seed)
    u = fs.user_index.reindex(df.userId).to_numpy().astype(float)
    i = fs.item_index.reindex(df.movieId).to_numpy().astype(float)
    ok = ~(np.isnan(u) | np.isnan(i))
    u, i = u[ok].astype("int32"), i[ok].astype("int32")
    y = (df.rating.to_numpy()[ok] >= pos_thr).astype("float32")
    pos_u = u[y == 1]
    p = np.power(np.maximum(fs.item_prob[1:], 1e-12), 0.75)
    p /= p.sum()
    neg_u = np.repeat(pos_u, neg_per_pos)
    neg_i = (rng.choice(fs.n_items, size=len(neg_u), p=p) + 1).astype("int32")
    U = np.concatenate([u, neg_u])
    I = np.concatenate([i, neg_i])
    Y = np.concatenate([y, np.zeros(len(neg_u), "float32")])
    order = rng.permutation(len(U))
    return U[order], I[order], Y[order]
