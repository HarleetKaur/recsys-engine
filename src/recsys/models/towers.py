"""Two-tower retrieval model built with TensorFlow Recommenders.

Each tower takes only an integer id. The user/item feature tables (stats, genres) are frozen embedding
layers inside the model, so the saved models and the embeddings need nothing but ids at serving time.
"""
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

keras = tf.keras
layers = keras.layers


def frozen_table(idx, rows, cols, name):
    """Lookup of a fixed feature table by id. The values are filled in later with load_tables()."""
    return layers.Embedding(rows, cols, embeddings_initializer="zeros", trainable=False, name=name)(idx)


def load_tables(model, tables):
    """tables: {layer name: array}. Also used to swap in newer user features before saving."""
    for name, values in tables.items():
        model.get_layer(name).set_weights([np.asarray(values, "float32")])


def save_savedmodel(model, path):
    try:
        model.save(path, save_format="tf")
    except (TypeError, ValueError):  # Keras 3 has no save_format; export() writes a SavedModel
        model.export(path)


def build_tower(kind, stats, genre, dim=64, hidden=128):
    """kind is 'user' or 'item'; stats/genre are the feature tables (row 0 = unknown id)."""
    rows = stats.shape[0]
    idx = keras.Input(shape=(), dtype="int32", name=f"{kind}_idx")
    emb = layers.Embedding(rows, dim, name=f"{kind}_emb")(idx)
    st = frozen_table(idx, rows, stats.shape[1], f"{kind}_stats_tbl")
    ge = frozen_table(idx, rows, genre.shape[1], f"{kind}_genre_tbl")
    x = layers.Concatenate()([emb, st, ge])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(dim)(x)
    out = layers.UnitNormalization(axis=-1, name=f"{kind}_vec")(x)
    model = keras.Model(idx, out, name=f"{kind}_tower")
    load_tables(model, {f"{kind}_stats_tbl": stats, f"{kind}_genre_tbl": genre})
    return model


class TwoTowerModel(tfrs.Model):
    """In-batch sampled softmax with a popularity (logQ) correction, on L2-normalised embeddings."""

    def __init__(self, user_tower, item_tower, item_prob, temperature=0.05):
        super().__init__()
        self.user_tower, self.item_tower = user_tower, item_tower
        self.item_prob = tf.constant(item_prob, tf.float32)
        self.task = tfrs.tasks.Retrieval(temperature=temperature)

    def compute_loss(self, features, training=False):
        u = self.user_tower(features["user_idx"])
        i = self.item_tower(features["item_idx"])
        q = tf.gather(self.item_prob, features["item_idx"])
        return self.task(u, i, candidate_sampling_probability=q, compute_metrics=False)


def embed_all(tower, n_rows, batch_size=65536):
    ids = np.arange(n_rows, dtype="int32")
    return tower.predict(ids, batch_size=batch_size, verbose=0).astype("float32")
