"""Nearest-neighbour index over item embeddings: FAISS (HNSW, IVF or flat), or exact numpy search if FAISS is missing."""
import json
import os

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None
    HAS_FAISS = False


class ItemIndex:
    """Indexes item rows 1..N (row 0, the unknown-item row, is left out). Returned ids are item_idx values."""

    def __init__(self, embeddings, index_type="hnsw", hnsw_m=32, ef_search=128, nlist=256, force_numpy=False):
        self.emb = np.ascontiguousarray(embeddings[1:], dtype="float32")
        self.index_type, self.ef_search = index_type, ef_search
        self.backend = "numpy"
        self.index = None
        if HAS_FAISS and not force_numpy:
            d = self.emb.shape[1]
            if index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = 200
                self.index.add(self.emb)
                self.index.hnsw.efSearch = ef_search
            elif index_type == "ivf":
                q = faiss.IndexFlatIP(d)
                self.index = faiss.IndexIVFFlat(q, d, min(nlist, max(1, len(self.emb) // 39)), faiss.METRIC_INNER_PRODUCT)
                self.index.train(self.emb)
                self.index.add(self.emb)
                self.index.nprobe = 16
            else:
                self.index = faiss.IndexFlatIP(d)
                self.index.add(self.emb)
            self.backend = f"faiss-{index_type}"

    def search(self, queries, k):
        """queries (Q, d) -> (scores, item_idx), both (Q, k). Slots FAISS could not fill have item_idx 0."""
        q = np.ascontiguousarray(queries, dtype="float32")
        k = min(k, len(self.emb))
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return scores, np.where(ids < 0, -1, ids) + 1
        return self.exact_search(q, k)

    def exact_search(self, queries, k):
        sims = queries @ self.emb.T
        k = min(k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1)
        ids = np.take_along_axis(part, order, 1)
        return np.take_along_axis(sims, ids, 1), ids + 1

    def search_unseen(self, query, k, seen=None):
        """Top-k items for one query vector, skipping the item_idx values in `seen`."""
        seen = np.asarray([] if seen is None else seen)
        scores, ids = self.search(query.reshape(1, -1), k + len(seen))
        scores, ids = scores[0], ids[0]
        keep = (ids > 0) & ~np.isin(ids, seen)
        return ids[keep][:k], scores[keep][:k]

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "item_emb.npy"), self.emb)
        json.dump({"index_type": self.index_type, "ef_search": self.ef_search}, open(os.path.join(directory, "index_meta.json"), "w"))
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "items.faiss"))

    @classmethod
    def load(cls, directory):
        emb = np.load(os.path.join(directory, "item_emb.npy"))
        meta = json.load(open(os.path.join(directory, "index_meta.json")))
        obj = cls.__new__(cls)
        obj.emb, obj.index_type, obj.ef_search, obj.index, obj.backend = emb, meta["index_type"], meta["ef_search"], None, "numpy"
        path = os.path.join(directory, "items.faiss")
        if HAS_FAISS and os.path.exists(path):
            obj.index = faiss.read_index(path)
            if meta["index_type"] == "hnsw":
                obj.index.hnsw.efSearch = meta["ef_search"]
            obj.backend = f"faiss-{meta['index_type']}"
        return obj
