"""Fallbacks for users without history: popular items, or items similar in genre to what the user says they like."""
import numpy as np


class ColdStart:
    def __init__(self, fs, alpha_pop=0.2):
        self.fs = fs
        idf = np.log((1 + fs.n_items) / (1 + fs.item_genre[1:].sum(0))) + 1.0
        v = fs.item_genre * idf
        n = np.linalg.norm(v, axis=1, keepdims=True)
        self.item_vec = np.divide(v, n, out=np.zeros_like(v), where=n > 0).astype("float32")
        self.idf, self.alpha = idf.astype("float32"), alpha_pop

    def popular(self, k, exclude=None, genres=None):
        order = self.fs.pop_order
        if genres:
            mask = self._genre_mask(genres)
            order = order[mask[order]]
        if exclude is not None and len(exclude):
            order = order[~np.isin(order, exclude)]
        return order[:k]

    def _genre_mask(self, genres):
        cols = [self.fs.genres.index(g) for g in genres if g in self.fs.genres]
        if not cols:
            return np.ones(self.fs.n_items + 1, bool)
        return self.fs.item_genre[:, cols].sum(1) > 0

    def content_based(self, k, seed_items=None, genres=None, exclude=None):
        """Rank by cosine similarity to a profile built from seed items and/or liked genres, plus a small popularity bonus."""
        prof = np.zeros(self.item_vec.shape[1], "float32")
        if seed_items is not None and len(seed_items):
            prof += self.item_vec[np.asarray(seed_items)].mean(0)
        if genres:
            g = np.zeros_like(prof)
            for name in genres:
                if name in self.fs.genres:
                    g[self.fs.genres.index(name)] = 1.0
            n = np.linalg.norm(g * self.idf)
            if n > 0:
                prof += (g * self.idf) / n
        if not prof.any():
            return self.popular(k, exclude)
        n = np.linalg.norm(prof)
        sims = self.item_vec @ (prof / n)
        sims = sims + self.alpha * self.fs.item_stats[:, 0]
        sims[0] = -np.inf
        drop = list(seed_items) if seed_items is not None else []
        if exclude is not None:
            drop = list(drop) + list(exclude)
        if drop:
            sims[np.asarray(drop, int)] = -np.inf
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        return top[np.argsort(-sims[top])]
