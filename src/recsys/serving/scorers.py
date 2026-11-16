"""Ways to score (user, item) pairs with a ranker: a local SavedModel or a Triton server."""
import numpy as np


class SavedModelScorer:
    """Loads a SavedModel and calls its serving_default signature - the same function Triton would serve."""

    def __init__(self, path, batch_size=65536):
        import tensorflow as tf

        self._tf = tf
        self._loaded = tf.saved_model.load(path)  # keep a reference or the signature is garbage collected
        self.fn = self._loaded.signatures["serving_default"]
        names = list(self.fn.structured_input_signature[1].keys())
        self.in_user = next(n for n in names if "user" in n)
        self.in_item = next(n for n in names if "item" in n)
        self.batch_size = batch_size

    def score_batch(self, user_idx, item_idx):
        tf, out = self._tf, []
        user_idx, item_idx = np.asarray(user_idx, "int32"), np.asarray(item_idx, "int32")
        for s in range(0, len(item_idx), self.batch_size):
            res = self.fn(**{self.in_user: tf.constant(user_idx[s:s + self.batch_size]),
                             self.in_item: tf.constant(item_idx[s:s + self.batch_size])})
            out.append(next(iter(res.values())).numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, "float32")

    def score(self, user_idx, item_idx):
        item_idx = np.asarray(item_idx, "int32")
        return self.score_batch(np.full(len(item_idx), user_idx, "int32"), item_idx)


class TritonScorer:
    """Calls a ranker hosted on Triton. Input/output names come from the model metadata."""

    def __init__(self, url, model_name):
        import tritonclient.http as httpclient

        self.http, self.model_name = httpclient, model_name
        self.client = httpclient.InferenceServerClient(url=url)
        meta = self.client.get_model_metadata(model_name)
        self.in_user = next(i["name"] for i in meta["inputs"] if "user" in i["name"])
        self.in_item = next(i["name"] for i in meta["inputs"] if "item" in i["name"])
        self.out = meta["outputs"][0]["name"]

    def score_batch(self, user_idx, item_idx):
        h = self.http
        user_idx, item_idx = np.asarray(user_idx, "int32"), np.asarray(item_idx, "int32")
        u = h.InferInput(self.in_user, [len(user_idx)], "INT32")
        u.set_data_from_numpy(user_idx)
        i = h.InferInput(self.in_item, [len(item_idx)], "INT32")
        i.set_data_from_numpy(item_idx)
        res = self.client.infer(self.model_name, [u, i], outputs=[h.InferRequestedOutput(self.out)])
        return res.as_numpy(self.out).reshape(-1)

    def score(self, user_idx, item_idx):
        item_idx = np.asarray(item_idx, "int32")
        return self.score_batch(np.full(len(item_idx), user_idx, "int32"), item_idx)
