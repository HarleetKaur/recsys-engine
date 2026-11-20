"""Score user/item pairs with either a local TensorFlow SavedModel or Triton's V2 HTTP API.

The Triton path deliberately uses Python's stdlib HTTP client: unlike a shared
tritonclient.http client, it does not keep a gevent connection/greenlet across
FastAPI worker threads. No mutable HTTP session is shared between requests.
"""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class SavedModelScorer:
    """Load a SavedModel and call its serving_default signature."""

    def __init__(self, path, batch_size=65536):
        import tensorflow as tf

        self._tf = tf
        self._loaded = tf.saved_model.load(path)
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
    """Thread-safe per-request HTTP inference with Triton's documented V2 JSON protocol."""

    def __init__(self, url, model_name, timeout=30):
        self.base_url = (url if "://" in url else "http://" + url).rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        meta = self._json("GET", f"/v2/models/{model_name}")
        inputs = meta["inputs"]
        self.in_user = next(i["name"] for i in inputs if "user" in i["name"])
        self.in_item = next(i["name"] for i in inputs if "item" in i["name"])
        self.out = meta["outputs"][0]["name"]
        # The exported rankers have inputs [-1, 1] and output [-1, 1].
        for inp in inputs:
            if inp["name"] in (self.in_user, self.in_item):
                if inp["datatype"] != "INT32" or inp["shape"][-1:] != [1]:
                    raise ValueError(f"Unexpected Triton input format for {inp['name']}: {inp}")

    def _json(self, method, endpoint, body=None):
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(self.base_url + endpoint, data=payload, method=method,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read(1500).decode("utf-8", errors="replace")
            raise RuntimeError(f"Triton {self.model_name} returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot reach Triton {self.model_name} at {self.base_url}: {exc}") from exc

    def score_batch(self, user_idx, item_idx):
        users = np.asarray(user_idx, dtype="int32").reshape(-1)
        items = np.asarray(item_idx, dtype="int32").reshape(-1)

        if len(users) != len(items):
            raise ValueError("User and item lengths differ")
        if len(items) == 0:
            return np.zeros(0, dtype="float32")

        chunks = []
        for start in range(0, len(items), 4):
            u = users[start:start + 4]
            i = items[start:start + 4]
            n = len(i)

            inputs = [
                {"name": self.in_user, "datatype": "INT32",
                 "shape": [n, 1], "data": u.tolist()},
                {"name": self.in_item, "datatype": "INT32",
                 "shape": [n, 1], "data": i.tolist()}
            ]

            data = self._json(
                "POST",
                f"/v2/models/{self.model_name}/infer",
                {"inputs": inputs, "outputs": [{"name": self.out}]}
            )

            output = next(
                o for o in data["outputs"] if o["name"] == self.out
            )
            scores = np.asarray(
                output["data"], dtype="float32"
            ).reshape(-1)

            if len(scores) != n or not np.isfinite(scores).all():
                raise ValueError("Invalid Triton scores")

            chunks.append(scores)

        return np.concatenate(chunks)

    def score(self, user_idx, item_idx):
        items = np.asarray(item_idx, "int32")
        return self.score_batch(np.full(len(items), user_idx, "int32"), items)
