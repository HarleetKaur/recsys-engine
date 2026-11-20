"""Offline contract checks; no GPU, real Triton server or real user clicks required."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import io
import json
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from recsys.serving.scorers import TritonScorer
from recsys.serving import api


class _Response(io.BytesIO):
    def __init__(self, object_):
        super().__init__(json.dumps(object_).encode())


class TestTritonV2Contract(unittest.TestCase):
    def setUp(self):
        self.bodies = []

        def server(request, timeout=30):
            if request.get_method() == "GET":
                return _Response({"inputs": [
                    {"name": "item_idx", "datatype": "INT32", "shape": [-1, 1]},
                    {"name": "user_idx", "datatype": "INT32", "shape": [-1, 1]}],
                    "outputs": [{"name": "score", "datatype": "FP32", "shape": [-1, 1]}]})
            request_body = json.loads(request.data)
            self.bodies.append(request_body)
            n = request_body["inputs"][0]["shape"][0]
            return _Response({"outputs": [{"name": "score", "datatype": "FP32",
                                          "shape": [n, 1], "data": [0.25] * n}]})

        self.mock = patch("recsys.serving.scorers.urlopen", side_effect=server)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_two_item_batch_uses_correct_shape(self):
        scorer = TritonScorer("triton:8000", "ranker_dcn")
        np.testing.assert_array_equal(scorer.score(1, [2, 3]), [0.25, 0.25])
        self.assertEqual([p["shape"] for p in self.bodies[-1]["inputs"]], [[2, 1], [2, 1]])
        self.assertEqual(self.bodies[-1]["inputs"][0]["data"], [1, 1])

    def test_per_request_client_thread_safe(self):
        scorer = TritonScorer("triton:8000", "ranker_wide_deep")
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(lambda n: scorer.score(n, [1, 2]), range(40)))
        self.assertEqual(len(result), 40)
        self.assertTrue(all(len(r) == 2 for r in result))

    def test_mismatched_inputs_rejected(self):
        scorer = TritonScorer("triton:8000", "ranker_dcn")
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            scorer.score_batch([1], [1, 2])
        self.assertEqual(len(scorer.score_batch([], [])), 0)


class TestRealHTTPTransport(unittest.TestCase):
    def test_parallel_triton_v2_http_calls_over_local_server(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                data = {"inputs": [{"name": "user_idx", "datatype": "INT32", "shape": [-1, 1]},
                                   {"name": "item_idx", "datatype": "INT32", "shape": [-1, 1]}],
                        "outputs": [{"name": "score", "datatype": "FP32", "shape": [-1, 1]}]}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = data["inputs"][0]["shape"][0]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"outputs": [{"name": "score", "shape": [n, 1],
                                                        "data": [0.73] * n}]}).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            scorer = TritonScorer(f"127.0.0.1:{server.server_port}", "ranker_dcn")
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda x: scorer.score(x, [1, 2, 3]), range(24)))
            self.assertEqual(len(responses), 24)
            self.assertTrue(all(np.allclose(r, [0.73] * 3) for r in responses))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class _FS:
    user_index = {1: 1}
    n_users = 1
    n_items = 3


class _Rec:
    fs = _FS()
    backend = "triton"
    index = types.SimpleNamespace(backend="faiss-hnsw")
    cache = types.SimpleNamespace(kind="redis")

    def recommend(self, user_id, k, seed_movie_ids=None, liked_genres=None):
        return {"request_id": "observed123", "user_id": user_id, "arm": "A", "source": "personalized",
                "items": [{"movie_id": 5, "score": 0.8}, {"movie_id": 7, "score": 0.4}], "latency_ms": 3}


class TestObservedAB(unittest.TestCase):
    def test_feedback_must_reference_shown_movie_and_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as root:
            old = dict(api.STATE)
            api.STATE.clear()
            api.STATE.update({"rec": _Rec(), "log_dir": root})
            try:
                client = TestClient(api.app)
                r = client.post("/recommend", json={"user_id": 1, "k": 2})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(client.get("/health").json()["backend"], "triton")
                self.assertEqual(client.post("/feedback", json={"request_id": "bad", "movie_id": 5}).status_code, 404)
                self.assertEqual(client.post("/feedback", json={"request_id": "observed123", "movie_id": 999}).status_code, 422)
                ok = client.post("/feedback", json={"request_id": "observed123", "movie_id": 5})
                self.assertEqual(ok.json()["status"], "recorded")
                dup = client.post("/feedback", json={"request_id": "observed123", "movie_id": 5})
                self.assertEqual(dup.json()["status"], "duplicate_ignored")
                rep = client.get("/ab/report")
                self.assertEqual(rep.status_code, 200)
                self.assertEqual(rep.json()["arms"]["A"]["clicks"], 1)
                self.assertEqual(rep.json()["traffic_source"], "observed_api_logs")
                self.assertFalse(rep.json()["sufficient_sample_for_interpretation"])
            finally:
                api.STATE.clear()
                api.STATE.update(old)


class TestAirflowDAGOffline(unittest.TestCase):
    def test_dag_has_sequential_retrain_and_cache_invalidation(self):
        class FakeDAG:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
        class FakeBashOperator:
            def __init__(self, task_id, bash_command):
                self.task_id, self.bash_command = task_id, bash_command
            def __rshift__(self, other):
                return other
        fake_airflow = types.ModuleType("airflow")
        fake_airflow.DAG = FakeDAG
        operators = types.ModuleType("airflow.operators")
        bash = types.ModuleType("airflow.operators.bash")
        bash.BashOperator = FakeBashOperator
        path = Path(__file__).resolve().parents[1] / "airflow/dags/recsys_retrain_dag.py"
        with patch.dict(sys.modules, {"airflow": fake_airflow, "airflow.operators": operators,
                                      "airflow.operators.bash": bash}):
            spec = importlib.util.spec_from_file_location("_recsys_dag_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(len(module.chain), 10)
        self.assertEqual(module.chain[-1].task_id, "flush_cache")
        self.assertEqual(module.dag.kwargs["schedule"], "@weekly")
        self.assertIn("recsys.cli", module.chain[0].bash_command)
        self.assertIn("clear_recsys_cache.py", module.chain[-1].bash_command)


if __name__ == "__main__":
    unittest.main()
