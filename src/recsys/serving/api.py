"""FastAPI service. Run: uvicorn recsys.serving.api:app --port 8080"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.config import load_config
from recsys.evaluation.ab_test import summarize

STATE = {}
_lock = threading.Lock()


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(10, ge=1, le=100)
    seed_movie_ids: Optional[List[int]] = None   # cold-start: movies the new user likes
    liked_genres: Optional[List[str]] = None     # cold-start: genres the new user likes


class FeedbackRequest(BaseModel):
    request_id: str
    movie_id: int
    event: str = "click"


@asynccontextmanager
async def lifespan(app):
    from recsys.pipeline.stages import build_recommender
    cfg = load_config()
    STATE["cfg"], STATE["rec"] = cfg, build_recommender(cfg)
    STATE["log_dir"] = cfg["serving"]["log_dir"]
    os.makedirs(STATE["log_dir"], exist_ok=True)
    yield


app = FastAPI(title="Two-stage Recommendation Engine", version="1.0.0", lifespan=lifespan)


def _append(name, rec):
    with _lock, open(os.path.join(STATE["log_dir"], name), "a") as f:
        f.write(json.dumps(rec) + "\n")


@app.get("/health")
def health():
    r = STATE["rec"]
    return {"status": "ok", "cache": r.cache.kind, "index": r.index.backend, "users": r.fs.n_users, "items": r.fs.n_items}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    res = STATE["rec"].recommend(req.user_id, req.k, req.seed_movie_ids, req.liked_genres)
    _append("impressions.jsonl", {"ts": time.time(), "request_id": res["request_id"], "user_id": req.user_id,
                                  "arm": res["arm"], "source": res["source"], "items": [i["movie_id"] for i in res["items"]]})
    return res


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    _append("clicks.jsonl", {"ts": time.time(), **req.model_dump()})
    return {"status": "recorded"}


def _read(name):
    p = os.path.join(STATE["log_dir"], name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


@app.get("/ab/report")
def ab_report():
    imps, clicks = _read("impressions.jsonl"), _read("clicks.jsonl")
    by_req = {}
    for c in clicks:
        by_req.setdefault(c["request_id"], []).append(c["movie_id"])
    recs = [{**i, "clicked": by_req.get(i["request_id"], [])} for i in imps if i["source"] == "personalized"]
    if not recs:
        raise HTTPException(404, "no personalized impressions logged yet")
    return summarize(recs)
