"""FastAPI service. Run: uvicorn recsys.serving.api:app --port 8080"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

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
    event: Literal["click"] = "click"


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
    return {"status": "ok", "cache": r.cache.kind, "backend": getattr(r, "backend", "keras"), "index": r.index.backend, "users": r.fs.n_users, "items": r.fs.n_items}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    res = STATE["rec"].recommend(req.user_id, req.k, req.seed_movie_ids, req.liked_genres)
    # Cache hits for known users are still eligible recommendation impressions.
    rec = STATE["rec"]
    known = req.user_id in rec.fs.user_index
    _append("impressions.jsonl", {"ts": time.time(), "request_id": res["request_id"], "user_id": req.user_id,
                                  "arm": res["arm"], "source": res["source"], "eligible_for_ab": bool(known),
                                  "items": [i["movie_id"] for i in res["items"]]})
    return res


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Do not let a fabricated/unknown request_id or unseen movie contaminate the report.
    impressions = _read("impressions.jsonl")
    matching = next((i for i in reversed(impressions) if i["request_id"] == req.request_id), None)
    if matching is None:
        raise HTTPException(404, "unknown request_id")
    if req.movie_id not in matching["items"]:
        raise HTTPException(422, "movie_id was not shown in this request")
    if any(c["request_id"] == req.request_id and c["movie_id"] == req.movie_id for c in _read("clicks.jsonl")):
        return {"status": "duplicate_ignored"}
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
    recs = [{**i, "clicked": by_req.get(i["request_id"], [])} for i in imps
            if i.get("eligible_for_ab", i.get("source") == "personalized")]
    if not recs:
        raise HTTPException(404, "no eligible personalized impressions logged yet")
    report = summarize(recs)
    report["traffic_source"] = "observed_api_logs"
    report["note"] = ("These are logged HTTP requests and submitted click events, not the offline "
                      "simulation. Real-user participation is not established by this endpoint.")
    a, b = report["arms"]["A"]["requests"], report["arms"]["B"]["requests"]
    report["sufficient_sample_for_interpretation"] = bool(a >= 100 and b >= 100)
    if a == 0 or b == 0:
        # Confidence intervals require observations from both arms; avoid JSON NaN/Infinity.
        report["request_level_ztest"] = None
        report["diff_ci95"] = [None, None]
    return report
