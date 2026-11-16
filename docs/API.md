# API

Start with `uvicorn recsys.serving.api:app --port 8080` (the artifacts from a pipeline run must exist).

## GET /health
Returns the cache backend (`redis` or `memory`), the index backend and the number of users and items.

## POST /recommend
```json
{"user_id": 42, "k": 10, "seed_movie_ids": [1, 260], "liked_genres": ["Sci-Fi"]}
```
`seed_movie_ids` and `liked_genres` are optional and only used for users the model has never seen.

The response contains `request_id`, `user_id`, `arm` (A or B), `source`, `model_version`, `latency_ms` and `items` (a list of `movie_id`, `title` and, for personalised results, `score`).
`source` tells how the list was produced:

| source | meaning |
|---|---|
| `personalized` | FAISS candidates re-scored by the ranker of the user's arm |
| `cache` | the same list served from Redis |
| `cold_content` | unknown user, built from the given seed movies or genres |
| `cold_popularity` | unknown user, nothing given: globally popular movies |

## POST /feedback
```json
{"request_id": "...", "movie_id": 318, "event": "click"}
```
Appends a line to `logs/clicks.jsonl`. Each `/recommend` call writes a line to `logs/impressions.jsonl`.

## GET /ab/report
Joins impressions and clicks of personalised requests and returns, per arm, the number of requests, impressions and clicks, the click rate, a two-proportion z-test on the per-request click rate and a 95% bootstrap interval for the difference. Returns 404 until at least one personalised request has been logged.
