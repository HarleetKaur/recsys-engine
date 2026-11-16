"""Result cache: Redis when reachable, otherwise a small in-process dict with the same interface."""
import json
import time


class InMemoryCache:
    kind = "memory"

    def __init__(self):
        self._d = {}

    def get(self, key):
        v = self._d.get(key)
        if v is None or v[0] < time.time():
            self._d.pop(key, None)
            return None
        return json.loads(v[1])

    def set(self, key, value, ttl):
        self._d[key] = (time.time() + ttl, json.dumps(value))

    def clear(self):
        self._d.clear()


class RedisCache:
    kind = "redis"

    def __init__(self, host="localhost", port=6379):
        import redis

        self.r = redis.Redis(host=host, port=port, socket_connect_timeout=1)
        self.r.ping()

    def get(self, key):
        v = self.r.get(key)
        return None if v is None else json.loads(v)

    def set(self, key, value, ttl):
        self.r.setex(key, int(ttl), json.dumps(value))

    def clear(self):
        for k in self.r.scan_iter("rec:*"):
            self.r.delete(k)


def make_cache(cfg_serving):
    try:
        return RedisCache(cfg_serving["redis_host"], cfg_serving["redis_port"])
    except Exception as e:  # redis package missing or server down
        print(f"[cache] Redis unavailable ({type(e).__name__}); using in-memory cache")
        return InMemoryCache()
