"""Clear only recommendation cache keys after a successful scheduled retrain."""
import argparse
import os

from recsys.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    host = os.getenv("RECSYS_REDIS_HOST", cfg["serving"]["redis_host"])
    port = int(os.getenv("RECSYS_REDIS_PORT", cfg["serving"]["redis_port"]))
    import redis
    client = redis.Redis(host=host, port=port, socket_connect_timeout=5)
    client.ping()
    count = 0
    batch = []
    for key in client.scan_iter(match="rec:*"):
        batch.append(key)
        if len(batch) >= 100:
            count += client.delete(*batch)
            batch.clear()
    if batch:
        count += client.delete(*batch)
    print(f"Successfully cleared {count} recommendation-cache keys")


if __name__ == "__main__":
    main()
