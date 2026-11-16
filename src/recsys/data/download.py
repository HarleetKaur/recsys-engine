"""Download MovieLens from GroupLens and unpack it into data/raw/."""
import os
import urllib.request
import zipfile

URLS = {
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}


def download_movielens(raw_dir, dataset="ml-25m"):
    target = os.path.join(raw_dir, dataset)
    if os.path.exists(os.path.join(target, "ratings.csv")):
        print(f"[download] {dataset} already present at {target}")
        return target
    os.makedirs(raw_dir, exist_ok=True)
    zpath = os.path.join(raw_dir, f"{dataset}.zip")
    print(f"[download] fetching {URLS[dataset]}")
    urllib.request.urlretrieve(URLS[dataset], zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw_dir)
    os.remove(zpath)
    if not os.path.exists(os.path.join(target, "ratings.csv")):
        raise RuntimeError(f"expected {target}/ratings.csv after extraction")
    return target
