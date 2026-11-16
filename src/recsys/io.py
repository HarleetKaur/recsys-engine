"""Read/write tables as parquet (Spark writes parquet directories) or csv.gz if pyarrow is missing."""
import os
import shutil

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:  # pragma: no cover
    HAS_PARQUET = False


def table_path(base, name):
    return os.path.join(base, name + (".parquet" if HAS_PARQUET else ".csv.gz"))


def write_table(df, base, name):
    os.makedirs(base, exist_ok=True)
    path = table_path(base, name)
    if os.path.isdir(path):  # a Spark run leaves a directory here
        shutil.rmtree(path)
    if HAS_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_table(base, name):
    parquet = os.path.join(base, name + ".parquet")
    if os.path.exists(parquet):  # a file (pandas) or a directory (Spark)
        return pd.read_parquet(parquet)
    csv = os.path.join(base, name + ".csv.gz")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"table '{name}' not found in {base}; run the split stage first")
