"""Spark job: clean the ratings, split by time, and build the feature tables.

It writes the same tables as the pandas path in recsys.data.split:
train, val, test, user_stats, item_stats, user_genre_counts (parquet, in processed_dir).
"""
import os

NO_GENRE = "(no genres listed)"


def run_spark(raw_dir, processed_dir, cfg_data):
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import DoubleType, IntegerType, LongType, StructField, StructType

    spark = (
        SparkSession.builder.appName("recsys-features")
        .master("local[*]")
        .config("spark.driver.memory", cfg_data.get("spark_driver_memory", "8g"))
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    base = os.path.join(raw_dir, cfg_data["dataset"])
    schema = StructType([
        StructField("userId", IntegerType()), StructField("movieId", IntegerType()),
        StructField("rating", DoubleType()), StructField("timestamp", LongType()),
    ])
    df = spark.read.csv(os.path.join(base, "ratings.csv"), header=True, schema=schema)
    df = df.dropna().dropDuplicates(["userId", "movieId"])
    df = df.filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))

    # Keep users with enough ratings; optionally take a fixed-size random sample of them.
    users = df.groupBy("userId").count().filter(F.col("count") >= cfg_data["min_user_interactions"])
    if cfg_data.get("max_users"):
        users = users.orderBy(F.rand(cfg_data["seed"])).limit(int(cfg_data["max_users"]))
    df = df.join(users.select("userId"), "userId").cache()

    t_val, t_test = df.approxQuantile("timestamp", [cfg_data["val_quantile"], cfg_data["test_quantile"]], 0.0005)
    train = df.filter(F.col("timestamp") <= t_val)
    val = df.filter((F.col("timestamp") > t_val) & (F.col("timestamp") <= t_test))
    test = df.filter(F.col("timestamp") > t_test)

    thr = float(cfg_data["positive_threshold"])
    positive = (F.col("rating") >= thr).cast("double")

    def stats(key):
        return train.groupBy(key).agg(
            F.count("*").alias("n_ratings"), F.avg("rating").alias("mean_rating"), F.avg(positive).alias("pos_rate")
        )

    movies = spark.read.csv(os.path.join(base, "movies.csv"), header=True, inferSchema=True, quote='"', escape='"')
    item_genres = movies.select("movieId", F.explode(F.split(F.col("genres"), r"\|")).alias("genre"))
    item_genres = item_genres.filter(F.col("genre") != NO_GENRE)
    user_genre_counts = (
        train.filter(F.col("rating") >= thr).join(item_genres, "movieId")
        .groupBy("userId", "genre").agg(F.count("*").alias("n_pos"))
    )

    tables = {"train": train, "val": val, "test": test, "user_stats": stats("userId"),
              "item_stats": stats("movieId"), "user_genre_counts": user_genre_counts}
    os.makedirs(processed_dir, exist_ok=True)
    for name, frame in tables.items():
        frame.write.mode("overwrite").parquet(os.path.join(processed_dir, f"{name}.parquet"))
        print(f"[spark] wrote {name}: {frame.count():,} rows")
    spark.stop()
