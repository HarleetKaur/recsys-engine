"""Two-stage recommender: candidate retrieval followed by ranking."""
import os

# TensorFlow Recommenders still expects Keras 2. With TF >= 2.16 (e.g. current Colab) that means
# selecting the legacy tf-keras package, which must happen before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__version__ = "1.0.0"
