"""Build a Triton model repository from the ranker SavedModels (TensorFlow backend)."""
import json
import os
import shutil

# max_batch_size 0: the models take 1-D id tensors whose length is the batch, so Triton does no extra batching.
# Input/output specs are left out on purpose; Triton fills them in from the SavedModel signature.
CONFIG = '''name: "{name}"
platform: "tensorflow_savedmodel"
max_batch_size: 0
instance_group [ {{ count: 1 kind: KIND_{kind} }} ]
'''


def _tf_version():
    try:
        import tensorflow as tf
        return tf.__version__
    except ImportError:
        return "unknown"


def export_model_repository(src_dir, repo_dir, variants, kind="CPU"):
    for v in variants:
        dst = os.path.join(repo_dir, f"ranker_{v}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(src_dir, v), os.path.join(dst, "1", "model.savedmodel"))
        with open(os.path.join(dst, "config.pbtxt"), "w") as f:
            f.write(CONFIG.format(name=f"ranker_{v}", kind=kind))
        print(f"[triton] exported {dst}")
    # Triton's TF backend must be the same TF version that exported the models; record what was used.
    with open(os.path.join(repo_dir, "EXPORT_INFO.json"), "w") as f:
        json.dump({"tensorflow_version": _tf_version(), "models": [f"ranker_{v}" for v in variants]}, f, indent=2)
