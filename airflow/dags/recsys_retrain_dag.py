"""Airflow 2/3 weekly retraining DAG.

Requires an Airflow worker/container with the ML pipeline dependencies installed
in RECSYS_PIPELINE_PYTHON. See docs/ADVANCED_INTEGRATION.md.
This DAG schedules the job; it is not evidence that a scheduler has run it.
"""
import os
import shlex
from datetime import datetime, timedelta

from airflow import DAG

try:
    from airflow.providers.standard.operators.bash import BashOperator  # Airflow 3
except ImportError:
    from airflow.operators.bash import BashOperator  # Airflow 2

HOME = os.environ.get("RECSYS_HOME", "/opt/recsys-engine")
PYTHON = os.environ.get("RECSYS_PIPELINE_PYTHON", "python")
CONFIG = os.environ.get("RECSYS_CONFIG", os.path.join(HOME, "configs", "config.yaml"))
REDIS_HOST = os.environ.get("RECSYS_REDIS_HOST", "redis")
STAGES = ["download", "split", "features", "train_retrieval", "build_index",
          "train_ranker", "evaluate", "ab_replay", "export_triton"]


def stage_command(stage):
    command = " ".join(map(shlex.quote, [PYTHON, "-m", "recsys.cli", stage, "--config", CONFIG]))
    return f"cd {shlex.quote(HOME)} && PYTHONPATH={shlex.quote(os.path.join(HOME, 'src'))} {command}"


with DAG(
    dag_id="recsys_weekly_retrain",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["recsys"],
) as dag:
    tasks = [BashOperator(task_id=stage, bash_command=stage_command(stage)) for stage in STAGES]
    flush_cache = BashOperator(
        task_id="flush_cache",
        bash_command=(f"cd {shlex.quote(HOME)} && "
                      f"PYTHONPATH={shlex.quote(os.path.join(HOME, 'src'))} "
                      f"RECSYS_REDIS_HOST={shlex.quote(REDIS_HOST)} "
                      f"{shlex.quote(PYTHON)} scripts/clear_recsys_cache.py --config {shlex.quote(CONFIG)}"),
    )
    chain = tasks + [flush_cache]
    for upstream, downstream in zip(chain, chain[1:]):
        upstream >> downstream
