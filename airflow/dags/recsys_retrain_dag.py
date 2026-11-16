"""Weekly retraining: data -> features -> retrieval -> index -> rankers -> evaluation -> Triton export.

Set RECSYS_HOME to the repo path on the Airflow worker (default /opt/recsys-engine).
"""
import os
from datetime import datetime, timedelta

from airflow import DAG

try:  # Airflow 3 moved the bash operator into the standard provider
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2
    from airflow.operators.bash import BashOperator

HOME = os.environ.get("RECSYS_HOME", "/opt/recsys-engine")
RUN = f"cd {HOME} && RECSYS_HOME={HOME} PYTHONPATH=src python -m recsys.cli "
STAGES = ["download", "split", "features", "train_retrieval", "build_index",
          "train_ranker", "evaluate", "export_triton"]

with DAG(
    dag_id="recsys_weekly_retrain",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["recsys"],
) as dag:
    tasks = [BashOperator(task_id=name, bash_command=RUN + name) for name in STAGES]
    # New models should not be hidden behind old cached results.
    flush_cache = BashOperator(
        task_id="flush_cache",
        bash_command="redis-cli --scan --pattern 'rec:*' | xargs -r redis-cli del",
    )
    chain = tasks + [flush_cache]
    for upstream, downstream in zip(chain, chain[1:]):
        upstream >> downstream
