# Separate Python environments: Airflow's own dependencies stay isolated from
# TensorFlow/TFRS/Spark used by the pipeline.
FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/recsys-venv /opt/airflow/runtime \
    && chown airflow:0 /opt/recsys-venv /opt/airflow/runtime \
    && chmod 775 /opt/airflow/runtime
USER airflow
COPY --chown=airflow:0 requirements.txt /tmp/recsys-requirements.txt
RUN python -m venv /opt/recsys-venv \
    && /opt/recsys-venv/bin/pip install --no-cache-dir 'tensorflow==2.21.0' \
    && /opt/recsys-venv/bin/pip install --no-cache-dir -r /tmp/recsys-requirements.txt
