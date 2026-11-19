# Advanced Infrastructure Verification

Date: 19 November 2026

## Apache Airflow
- Airflow 3.1.6 Docker image used for validation.
- Retraining DAG loaded successfully.
- All nine tasks were recognized.
- All task dependencies passed structural checks.
- Weekly schedule configured.
- Actual scheduled retraining not yet verified.

## Triton Inference Server
- Docker Compose includes Triton.
- Model configuration files are present.
- Exported model weights not found locally.
- Live Triton inference not yet verified.

## A/B Testing
- Saved MovieLens 25M subset replay inspected.
- Model A: 180 impressions, 3 clicks.
- Model B: 170 impressions, 7 clicks.
- Impression CTR: A 1.67%, B 4.12%.
- Request-level test p-value: approximately 0.612.
- Difference not statistically significant.
- Simulated replay, not live user traffic.

## Scope
Structural checks and saved experiment results
do not establish a production deployment.
