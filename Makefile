export PYTHONPATH := src:tests

test:        ## unit tests (no TensorFlow needed)
	python -m unittest discover -s tests -v

smoke:       ## every stage on tiny synthetic data (needs the full requirements)
	python scripts/smoke_test.py

pipeline:    ## full run on the dataset set in configs/config.yaml
	python -m recsys.cli all

report:      ## docs/RESULTS.md and reports/ndcg.png from reports/*.json
	python scripts/make_report.py

serve:
	uvicorn recsys.serving.api:app --port 8080
