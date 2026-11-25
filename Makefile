install:
	pip install -r requirements.txt
test:
	PYTHONPATH=src pytest -q
baselines:
	PYTHONPATH=src python scripts/run_baselines.py
