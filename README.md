\# Demand Forecasting Pipeline



Weekly demand forecasting for 500 store-item series, built as an end-to-end ML system: leakage-free validation, a LightGBM model, experiment tracking, a REST API, Docker, and CI.



!\[API demo](docs/api\_demo.png)



\## Result



On a held-out final fold (8-week horizon, 500 series), LightGBM cut error by \*\*38%\*\* versus the best baseline.



| Model | Holdout WAPE |

|---|---|

| \*\*LightGBM (num\_leaves=63)\*\* | \*\*0.0433\*\* |

| Seasonal naive (same week last year) | 0.0703 |

| Moving average (4 weeks) | 0.1779 |

| Naive (last value) | 0.2026 |



Data: Kaggle \*Store Item Demand Forecasting Challenge\* (50 items, 10 stores, daily sales aggregated to weekly).

Note: this is retail data and it is very regular, which is why even the seasonal baseline scores well. The relative improvement is the meaningful figure.



\## Design choices



\- \*\*Rolling-origin validation.\*\* Training data always comes before the test window, with several folds. No shuffling.

\- \*\*Selection vs holdout.\*\* The 4 older folds picked the model config. The newest fold was held out and reported once, so the result is not tuned against.

\- \*\*Leakage test.\*\* A unit test corrupts all data after the forecast origin and checks that the features do not change.

\- \*\*One global model.\*\* A single direct multi-horizon LightGBM serves all 500 series. Targets are scaled by each series' recent level so small and large items share one model.

\- \*\*Hyperparameters barely mattered.\*\* Six configs scored within 0.0396 to 0.0405 WAPE on the selection folds. The gain came from features and validation design.



\## Run it (Windows)



```

pip install -r requirements.txt

set PYTHONPATH=src

python -m pytest -q

python scripts\\run\_baselines.py

python scripts\\track\_experiments.py

```



Serve forecasts:



```

python scripts\\train\_final.py

uvicorn forecasting.api:app --port 8000

```



Then open http://localhost:8000/docs and try `GET /forecast?store=1\&item=5`.



Browse experiment runs: `mlflow ui --backend-store-uri sqlite:///mlruns.db`



\## Docker



```

docker build -t demand-forecasting .

docker run -p 8000:8000 demand-forecasting

```



To use the real data, download `train.csv` from Kaggle into the `data/` folder. Without it, the code uses synthetic demo data.



\## Project structure



\- `src/forecasting/` data loading, metrics, baselines, validation, features, model, API

\- `scripts/` baselines, experiments, final training

\- `tests/` unit and API tests (run in CI with ruff)



\## Roadmap



\- \[x] Baselines, rolling validation, LightGBM

\- \[x] MLflow tracking

\- \[x] FastAPI, Docker, CI

\- \[ ] Drift monitoring (Evidently) and a Streamlit dashboard

