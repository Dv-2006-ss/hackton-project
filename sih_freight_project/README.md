# SIH 2026 - PS 26006: Intelligent Freight Forecasting Model

## Setup
```
pip install -r requirements.txt
```

## Run order
```
python src/generate_sample_data.py   # generates prototype data
python src/forecast_model.py         # test ARIMA forecasting
python src/monte_carlo.py            # test Monte Carlo simulation
python src/vessel_recommendation.py  # test vessel recommendation logic
streamlit run src/dashboard.py       # launch the full dashboard
```

## Project structure
- `src/generate_sample_data.py` - synthetic freight rate data generator (prototype only, no official dataset was provided by SIH)
- `src/forecast_model.py` - ARIMA baseline forecasting (FR1)
- `src/monte_carlo.py` - Monte Carlo risk simulation, the project's core differentiator (FR2, FR3, FR6)
- `src/vessel_recommendation.py` - port-constraint filtering + risk-adjusted vessel ranking (FR4)
- `src/dashboard.py` - Streamlit dashboard tying everything together (FR8)

## Before the final demo
Replace `data/sample_freight_rates.csv` with real historical freight rate
data if you can source it (e.g. Baltic Dry Index sub-indices) — the pipeline
works the same either way, just point `load_data()` in `forecast_model.py`
at the new file.
