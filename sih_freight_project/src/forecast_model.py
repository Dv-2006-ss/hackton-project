"""
Baseline forecasting model (FR1).

Uses ARIMA (statsmodels) to forecast future freight rates from historical
data. This produces the "trend" that the Monte Carlo engine (FR2) will
add realistic randomness around.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Point to real historical BDRY market proxy data, with automatic fallback
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bdry_historical.csv"


def load_data(path=None):
    """
    Loads daily historical freight rates.
    Defaults to cached BDRY market proxy data (Breakwave Dry Bulk Shipping ETF).
    """
    if path is not None:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if df.index.freq is None:
            df = df.asfreq("D").ffill()
        return df["freight_rate_usd_per_ton"]

    try:
        from data_loader import load_freight_data
        return load_freight_data(cache_path=DEFAULT_DATA_PATH)
    except Exception:
        df = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["date"]).set_index("date")
        if df.index.freq is None:
            df = df.asfreq("D").ffill()
        return df["freight_rate_usd_per_ton"]


def fit_arima_and_forecast(series, forecast_days=30, order=(2, 1, 2)):
    """
    Fits ARIMA(2, 1, 2) for the conditional mean trend, and fits GARCH(1, 1)
    on the ARIMA residuals to model time-varying volatility clustering.

    Returns:
      - forecast: predicted mean rate for each future day (pandas Series)
      - conf_int: statsmodels' parametric confidence interval
      - volatility_forecast: array of day-by-day conditional volatility (std dev) from GARCH(1, 1)
      - fitted: fitted ARIMA results object (with .garch_fit and .flat_residual_std attached)
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=forecast_days)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.2)  # 80% CI

    # Baseline flat residual std dev
    flat_std = float(np.std(fitted.resid))
    fitted.flat_residual_std = flat_std

    # GARCH(1, 1) modeling on ARIMA residuals
    try:
        from arch import arch_model
        resid = fitted.resid
        # Zero-mean GARCH(1, 1) since residuals have mean zero
        garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1)
        garch_fit = garch.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_days)
        variance_forecast = garch_forecast.variance.iloc[-1].values
        volatility_forecast = np.sqrt(variance_forecast)
        fitted.garch_fit = garch_fit
    except Exception as e:
        warnings.warn(f"GARCH fitting fallback to constant volatility: {e}")
        volatility_forecast = np.full(forecast_days, flat_std)
        fitted.garch_fit = None

    return forecast, conf_int, volatility_forecast, fitted


if __name__ == "__main__":
    series = load_data()
    forecast, conf_int, volatility_forecast, fitted = fit_arima_and_forecast(series)

    print("Model summary (AIC as fit-quality indicator):", fitted.aic)
    flat_std = getattr(fitted, "flat_residual_std", float(np.mean(volatility_forecast)))
    print(f"Old flat residual std dev: {flat_std:.3f}")

    if hasattr(fitted, "garch_fit") and fitted.garch_fit is not None:
        params = fitted.garch_fit.params
        print("\nGARCH(1, 1) Fitted Parameters:")
        print(f"  omega   : {params.get('omega', 0):.6f}")
        print(f"  alpha[1]: {params.get('alpha[1]', 0):.6f} (recent shock impact)")
        print(f"  beta[1] : {params.get('beta[1]', 0):.6f} (volatility persistence)")
        print(f"  alpha+beta: {params.get('alpha[1]', 0) + params.get('beta[1]', 0):.6f}")

    print("\n30-day Day-by-Day Volatility: Old Flat vs New GARCH(1, 1):")
    print(f"{'Day':<6} {'Date':<12} {'Old Flat Vol':<15} {'New GARCH Vol':<15} {'Difference':<12}")
    print("-" * 62)
    for idx, (dt, g_vol) in enumerate(zip(forecast.index, volatility_forecast), 1):
        diff = g_vol - flat_std
        print(f"{idx:<6} {dt.date()}   {flat_std:<15.4f} {g_vol:<15.4f} {diff:+.4f}")

    print("\nNext 10 days mean forecast:")
    print(forecast.head(10))
