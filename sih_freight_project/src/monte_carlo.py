"""
Monte Carlo simulation engine (FR2) - the project's core differentiator.

Takes the ARIMA baseline forecast + residual volatility, and simulates
many possible future price paths by adding randomized daily shocks.
Instead of one predicted number, this produces a full probability
distribution for every future day - which is what lets us answer:

  - FR3: which entry day has the highest probability of being cheapest
  - FR4: risk-adjusted cost per vessel type
  - FR6: how wide the outcome spread is (= volatility / risk flag)
  - FR7: spot booking vs multi-voyage contract comparison
"""

import numpy as np
import pandas as pd

from forecast_model import load_data, fit_arima_and_forecast


def run_monte_carlo(forecast_mean, residual_std, n_simulations=1000, seed=42):
    """
    forecast_mean: pandas Series, ARIMA predicted mean for each future day
    residual_std: float OR array-like (length forecast_days) of daily conditional
                  volatilities from GARCH(1, 1).

    Returns a DataFrame of shape (n_simulations, forecast_days) - each row
    is one simulated possible future price path.
    """
    rng = np.random.default_rng(seed)
    forecast_days = len(forecast_mean)

    vol_vector = residual_std.values if hasattr(residual_std, "values") else residual_std

    # Each simulated path = baseline forecast + cumulative random walk noise
    # Daily shocks use time-varying GARCH volatility for that day
    simulations = np.zeros((n_simulations, forecast_days))
    for i in range(n_simulations):
        daily_shocks = rng.normal(0, vol_vector, size=forecast_days)
        cumulative_walk = np.cumsum(daily_shocks) * 0.3  # dampened random walk
        simulations[i] = forecast_mean.values + cumulative_walk

    simulations = np.clip(simulations, 1, None)  # rates can't go negative
    sim_df = pd.DataFrame(simulations, columns=forecast_mean.index)
    return sim_df


def summarize_simulation(sim_df):
    """
    Produces the probability-based summary used by the recommendation layer:
      - mean, p10, p50 (median), p90 per day
      - probability each day is the cheapest day in the window
      - overall volatility score (risk flag input)
    """
    summary = pd.DataFrame({
        "mean": sim_df.mean(axis=0),
        "p10": sim_df.quantile(0.10, axis=0),
        "p50_median": sim_df.quantile(0.50, axis=0),
        "p90": sim_df.quantile(0.90, axis=0),
    })

    # probability each day is the cheapest across simulations
    cheapest_day_idx = sim_df.values.argmin(axis=1)
    day_cols = sim_df.columns
    prob_cheapest = pd.Series(0.0, index=day_cols)
    counts = pd.Series(cheapest_day_idx).value_counts(normalize=True)
    for idx, prob in counts.items():
        prob_cheapest.iloc[idx] = prob
    summary["prob_cheapest_day"] = prob_cheapest.values

    volatility_score = float(sim_df.std(axis=0).mean())

    return summary, volatility_score


def risk_flag(volatility_score, low_thresh=1.5, high_thresh=3.5):
    if volatility_score < low_thresh:
        return "LOW volatility - market conditions stable"
    elif volatility_score < high_thresh:
        return "MODERATE volatility - normal market fluctuation"
    else:
        return "HIGH volatility - caution advised, consider hedging/waiting"


if __name__ == "__main__":
    series = load_data()
    forecast_mean, conf_int, residual_std, fitted = fit_arima_and_forecast(series, forecast_days=30)

    sim_df = run_monte_carlo(forecast_mean, residual_std, n_simulations=1000)
    summary, volatility_score = summarize_simulation(sim_df)

    print("Simulation summary (first 10 days):")
    print(summary.head(10).round(2))

    best_day = summary["prob_cheapest_day"].idxmax()
    best_prob = summary["prob_cheapest_day"].max()
    print(f"\nBest entry day: {best_day.date()} "
          f"(probability of being cheapest: {best_prob:.1%})")

    print(f"\nVolatility score: {volatility_score:.2f}")
    print(f"Risk flag: {risk_flag(volatility_score)}")
