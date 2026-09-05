"""
FR5 - Idle Scenario Management
FR7 - Spot Booking vs Multi-Voyage Contract Comparison

Both use the same Monte Carlo simulation output already built.
"""

import numpy as np


def estimate_idle_risk(summary, volatility_score, demand_threshold_percentile=25):
    """
    FR5: Estimate probability/risk of the vessel sitting idle.

    Approach: low forecasted rates combined with high volatility often
    signal weak demand periods (few charterers competing to book cargo),
    which correlates with higher idle risk. This is a simplified proxy
    suitable for a prototype - a production version would use actual
    port congestion + booking volume data.
    """
    low_rate_threshold = summary["p50_median"].quantile(demand_threshold_percentile / 100)
    low_demand_days = summary[summary["p50_median"] <= low_rate_threshold]

    idle_risk_pct = len(low_demand_days) / len(summary) * 100

    if idle_risk_pct > 30:
        risk_level = "HIGH"
        suggestion = ("Multiple low-demand days detected in this window. Consider "
                     "flexible positioning or backhaul cargo to reduce idle time.")
    elif idle_risk_pct > 15:
        risk_level = "MODERATE"
        suggestion = "Some low-demand days present. Monitor booking pace closely."
    else:
        risk_level = "LOW"
        suggestion = "Demand looks consistent across the forecast window."

    return {
        "idle_risk_pct": round(idle_risk_pct, 1),
        "risk_level": risk_level,
        "low_demand_days": low_demand_days.index.tolist(),
        "suggestion": suggestion,
    }


def compare_spot_vs_multivoyage(sim_df, n_spot_bookings=4, contract_rate_discount=0.05):
    """
    FR7: Compare two strategies across the SAME simulated future price paths:

      Strategy A (Spot): book cargo repeatedly at spot rates as they arise,
                          i.e. average cost = average of simulated rates
                          across the whole window.

      Strategy B (Multi-voyage contract): lock in one rate now for the
                          whole window, at a small discount to reflect the
                          real-world discount carriers give for guaranteed
                          multi-voyage volume (typically 3-8%).

    Returns which strategy wins in what % of simulations, and average cost
    for each - giving a data-backed answer to the PS's core objective.
    """
    n_simulations = sim_df.shape[0]

    spot_costs = sim_df.mean(axis=1).values

    day0_rate = sim_df.iloc[:, 0].values
    multivoyage_costs = day0_rate * (1 - contract_rate_discount)

    multivoyage_wins = np.sum(multivoyage_costs < spot_costs)
    spot_wins = n_simulations - multivoyage_wins

    result = {
        "spot_avg_cost": round(float(np.mean(spot_costs)), 2),
        "multivoyage_avg_cost": round(float(np.mean(multivoyage_costs)), 2),
        "multivoyage_win_pct": round(multivoyage_wins / n_simulations * 100, 1),
        "spot_win_pct": round(spot_wins / n_simulations * 100, 1),
        "recommendation": (
            "Multi-voyage contract" if multivoyage_wins > spot_wins else "Spot booking"
        ),
    }
    return result


if __name__ == "__main__":
    from forecast_model import load_data, fit_arima_and_forecast
    from monte_carlo import run_monte_carlo, summarize_simulation

    series = load_data()
    forecast_mean, conf_int, residual_std, fitted = fit_arima_and_forecast(series, forecast_days=30)
    sim_df = run_monte_carlo(forecast_mean, residual_std, n_simulations=1000)
    summary, volatility_score = summarize_simulation(sim_df)

    idle = estimate_idle_risk(summary, volatility_score)
    print("Idle Scenario Assessment:")
    print(idle)

    comparison = compare_spot_vs_multivoyage(sim_df)
    print("\nSpot vs Multi-Voyage Comparison:")
    print(comparison)
