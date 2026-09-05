"""
Streamlit dashboard (FR8) - clean, minimalist, guided-flow version.

User flow: Origin -> Destination -> Cargo (material + weight) ->
Vessel preference -> Predict. Ties together forecasting, Monte Carlo
simulation, origin/vessel-adjusted costing, idle-risk assessment, and
the spot-vs-multi-voyage comparison.

Run with: python -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from forecast_model import load_data, fit_arima_and_forecast
from monte_carlo import run_monte_carlo, summarize_simulation, risk_flag
from vessel_recommendation import (
    eligible_vessel_types, PORT_CONSTRAINTS, VESSEL_SPECS,
    compute_costs_from_simulation, rank_by_risk_adjusted_cost,
)
from strategy_comparison import estimate_idle_risk, compare_spot_vs_multivoyage
from route_data import ORIGINS, MATERIALS, VESSEL_COST_MULTIPLIER


st.set_page_config(page_title="Freight Forecasting Advisor", layout="centered", page_icon="🚢")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { max-width: 780px; padding-top: 2rem; }
    h1, h2, h3 { font-weight: 600; }
    .result-value { font-size: 1.6rem; font-weight: 700; color: #3B82F6; }
    .result-label { font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase;
                     letter-spacing: 0.04em; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🚢 Freight Forecasting Advisor")
st.caption("SIH 2026 · PS 26006 · Ministry of Steel / SAIL — vessel chartering & bulk cargo procurement")
st.divider()

st.subheader("1. Route")
col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox("Origin", list(ORIGINS.keys()))
with col2:
    destination = st.selectbox("Destination Port (East Coast India)", list(PORT_CONSTRAINTS.keys()))

st.subheader("2. Cargo")
col3, col4 = st.columns(2)
with col3:
    default_materials = ORIGINS[origin]["typical_cargo"]
    material = st.selectbox("Material", MATERIALS,
                             index=MATERIALS.index(default_materials[0]) if default_materials[0] in MATERIALS else 0)
with col4:
    cargo_volume = st.number_input("Weight (tons)", min_value=5000, max_value=200000, value=75000, step=1000)

st.subheader("3. Vessel")
vessel_choice = st.radio("Vessel type", ["Auto-recommend best fit"] + list(VESSEL_SPECS.keys()),
                          horizontal=True)

st.write("")
predict = st.button("Predict", type="primary")

if predict:
    with st.spinner("Analyzing route and simulating market conditions..."):
        series = load_data()
        forecast_mean, conf_int, residual_std, fitted = fit_arima_and_forecast(series, forecast_days=30)
        sim_df = run_monte_carlo(forecast_mean, residual_std, n_simulations=1000)
        summary, base_volatility = summarize_simulation(sim_df)

        origin_factor = ORIGINS[origin]["distance_factor"]

        adjusted_summary = summary.copy()
        for col in ["mean", "p10", "p50_median", "p90"]:
            adjusted_summary[col] = adjusted_summary[col] * origin_factor

        base_mean_cost = summary["mean"].mean()
        eligible = eligible_vessel_types(destination, cargo_volume)

        if not eligible:
            st.error(f"No vessel type physically fits {destination}'s port constraints for "
                     f"{cargo_volume:,} tons. Try a smaller cargo volume or a different port.")
            st.stop()

        costs = compute_costs_from_simulation(eligible, base_mean_cost, base_volatility, origin_factor)

        if vessel_choice == "Auto-recommend best fit":
            ranking = rank_by_risk_adjusted_cost(eligible, costs)
            chosen_vessel = ranking.iloc[0]["vessel_type"]
        else:
            if vessel_choice not in eligible:
                st.warning(f"{vessel_choice} does not fit {destination}'s port constraints or this "
                          f"cargo volume. Showing the best available option instead.")
                ranking = rank_by_risk_adjusted_cost(eligible, costs)
                chosen_vessel = ranking.iloc[0]["vessel_type"]
            else:
                chosen_vessel = vessel_choice
                ranking = rank_by_risk_adjusted_cost(eligible, costs)

        vessel_multiplier = VESSEL_COST_MULTIPLIER.get(chosen_vessel, 1.0)
        total_voyage_factor = origin_factor * vessel_multiplier

        # Scale the simulation summary by the total voyage factor (origin distance + vessel economy of scale)
        # so that point estimate, p10, and p90 move together from the exact same distribution.
        adjusted_summary = summary.copy()
        for col in ["mean", "p10", "p50_median", "p90"]:
            adjusted_summary[col] = adjusted_summary[col] * total_voyage_factor

        best_day = adjusted_summary["p50_median"].idxmin()
        entry_confidence = summary.loc[best_day, "prob_cheapest_day"]
        adjusted_volatility = base_volatility * total_voyage_factor
        flag = risk_flag(adjusted_volatility)

        idle = estimate_idle_risk(adjusted_summary, adjusted_volatility)

        sim_df_adjusted = sim_df * total_voyage_factor
        strategy = compare_spot_vs_multivoyage(sim_df_adjusted)

    st.divider()
    st.subheader("Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown('<div class="result-label">Recommended Vessel</div>'
                    f'<div class="result-value">{chosen_vessel}</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="result-label">Best Entry Day</div>'
                    f'<div class="result-value">{best_day.date()}</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="result-label">Market Risk</div>'
                    f'<div class="result-value">{flag.split(" - ")[0]}</div>', unsafe_allow_html=True)

    st.write("")
    point_est = float(adjusted_summary.loc[best_day, "p50_median"])
    p10_day = float(adjusted_summary.loc[best_day, "p10"])
    p90_day = float(adjusted_summary.loc[best_day, "p90"])
    risk_level = flag.split(" - ")[0].replace(" volatility", "").strip()
    market_cond = flag.split(" - ")[1] if " - " in flag else ""
    risk_desc = f"{risk_level} volatility, {market_cond}" if market_cond else f"{risk_level} volatility"
    callout_text = (
        f"**Estimated freight rate: \\${point_est:.2f}/ton** "
        f"(70% confidence range: \\${p10_day:.2f}–\\${p90_day:.2f}/ton) "
        f"for {material} ({cargo_volume:,} tons) from {origin} to {destination} "
        f"via {chosen_vessel} — {risk_desc}."
    )
    st.info(callout_text)

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Vessel Fit", "Idle Risk", "Contract Strategy"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=adjusted_summary.index, y=adjusted_summary["p90"], mode="lines",
                                  line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=adjusted_summary.index, y=adjusted_summary["p10"], mode="lines",
                                  fill="tonexty", fillcolor="rgba(59,130,246,0.2)",
                                  line=dict(width=0), name="10th-90th percentile"))
        fig.add_trace(go.Scatter(x=adjusted_summary.index, y=adjusted_summary["p50_median"], mode="lines",
                                  line=dict(color="#3B82F6", width=3), name="Median forecast"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                           yaxis_title="Freight Rate (USD/ton)",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Entry confidence: {entry_confidence:.1%} probability {best_day.date()} "
                  "is the cheapest day in this 30-day window.")

    with tab2:
        vessel_names = []
        v_costs = []
        v_colors = []
        v_labels = []
        base_p50_day = float(summary.loc[best_day, "p50_median"])
        for _, row in ranking.iterrows():
            vessel_name = row["vessel_type"]
            v_mult = VESSEL_COST_MULTIPLIER.get(vessel_name, 1.0)
            v_est_cost = round(base_p50_day * origin_factor * v_mult, 2)
            vessel_names.append(vessel_name)
            v_costs.append(v_est_cost)
            if vessel_name == chosen_vessel:
                v_colors.append("#3B82F6")
                v_labels.append(f"${v_est_cost:.2f}/ton (Recommended)")
            else:
                v_colors.append("#94A3B8")
                v_labels.append(f"${v_est_cost:.2f}/ton")

        fig_vessel = go.Figure(go.Bar(
            y=vessel_names,
            x=v_costs,
            orientation="h",
            text=v_labels,
            textposition="auto",
            marker_color=v_colors,
        ))
        fig_vessel.update_layout(
            height=max(180, len(vessel_names) * 60 + 80),
            margin=dict(l=10, r=20, t=20, b=20),
            xaxis_title="Estimated Freight Rate (USD/ton)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_vessel, use_container_width=True)
        st.caption(f"Vessel types filtered to those physically compliant with {destination}'s port constraints and {cargo_volume:,} tons cargo.")

    with tab3:
        dates = adjusted_summary.index
        low_days_set = set(d.date() for d in idle["low_demand_days"])
        all_date_strs = [d.strftime("%b %d") for d in dates]
        colors = ["#EF4444" if d.date() in low_days_set else "#22C55E" for d in dates]
        hover_texts = ["Low Demand (Idle Risk)" if d.date() in low_days_set else "Normal Demand" for d in dates]

        fig_idle = go.Figure()
        # Main single trace ensuring exact chronological ordering
        fig_idle.add_trace(go.Bar(
            x=all_date_strs,
            y=[1] * len(all_date_strs),
            marker_color=colors,
            hovertext=hover_texts,
            hovertemplate="<b>%{x}</b><br>Status: %{hovertext}<extra></extra>",
            showlegend=False,
        ))
        # Legend proxies
        fig_idle.add_trace(go.Bar(
            x=[None], y=[None],
            name="Normal Demand",
            marker_color="#22C55E",
            showlegend=True,
        ))
        fig_idle.add_trace(go.Bar(
            x=[None], y=[None],
            name="Low Demand (Idle Risk)",
            marker_color="#EF4444",
            showlegend=True,
        ))

        fig_idle.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=30, b=30),
            yaxis=dict(visible=False),
            legend=dict(orientation="h", y=1.25, x=0),
            xaxis=dict(tickangle=-45, categoryorder="array", categoryarray=all_date_strs),
        )
        st.plotly_chart(fig_idle, use_container_width=True)
        st.caption(f"30-day demand timeline: {idle['suggestion']}")

    with tab4:
        spot_cost = strategy["spot_avg_cost"]
        multi_cost = strategy["multivoyage_avg_cost"]
        is_multi_better = multi_cost < spot_cost

        strat_names = ["Spot Booking", "Multi-Voyage Contract"]
        strat_costs = [spot_cost, multi_cost]
        strat_colors = ["#94A3B8", "#3B82F6"] if is_multi_better else ["#3B82F6", "#94A3B8"]
        strat_labels = [
            f"${spot_cost:.2f}/ton" + ("<br>★ Recommended" if not is_multi_better else ""),
            f"${multi_cost:.2f}/ton" + ("<br>★ Recommended" if is_multi_better else ""),
        ]

        fig_strat = go.Figure(go.Bar(
            x=strat_names,
            y=strat_costs,
            text=strat_labels,
            textposition="outside",
            marker_color=strat_colors,
        ))
        fig_strat.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=30),
            yaxis_title="Average Freight Cost (USD/ton)",
            yaxis=dict(range=[0, max(strat_costs) * 1.25]),
        )
        st.plotly_chart(fig_strat, use_container_width=True)
        st.caption("Answers the PS objective of shifting from single spot contracts to short/medium-term multi-voyage contracts, backed by simulation.")

    st.divider()
    st.caption("Forecasts use synthetic/proxy data for prototype demonstration (no official "
              "dataset was provided with the problem statement). Origin distance and vessel "
              "economies-of-scale factors are illustrative estimates.")
