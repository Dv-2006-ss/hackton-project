"""
Streamlit dashboard (FR8) - clean, minimalist, guided-flow version.

User flow: Origin -> Destination -> Cargo (material + weight) ->
Vessel preference -> Predict. Ties together forecasting, Monte Carlo
simulation, origin/vessel-adjusted costing, idle-risk assessment, and
the spot-vs-multi-voyage comparison.

Run with: python -m streamlit run dashboard.py
"""

import base64
import os
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


def generate_ship_svg(vessel_class: str, is_recommended: bool = False) -> str:
    """
    Generates a clean, stylized side-view cargo ship silhouette SVG
    scaled to reflect the relative size difference across bulk vessel classes.
    """
    fill_color = "#2563EB" if is_recommended else "#64748B"
    bridge_color = "#1D4ED8" if is_recommended else "#475569"
    deck_color = "#93C5FD" if is_recommended else "#94A3B8"

    scales = {
        "Handysize": {"hull_len": 55, "bridge_w": 10, "holds": 3, "h": 16, "bow_lead": 8},
        "Supramax":  {"hull_len": 72, "bridge_w": 12, "holds": 4, "h": 19, "bow_lead": 10},
        "Panamax":   {"hull_len": 90, "bridge_w": 14, "holds": 5, "h": 22, "bow_lead": 12},
        "Capesize":  {"hull_len": 118, "bridge_w": 16, "holds": 6, "h": 26, "bow_lead": 15},
    }
    s = scales.get(vessel_class, scales["Panamax"])
    hl = s["hull_len"]
    h = s["h"]
    bw = s["bridge_w"]
    bow_end = hl + s["bow_lead"] + 4

    hold_rects = ""
    start_x = 22 + bw
    for i in range(s["holds"]):
        hx = start_x + i * 11
        if hx + 8 < hl:
            hold_rects += f'<rect x="{hx}" y="{28 - h + 7}" width="7" height="2" fill="{deck_color}" rx="0.5"/>'

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 145 32" width="145" height="32">
        <line x1="2" y1="28" x2="{bow_end + 4}" y2="28" stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.6"/>
        <path d="M 6 28 L {hl} 28 Q {hl + s['bow_lead']} 25 {bow_end} {28 - h} L {hl} {28 - h} L 6 {28 - h} Z" fill="{fill_color}" />
        <rect x="12" y="{28 - h - 7}" width="{bw}" height="7" fill="{bridge_color}" rx="1"/>
        <rect x="15" y="{28 - h - 11}" width="7" height="4" fill="{bridge_color}" rx="1"/>
        <rect x="13" y="{28 - h - 10}" width="2.5" height="3" fill="#EF4444"/>
        {hold_rects}
    </svg>'''
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def get_vessel_photo_base64(vessel_name: str) -> str:
    """
    Returns base64 data URI of the local vessel photo, or empty string if not found.
    Fails gracefully if file is inaccessible.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "assets", "vessels", f"{vessel_name.lower()}.jpg")
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64_img}"
    except Exception:
        pass
    return ""


st.set_page_config(page_title="Freight Forecasting Advisor", layout="centered", page_icon="🚢")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Base Canvas & Palette */
    .stApp {
        background-color: #0B0F14 !important;
        color: #F5F7FA !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Typography Hierarchy */
    h1 {
        font-size: 2.3rem !important;
        font-weight: 750 !important;
        color: #F5F7FA !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 0.25rem !important;
    }
    h2, h3 {
        font-size: 1.25rem !important;
        font-weight: 650 !important;
        color: #F5F7FA !important;
        letter-spacing: -0.015em !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h4 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #F5F7FA !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #98A2B3 !important;
        font-size: 0.88rem !important;
        line-height: 1.45 !important;
    }
    p, span, label {
        color: #F5F7FA;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.25rem 0 !important;
    }

    /* Input Widgets: Selectbox & Number Input */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #151B24 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #F5F7FA !important;
        transition: border-color 0.15s ease-in-out;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
    }
    div[data-baseweb="select"] * {
        color: #F5F7FA !important;
    }
    input {
        color: #F5F7FA !important;
    }
    div[data-testid="stWidgetLabel"] p {
        color: #98A2B3 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }

    /* Radio Group (Vessel selection) */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #10151D;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 12px;
        gap: 16px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        color: #F5F7FA !important;
        font-size: 0.9rem !important;
    }

    /* Predict / Generate Forecast Button */
    div[data-testid="stButton"] > button,
    button[kind="primary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="baseButton-primary"],
    .stButton > button {
        width: 100% !important;
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px !important;
        font-weight: 650 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stButton"] > button:hover,
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    .stButton > button:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stButton"] > button:active,
    button[kind="primary"]:active,
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Result Metric Cards */
    .result-card {
        background-color: #10151D;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .result-label {
        font-size: 0.75rem !important;
        color: #98A2B3 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
        margin-bottom: 4px;
    }
    .result-value {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #3B82F6 !important;
        letter-spacing: -0.02em !important;
    }

    /* Callout Alert / Info Box */
    [data-testid="stAlert"] {
        background-color: #151B24 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-left: 4px solid #3B82F6 !important;
        border-radius: 8px !important;
        color: #F5F7FA !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stAlert"] p {
        color: #F5F7FA !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #98A2B3 !important;
        font-weight: 550 !important;
        background-color: transparent !important;
        padding: 0.6rem 1.1rem !important;
        font-size: 0.92rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3B82F6 !important;
        font-weight: 650 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #3B82F6 !important;
    }
    /* Container and Column Card Subtleties */
    div[data-testid="stHorizontalBlock"] {
        background-color: #10151D;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 0.5rem;
    }
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
predict = st.button("Generate Forecast", type="primary")

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
            st.error(f"No vessels fit this route/cargo combination: {destination}'s port constraints cannot accommodate {cargo_volume:,} tons. Try a smaller cargo volume or a different port.")
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
        st.markdown('<div class="result-card">'
                    '<div class="result-label">Recommended Vessel</div>'
                    f'<div class="result-value">{chosen_vessel}</div>'
                    '</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="result-card">'
                    '<div class="result-label">Best Entry Day</div>'
                    f'<div class="result-value">{best_day.date()}</div>'
                    '</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="result-card">'
                    '<div class="result-label">Market Risk</div>'
                    f'<div class="result-value">{flag.split(" - ")[0]}</div>'
                    '</div>', unsafe_allow_html=True)

    st.write("")
    point_est = float(adjusted_summary.loc[best_day, "p50_median"])
    p10_day = float(adjusted_summary.loc[best_day, "p10"])
    p90_day = float(adjusted_summary.loc[best_day, "p90"])
    risk_level = flag.split(" - ")[0].replace(" volatility", "").strip()
    market_cond = flag.split(" - ")[1] if " - " in flag else ""
    risk_desc = f"{risk_level} volatility, {market_cond}" if market_cond else f"{risk_level} volatility"
    callout_text = (
        f"**Estimated freight rate: \\${point_est:,.2f}/ton** "
        f"(70% confidence range: \\${p10_day:,.2f}–\\${p90_day:,.2f}/ton) "
        f"for {material} ({cargo_volume:,} tons) from {origin} to {destination} "
        f"via {chosen_vessel} — {risk_desc}."
    )
    st.info(callout_text)

    tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Vessel Fit", "Idle Risk", "Contract Strategy"])

    with tab1:
        st.markdown(f"#### 30-Day Freight Rate Forecast: {origin} → {destination} via {chosen_vessel}")
        st.caption(f"Shows expected price range over the next 30 days for {cargo_volume:,} tons of {material}. Hover over any point for the exact value and confidence range.")

        hover_texts = [
            f"<b>{d.strftime('%b %d, %Y')}</b><br>"
            f"Median: ${med:,.2f}/ton<br>"
            f"Range: ${p10:,.2f}–${p90:,.2f}/ton"
            for d, med, p10, p90 in zip(
                adjusted_summary.index,
                adjusted_summary["p50_median"],
                adjusted_summary["p10"],
                adjusted_summary["p90"],
            )
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=adjusted_summary.index,
            y=adjusted_summary["p90"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=adjusted_summary.index,
            y=adjusted_summary["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            name="10th–90th Percentile Range"
        ))
        fig.add_trace(go.Scatter(
            x=adjusted_summary.index,
            y=adjusted_summary["p50_median"],
            mode="lines",
            line=dict(color="#2563EB", width=3),
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Median Forecast"
        ))
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Forecast Date",
            yaxis_title="Freight Rate (USD/ton)",
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig, use_container_width=True)

        flag_cond = flag.split(" - ")[0]
        insight_fc = (
            f"💡 **Key Insight:** Prices are projected to be lowest on {best_day.strftime('%b %d')} "
            f"at ${point_est:,.2f}/ton, with {entry_confidence:.1%} simulation probability of being the cheapest day "
            f"in this 30-day window under {flag_cond.lower()}."
        )
        st.info(insight_fc)

    with tab2:
        st.markdown(f"#### Vessel Fit & Cost Comparison: {destination} ({cargo_volume:,} tons {material})")
        st.caption(f"Real vessel class photos, physical port-compliance specifications, and risk-adjusted freight rates.")

        if ranking.empty:
            st.info("No vessels fit this route/cargo combination.")
        else:
            base_p50_day = float(summary.loc[best_day, "p50_median"])

            # 1. Real Vessel Photo Cards with Limits and Highlight Styling
            current_dir = os.path.dirname(os.path.abspath(__file__))
            n_cards = len(ranking)
            card_cols = st.columns(n_cards) if n_cards <= 3 else st.columns(2)

            for idx, (_, row) in enumerate(ranking.iterrows()):
                vessel_name = row["vessel_type"]
                v_mult = VESSEL_COST_MULTIPLIER.get(vessel_name, 1.0)
                v_est_cost = round(base_p50_day * origin_factor * v_mult, 2)
                spec = VESSEL_SPECS.get(vessel_name, {})
                draft_val = spec.get("max_draft_m", "-")
                loa_val = spec.get("max_loa_m", "-")
                dwt_max = spec.get("dwt_range", (0, 0))[1]
                is_rec = (vessel_name == chosen_vessel)

                col = card_cols[idx % len(card_cols)]
                with col:
                    v_key = vessel_name.lower()
                    img_path = os.path.join(current_dir, "assets", "vessels", f"{v_key}.jpg")

                    border_color = "#3B82F6" if is_rec else "rgba(255, 255, 255, 0.08)"
                    bg_color = "#151B24" if is_rec else "#10151D"
                    badge_html = (
                        '<span style="background-color: #3B82F6; color: white; padding: 2px 8px; '
                        'border-radius: 12px; font-size: 0.72rem; font-weight: 600;">★ Recommended</span>'
                        if is_rec else
                        '<span style="background-color: #1F2937; color: #98A2B3; border: 1px solid rgba(255,255,255,0.08); padding: 2px 8px; '
                        'border-radius: 12px; font-size: 0.72rem; font-weight: 500;">Compliant</span>'
                    )

                    st.markdown(
                        f"""
                        <div style="border: 2px solid {border_color}; background-color: {bg_color};
                                    border-radius: 10px; padding: 10px 12px 6px 12px; margin-bottom: 6px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 1.05rem; font-weight: 700; color: #F5F7FA;">{vessel_name}</span>
                                {badge_html}
                            </div>
                            <div style="font-size: 1.3rem; font-weight: 700; color: {'#3B82F6' if is_rec else '#F5F7FA'}; margin-top: 2px;">
                                ${v_est_cost:,.2f}<span style="font-size: 0.8rem; font-weight: 500; color: #98A2B3;"> / ton</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Graceful image loading - fails safely if images are missing/inaccessible
                    if os.path.exists(img_path):
                        try:
                            st.image(img_path, caption=f"{vessel_name} Class", width=220)
                        except Exception:
                            st.caption(f"🚢 {vessel_name} Class Bulk Carrier")
                    else:
                        st.caption(f"🚢 {vessel_name} Class Bulk Carrier")

                    st.markdown(
                        f"""
                        <div style="font-size: 0.82rem; background: #151B24; color: #F5F7FA; padding: 8px 10px;
                                    border-radius: 6px; margin-top: -6px; margin-bottom: 12px; border: 1px solid rgba(255, 255, 255, 0.08); line-height: 1.4;">
                            <span style="color: #98A2B3;">Max Draft:</span> <b style="color: #F5F7FA;">{draft_val} m</b><br>
                            <span style="color: #98A2B3;">Max LOA:</span> <b style="color: #F5F7FA;">{loa_val} m</b><br>
                            <span style="color: #98A2B3;">DWT Capacity:</span> <b style="color: #F5F7FA;">up to {dwt_max:,} t</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # 2. Comparative Rate Bar Chart
            vessel_names = []
            v_costs = []
            v_colors = []
            v_labels = []
            v_hovers = []
            y_categories = []
            for _, row in ranking.iterrows():
                vessel_name = row["vessel_type"]
                v_mult = VESSEL_COST_MULTIPLIER.get(vessel_name, 1.0)
                v_est_cost = round(base_p50_day * origin_factor * v_mult, 2)
                vessel_names.append(vessel_name)
                v_costs.append(v_est_cost)

                spec = VESSEL_SPECS.get(vessel_name, {})
                draft_val = spec.get("max_draft_m", "-")
                loa_val = spec.get("max_loa_m", "-")
                dwt_max = spec.get("dwt_range", (0, 0))[1]

                y_categories.append(f"<b>{vessel_name}</b>")

                if vessel_name == chosen_vessel:
                    v_colors.append("#2563EB")
                    v_labels.append(f"${v_est_cost:,.2f}/ton (★ Recommended)")
                    v_hovers.append(
                        f"<b>{vessel_name} (Recommended)</b><br>"
                        f"Estimated Rate: ${v_est_cost:,.2f}/ton<br>"
                        f"Max Draft: {draft_val}m | Max LOA: {loa_val}m | Capacity: up to {dwt_max:,}t DWT"
                    )
                else:
                    v_colors.append("#94A3B8")
                    v_labels.append(f"${v_est_cost:,.2f}/ton (Compliant)")
                    v_hovers.append(
                        f"<b>{vessel_name}</b><br>"
                        f"Estimated Rate: ${v_est_cost:,.2f}/ton<br>"
                        f"Max Draft: {draft_val}m | Max LOA: {loa_val}m | Capacity: up to {dwt_max:,}t DWT"
                    )

            fig_vessel = go.Figure(go.Bar(
                y=y_categories,
                x=v_costs,
                orientation="h",
                text=v_labels,
                textposition="outside",
                cliponaxis=False,
                marker_color=v_colors,
                hovertext=v_hovers,
                hovertemplate="%{hovertext}<extra></extra>",
            ))

            for i, v_name in enumerate(vessel_names):
                photo_uri = get_vessel_photo_base64(v_name)
                img_src = photo_uri if photo_uri else generate_ship_svg(v_name, is_recommended=(v_name == chosen_vessel))
                fig_vessel.add_layout_image(
                    dict(
                        source=img_src,
                        xref="paper",
                        yref="y",
                        x=-0.04,
                        y=y_categories[i],
                        sizex=0.14,
                        sizey=0.85,
                        xanchor="right",
                        yanchor="middle",
                        layer="above"
                    )
                )

            max_c = max(v_costs) if v_costs else 30
            fig_vessel.update_layout(
                height=max(180, len(vessel_names) * 65 + 70),
                margin=dict(l=190, r=40, t=10, b=30),
                xaxis=dict(
                    title="Estimated Freight Rate (USD/ton)",
                    range=[0, max_c * 1.5],
                    showgrid=True,
                    gridcolor="#F1F5F9"
                ),
                yaxis=dict(
                    autorange="reversed",
                    showgrid=False
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_vessel, use_container_width=True)

            port_lim = PORT_CONSTRAINTS.get(destination, {})
            port_draft = port_lim.get("max_draft_m", "")
            port_loa = port_lim.get("max_loa_m", "")
            port_limits_str = f"{destination}'s {port_draft}m draft and {port_loa}m LOA limits"

            if len(v_costs) > 1:
                cheapest_name = ranking.iloc[0]["vessel_type"]
                next_name = ranking.iloc[1]["vessel_type"]
                diff = v_costs[1] - v_costs[0]
                insight_vf = (
                    f"💡 **Key Insight:** {cheapest_name} is the cheapest eligible vessel, "
                    f"costing ${diff:,.2f}/ton less than {next_name} while comfortably fitting within {port_limits_str}."
                )
            else:
                insight_vf = (
                    f"💡 **Key Insight:** {chosen_vessel} is the sole vessel class meeting both {port_limits_str} "
                    f"and the {cargo_volume:,}-ton cargo requirements."
                )
            st.info(insight_vf)

    with tab3:
        st.markdown(f"#### 30-Day Chartering Demand & Idle Risk: {chosen_vessel} on {origin} → {destination}")
        st.caption("Daily demand calendar identifying low-demand risk windows where spot vessel chartering may experience delays.")

        dates = adjusted_summary.index
        low_days_set = set(d.date() for d in idle["low_demand_days"])
        all_date_strs = [d.strftime("%b %d") for d in dates]
        colors = ["#EF4444" if d.date() in low_days_set else "#10B981" for d in dates]
        hover_texts = ["Low Demand (Idle Risk)" if d.date() in low_days_set else "Normal Demand" for d in dates]

        fig_idle = go.Figure()
        # Main single trace ensuring exact chronological ordering
        fig_idle.add_trace(go.Bar(
            x=all_date_strs,
            y=[1] * len(all_date_strs),
            marker_color=colors,
            hovertext=hover_texts,
            hovertemplate="<b>%{x}</b><br>Demand Status: %{hovertext}<extra></extra>",
            showlegend=False,
        ))
        # Legend proxies
        fig_idle.add_trace(go.Bar(
            x=[None], y=[None],
            name="Normal Demand",
            marker_color="#10B981",
            showlegend=True,
        ))
        fig_idle.add_trace(go.Bar(
            x=[None], y=[None],
            name="Low Demand (Idle Risk)",
            marker_color="#EF4444",
            showlegend=True,
        ))

        fig_idle.update_layout(
            height=190,
            margin=dict(l=10, r=10, t=30, b=30),
            yaxis=dict(visible=False),
            xaxis_title="Forecast Date",
            legend=dict(orientation="h", y=1.25, x=0),
            xaxis=dict(tickangle=-45, categoryorder="array", categoryarray=all_date_strs),
        )
        st.plotly_chart(fig_idle, use_container_width=True)

        num_low = len(idle["low_demand_days"])
        if num_low > 0:
            insight_idle = (
                f"💡 **Key Insight:** {num_low} of the next 30 days show low-demand risk "
                f"({idle['idle_risk_pct']:.1f}% idle risk) — {idle['suggestion']}"
            )
        else:
            insight_idle = (
                "💡 **Key Insight:** Demand remains consistent across all 30 forecast days "
                "with 0 high-risk idle days detected."
            )
        st.info(insight_idle)

    with tab4:
        st.markdown(f"#### Spot Booking vs. Multi-Voyage Contract: Which Costs Less?")
        st.caption("Spot offers flexibility with no commitment; Multi-Voyage locks in a rate for price certainty — here's which is cheaper for this route.")

        spot_cost = strategy["spot_avg_cost"]
        multi_cost = strategy["multivoyage_avg_cost"]
        is_multi_better = multi_cost < spot_cost

        strat_names = ["Spot Booking", "Multi-Voyage Contract"]
        strat_costs = [spot_cost, multi_cost]
        strat_colors = ["#94A3B8", "#3B82F6"] if is_multi_better else ["#3B82F6", "#94A3B8"]
        strat_labels = [
            f"${spot_cost:,.2f}/ton" + ("<br>★ Recommended" if not is_multi_better else ""),
            f"${multi_cost:,.2f}/ton" + ("<br>★ Recommended" if is_multi_better else ""),
        ]
        strat_hovers = [
            f"<b>Spot Booking</b><br>Average Cost: ${spot_cost:,.2f}/ton<br>Win Rate: {strategy['spot_win_pct']:.1f}%",
            f"<b>Multi-Voyage Contract</b><br>Average Cost: ${multi_cost:,.2f}/ton<br>Win Rate: {strategy['multivoyage_win_pct']:.1f}%",
        ]

        fig_strat = go.Figure(go.Bar(
            x=strat_names,
            y=strat_costs,
            text=strat_labels,
            textposition="outside",
            marker_color=strat_colors,
            hovertext=strat_hovers,
            hovertemplate="%{hovertext}<extra></extra>",
        ))
        fig_strat.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=30),
            xaxis_title="Contracting Mechanism",
            yaxis_title="Average Freight Cost (USD/ton)",
            yaxis=dict(range=[0, max(strat_costs) * 1.25]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        if is_multi_better:
            savings = spot_cost - multi_cost
            insight_strat = (
                f"💡 **Key Insight:** Multi-voyage contracts are cheaper in this scenario, "
                f"saving ${savings:,.2f}/ton on average and winning in {strategy['multivoyage_win_pct']:.1f}% of simulations."
            )
            multi_note = "Price locked in — less flexible, but cheaper on average here."
            spot_note = "Flexible, no commitment — but fully exposed to price swings and higher expected costs."
        else:
            savings = multi_cost - spot_cost
            insight_strat = (
                f"💡 **Key Insight:** Spot booking is cheaper in this scenario, "
                f"saving ${savings:,.2f}/ton on average and winning in {strategy['spot_win_pct']:.1f}% of simulations."
            )
            spot_note = "Flexible, no commitment — cheaper on average here with favorable spot conditions."
            multi_note = "Price locked in — provides price certainty, but carries a slight term premium."
        st.info(insight_strat)

        col_spot, col_multi = st.columns(2)
        with col_spot:
            st.markdown(
                f"""
                <div style="background-color: #10151D; border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 8px; padding: 10px 14px; height: 100%;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: {'#3B82F6' if not is_multi_better else '#F5F7FA'}; margin-bottom: 3px;">
                        Spot Booking {'★' if not is_multi_better else ''}
                    </div>
                    <div style="font-size: 0.83rem; color: #98A2B3; line-height: 1.4;">
                        Flexible, no commitment — but fully exposed to price swings.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_multi:
            st.markdown(
                f"""
                <div style="background-color: #10151D; border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 8px; padding: 10px 14px; height: 100%;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: {'#3B82F6' if is_multi_better else '#F5F7FA'}; margin-bottom: 3px;">
                        Multi-Voyage Contract {'★' if is_multi_better else ''}
                    </div>
                    <div style="font-size: 0.83rem; color: #98A2B3; line-height: 1.4;">
                        Price locked in — less flexible, but cheaper on average here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()
    st.caption("Forecasts use synthetic/proxy data for prototype demonstration (no official "
              "dataset was provided with the problem statement). Origin distance and vessel "
              "economies-of-scale factors are illustrative estimates.")
