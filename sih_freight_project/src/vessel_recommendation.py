"""
Vessel type recommendation engine (FR4).

Step 1: hard-filter vessel types that physically cannot dock at the
        selected port (draft/LOA/beam limits).
Step 2: among eligible vessel types, rank by risk-adjusted cost using
        the Monte Carlo simulation output (mean cost + volatility penalty).
"""

import pandas as pd

VESSEL_SPECS = {
    "Handysize":  {"dwt_range": (10000, 40000),  "max_draft_m": 10.5, "max_loa_m": 190, "max_beam_m": 30},
    "Supramax":   {"dwt_range": (40000, 60000),  "max_draft_m": 12.5, "max_loa_m": 200, "max_beam_m": 32.5},
    "Panamax":    {"dwt_range": (60000, 80000),  "max_draft_m": 14.5, "max_loa_m": 229, "max_beam_m": 32.3},
    "Capesize":   {"dwt_range": (80000, 200000), "max_draft_m": 18.0, "max_loa_m": 300, "max_beam_m": 50},
}

PORT_CONSTRAINTS = {
    "Paradip":        {"max_draft_m": 18.0, "max_loa_m": 300, "max_beam_m": 50},
    "Vizag":          {"max_draft_m": 17.0, "max_loa_m": 280, "max_beam_m": 45},
    "Gangavaram":     {"max_draft_m": 18.5, "max_loa_m": 300, "max_beam_m": 50},
    "Gopalpur":       {"max_draft_m": 14.0, "max_loa_m": 230, "max_beam_m": 32.5},
    "Dhamra":         {"max_draft_m": 18.0, "max_loa_m": 300, "max_beam_m": 50},
    "Sagar-Sandheads":{"max_draft_m": 13.0, "max_loa_m": 225, "max_beam_m": 32.3},
    "Haldia":         {"max_draft_m": 9.0,  "max_loa_m": 186, "max_beam_m": 28},
}


def eligible_vessel_types(port_name, cargo_volume_tons):
    """Step 1: filter vessel types that fit both the port AND the cargo volume."""
    port = PORT_CONSTRAINTS[port_name]
    eligible = []
    for vessel, spec in VESSEL_SPECS.items():
        fits_port = (
            spec["max_draft_m"] <= port["max_draft_m"]
            and spec["max_loa_m"] <= port["max_loa_m"]
            and spec["max_beam_m"] <= port["max_beam_m"]
        )
        fits_cargo = spec["dwt_range"][0] <= cargo_volume_tons <= spec["dwt_range"][1] * 1.1
        if fits_port and fits_cargo:
            eligible.append(vessel)
    return eligible


def compute_costs_from_simulation(eligible_vessels, base_mean_cost, base_volatility,
                                   origin_distance_factor):
    """
    Derives a realistic per-vessel cost estimate from the actual Monte Carlo
    simulation output, adjusted by:
      - origin_distance_factor: how far the selected origin is (from route_data.py)
      - vessel economies of scale: larger vessels cost less per ton (from route_data.py)
    """
    from route_data import VESSEL_COST_MULTIPLIER

    costs = {}
    for vessel in eligible_vessels:
        multiplier = VESSEL_COST_MULTIPLIER.get(vessel, 1.0)
        adjusted_mean = base_mean_cost * origin_distance_factor * multiplier
        adjusted_volatility = base_volatility * origin_distance_factor * multiplier
        costs[vessel] = (round(adjusted_mean, 2), round(adjusted_volatility, 2))
    return costs


def rank_by_risk_adjusted_cost(eligible_vessels, simulated_cost_per_vessel):
    """
    Step 2: rank eligible vessels by risk-adjusted cost.

    risk-adjusted score = mean_cost + (volatility * risk_penalty_weight)
    Lower score = better (cheap AND stable).
    """
    risk_penalty_weight = 0.5
    scored = []
    for vessel in eligible_vessels:
        mean_cost, volatility = simulated_cost_per_vessel[vessel]
        score = mean_cost + (volatility * risk_penalty_weight)
        scored.append({"vessel_type": vessel, "mean_cost": mean_cost,
                        "volatility": volatility, "risk_adjusted_score": score})
    ranked = sorted(scored, key=lambda x: x["risk_adjusted_score"])
    return pd.DataFrame(ranked)


if __name__ == "__main__":
    port = "Paradip"
    cargo_volume = 75000

    eligible = eligible_vessel_types(port, cargo_volume)
    print(f"Eligible vessel types for {port}, {cargo_volume} tons cargo: {eligible}")

    from route_data import ORIGINS
    base_mean, base_vol = 19.7, 1.9
    factor = ORIGINS["Australia"]["distance_factor"]
    costs = compute_costs_from_simulation(eligible, base_mean, base_vol, factor)
    ranking = rank_by_risk_adjusted_cost(eligible, costs)
    print("\nRanked recommendation (best first):")
    print(ranking.round(2))
