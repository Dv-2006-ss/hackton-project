"""
Route and cargo reference data - grounds the model in the actual PS
background (5 named origins, coal/bulk cargo, vessel economies of scale).

distance_factor: relative freight cost multiplier vs a baseline route.
Rough real-world basis: Indonesia and Australia are the shortest sailing
distances to India's East Coast; Mozambique is moderate (around the Cape
or via Suez); Russia and the US are the longest / most expensive routes
for this trade. These are illustrative multipliers for the prototype, not
official freight indices.
"""

ORIGINS = {
    "Indonesia":  {"distance_factor": 0.80, "typical_cargo": ["Coal"]},
    "Australia":  {"distance_factor": 1.00, "typical_cargo": ["Coal", "Iron Ore"]},
    "Mozambique": {"distance_factor": 1.15, "typical_cargo": ["Coal"]},
    "Russia":     {"distance_factor": 1.35, "typical_cargo": ["Coal"]},
    "US":         {"distance_factor": 1.55, "typical_cargo": ["Coal"]},
}

MATERIALS = ["Coal", "Iron Ore", "Limestone / Other Bulk"]

VESSEL_COST_MULTIPLIER = {
    "Handysize": 1.15,
    "Supramax": 1.05,
    "Panamax": 1.00,
    "Capesize": 0.85,
}
