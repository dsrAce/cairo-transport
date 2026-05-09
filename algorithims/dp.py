"""
dp.py  —  Dynamic Programming solutions for Cairo transport

Two problems solved:

1. Road Maintenance Knapsack (0/1 Knapsack)
   Given a budget and a list of maintenance projects, choose the subset
   that maximises total benefit without exceeding the budget.
   Time:  O(n × W)   Space: O(n × W)

2. Transit Scheduling (Weighted Interval Scheduling)
   Given a set of bus/metro routes each with a start time, end time, and
   passenger value, select the non-overlapping set with maximum total value.
   Time:  O(n log n)  Space: O(n)

Road data is derived from the official project data (road segments with
condition scores) and public transport routes from the provided data.
"""

# ── Road Maintenance Data (from "Road Network Data" in provided data) ─────────
# Each road segment that needs maintenance is listed with an estimated
# repair cost (million EGP) and a benefit score (congestion reduction value).
MAINTENANCE_ROADS = [
    {"name": "Ring Road South (Maadi–Helwan)",     "cost": 15, "benefit": 90},
    {"name": "Salah Salem Highway (Downtown)",      "cost": 10, "benefit": 70},
    {"name": "Autostrad Road (Nasr City)",          "cost": 12, "benefit": 75},
    {"name": "Nasr Road (Nasr City Central)",       "cost":  8, "benefit": 55},
    {"name": "Maadi Corniche",                      "cost":  6, "benefit": 45},
    {"name": "Heliopolis Main Street",              "cost":  9, "benefit": 60},
    {"name": "Shubra–Benha Road",                   "cost": 14, "benefit": 85},
    {"name": "October Bridge Overhaul",             "cost": 20, "benefit": 95},
    {"name": "Giza–6th October Connector",          "cost": 11, "benefit": 65},
    {"name": "Helwan Industrial Road",              "cost":  7, "benefit": 40},
]

# ── Transit Route Data (from "Public Transportation Data" in provided data) ───
# Routes are modelled with a start hour, end hour, and daily passenger value.
# These map to the official bus/metro lines in the provided dataset.
TRANSIT_ROUTES = [
    {"name": "M1 — Helwan → New Marg (early)",      "start":  6, "end":  9, "value": 500},
    {"name": "M2 — Shubra → Giza (morning)",        "start":  7, "end": 10, "value": 450},
    {"name": "M3 — Airport → Imbaba (morning)",     "start":  8, "end": 12, "value": 600},
    {"name": "B1 — Maadi → Downtown",               "start":  9, "end": 11, "value": 350},
    {"name": "B2 — 6th Oct → Dokki",                "start": 10, "end": 14, "value": 400},
    {"name": "B4 — New Cairo → Downtown",           "start": 12, "end": 15, "value": 300},
    {"name": "B7 — New Admin Capital → Al Rehab",   "start": 13, "end": 16, "value": 420},
    {"name": "M1 — Helwan → New Marg (evening)",    "start": 15, "end": 18, "value": 380},
    {"name": "M3 — Airport → Imbaba (evening)",     "start": 16, "end": 19, "value": 410},
    {"name": "B8 — Smart Village → 6th Oct",        "start": 17, "end": 20, "value": 360},
    {"name": "B5 — Helwan → Maadi",                 "start": 18, "end": 21, "value": 330},
    {"name": "B6 — Shubra → Nasr City",             "start": 20, "end": 23, "value": 290},
]


# ── 0/1 Knapsack ─────────────────────────────────────────────────────────────
def road_maintenance_knapsack(roads, budget):
    """
    Classic 0/1 Knapsack DP.

    dp[i][w] = best benefit achievable using the first i roads with budget w.

    After filling the table we trace back to find which roads were selected.

    Returns (selected_names, total_benefit, total_cost, dp_table).
    """
    n  = len(roads)
    # dp table: (n+1) rows × (budget+1) columns, all starting at 0
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost    = roads[i - 1]["cost"]
        benefit = roads[i - 1]["benefit"]
        for w in range(budget + 1):
            # Option A: skip this road
            dp[i][w] = dp[i - 1][w]
            # Option B: include this road (only if we can afford it)
            if w >= cost:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - cost] + benefit)

    # Traceback — walk the table backwards to find the chosen roads
    selected = []
    remaining_budget = budget
    for i in range(n, 0, -1):
        if dp[i][remaining_budget] != dp[i - 1][remaining_budget]:
            selected.append(roads[i - 1]["name"])
            remaining_budget -= roads[i - 1]["cost"]

    total_benefit = dp[n][budget]
    total_cost    = sum(r["cost"] for r in roads if r["name"] in selected)

    return selected, total_benefit, total_cost, dp


# ── Weighted Interval Scheduling ─────────────────────────────────────────────
def transit_scheduling(routes):
    """
    Weighted Interval Scheduling via DP.

    Sort routes by finish time.  For each route i, let p(i) = the index of
    the last route that finishes before route i starts.
    dp[i] = max value using only the first i routes.

    Returns (selected_route_names, total_passenger_value).
    """
    # Sort by end time so we can apply the standard algorithm
    routes = sorted(routes, key=lambda r: r["end"])
    n      = len(routes)

    def latest_compatible(i):
        """
        Binary-search style: find the latest route j that ends at or before
        route i starts.  We do a simple linear scan here since n is small.
        """
        for j in range(i - 1, -1, -1):
            if routes[j]["end"] <= routes[i]["start"]:
                return j
        return -1   # No compatible route exists

    # p[i] = index of latest route compatible with route i
    p  = [latest_compatible(i) for i in range(n)]

    # Build the DP table
    dp    = [0] * (n + 1)
    for i in range(1, n + 1):
        value_if_included = routes[i - 1]["value"] + (dp[p[i - 1] + 1] if p[i - 1] >= 0 else 0)
        value_if_skipped  = dp[i - 1]
        dp[i] = max(value_if_included, value_if_skipped)

    # Traceback
    selected = []
    i = n
    while i > 0:
        value_if_included = routes[i - 1]["value"] + (dp[p[i - 1] + 1] if p[i - 1] >= 0 else 0)
        if value_if_included >= dp[i - 1]:
            selected.append(routes[i - 1]["name"])
            i = p[i - 1] + 1 if p[i - 1] >= 0 else 0
        else:
            i -= 1

    return list(reversed(selected)), dp[n]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Knapsack
    selected, benefit, cost, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 35)
    print(f"Knapsack | budget=35 | spent={cost} | benefit={benefit}")
    for name in selected:
        print(f"  ✓ {name}")

    print()

    # Transit scheduling
    routes, value = transit_scheduling(TRANSIT_ROUTES)
    total = sum(r["value"] for r in TRANSIT_ROUTES)
    print(f"Transit scheduling | passengers={value}/{total} ({100*value//total}% coverage)")
    for name in routes:
        print(f"  ✓ {name}")
