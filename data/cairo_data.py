"""
cairo_data.py  —  Official CSE112 project dataset
Nodes and edges taken directly from "CSE112-Project Provided Data".

Node IDs map to the official numeric/facility IDs where possible.
Coordinates are the real lat/lon from the data, scaled for drawing.
"""

from graph.graph import Graph


def build_cairo_graph():
    g = Graph()

    # ------------------------------------------------------------------ #
    # DISTRICTS & NEIGHBOURHOODS
    # Source: "Neighborhoods and Districts" section of provided data
    # (id, name, population, type, real_x, real_y)
    # We use real coordinates scaled to a 0–100 grid for visualisation.
    # ------------------------------------------------------------------ #
    g.add_node("1",  "Maadi",                    250_000, False, None,         31.25, 29.96)
    g.add_node("2",  "Nasr City",                500_000, False, None,         31.34, 30.06)
    g.add_node("3",  "Downtown Cairo",           100_000, False, None,         31.24, 30.04)
    g.add_node("4",  "New Cairo",                300_000, False, None,         31.47, 30.03)
    g.add_node("5",  "Heliopolis",               200_000, False, None,         31.32, 30.09)
    g.add_node("6",  "Zamalek",                   50_000, False, None,         31.22, 30.06)
    g.add_node("7",  "6th October City",         400_000, False, None,         30.98, 29.93)
    g.add_node("8",  "Giza",                     550_000, False, None,         31.21, 29.99)
    g.add_node("9",  "Mohandessin",              180_000, False, None,         31.20, 30.05)
    g.add_node("10", "Dokki",                    220_000, False, None,         31.21, 30.03)
    g.add_node("11", "Shubra",                   450_000, False, None,         31.24, 30.11)
    g.add_node("12", "Helwan",                   350_000, False, None,         31.33, 29.85)
    g.add_node("13", "New Admin. Capital",        50_000, False, "government", 31.80, 30.02)
    g.add_node("14", "Al Rehab",                 120_000, False, None,         31.49, 30.06)
    g.add_node("15", "Sheikh Zayed",             150_000, False, None,         30.94, 30.01)

    # ------------------------------------------------------------------ #
    # IMPORTANT FACILITIES
    # ------------------------------------------------------------------ #
    g.add_node("F1", "Cairo Int'l Airport",           0, True, "government",  31.41, 30.11)
    g.add_node("F2", "Ramses Railway Station",        0, True, "government",  31.25, 30.06)
    g.add_node("F3", "Cairo University",              0, True, "government",  31.21, 30.03)
    g.add_node("F4", "Al-Azhar University",           0, True, "government",  31.26, 30.05)
    g.add_node("F5", "Egyptian Museum",               0, True, "government",  31.23, 30.05)
    g.add_node("F6", "Cairo Int'l Stadium",           0, True, "government",  31.30, 30.07)
    g.add_node("F7", "Smart Village",                 0, True, "government",  30.97, 30.07)
    g.add_node("F8", "Cairo Festival City",           0, True, "government",  31.40, 30.03)
    g.add_node("F9", "Qasr El Aini Hospital",         0, True, "hospital",    31.23, 30.03)
    g.add_node("F10","Maadi Military Hospital",       0, True, "hospital",    31.25, 29.95)

    # ------------------------------------------------------------------ #
    # EXISTING ROADS
    # Source: "Road Network Data — Existing Roads" section
    # Edge weight = distance (km); base_traffic derived from condition score
    # condition 10 → multiplier 1.0 (perfect), condition 5 → 1.5 (poor)
    # formula: multiplier = 1 + (10 - condition) * 0.1
    # ------------------------------------------------------------------ #
    def tm(condition):
        """Convert road condition (1–10) to a base traffic multiplier."""
        return round(1.0 + (10 - condition) * 0.1, 2)

    # (from, to, dist_km, condition)
    existing_roads = [
        ("1",  "3",   8.5,  7),
        ("1",  "8",   6.2,  6),
        ("2",  "3",   5.9,  8),
        ("2",  "5",   4.0,  9),
        ("3",  "5",   6.1,  7),
        ("3",  "6",   3.2,  8),
        ("3",  "9",   4.5,  6),
        ("3",  "10",  3.8,  7),
        ("4",  "2",  15.2,  9),
        ("4",  "14",  5.3, 10),
        ("5",  "11",  7.9,  7),
        ("6",  "9",   2.2,  8),
        ("7",  "8",  24.5,  8),
        ("7",  "15",  9.8,  9),
        ("8",  "10",  3.3,  7),
        ("8",  "12", 14.8,  5),
        ("9",  "10",  2.1,  7),
        ("10", "11",  8.7,  6),
        ("11", "F2",  3.6,  7),
        ("12", "1",  12.7,  6),
        ("13", "4",  45.0, 10),
        ("14", "13", 35.5,  9),
        ("15", "7",   9.8,  9),
        ("F1", "5",   7.5,  9),
        ("F1", "2",   9.2,  8),
        ("F2", "3",   2.5,  7),
        ("F7", "15",  8.3,  8),
        ("F8", "4",   6.1,  9),
    ]
    for u, v, dist, cond in existing_roads:
        g.add_edge(u, v, dist, tm(cond))

    # Hospital connections — ensure F9 and F10 connect to their neighbourhoods
    g.add_edge("F9",  "3",  0.5, 1.1)   # Qasr El Aini is in Downtown
    g.add_edge("F9",  "6",  1.2, 1.2)
    g.add_edge("F10", "1",  1.0, 1.1)   # Maadi Military next to Maadi
    g.add_edge("F10", "12", 2.0, 1.2)

    # Connect isolated facility nodes so the graph is fully connected
    # (F3, F4, F5, F6 have no edges in the provided data — we connect them
    #  to their geographically nearest district)
    g.add_edge("F3", "10",  0.2, 1.0)   # Cairo University — right next to Dokki
    g.add_edge("F4", "3",   0.5, 1.1)   # Al-Azhar University — near Downtown
    g.add_edge("F5", "3",   0.4, 1.0)   # Egyptian Museum — near Downtown
    g.add_edge("F6", "5",   1.5, 1.1)   # Cairo Int'l Stadium — near Heliopolis

    return g


# ------------------------------------------------------------------ #
# Quick sanity check
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    g = build_cairo_graph()
    print(f"Nodes : {len(g.nodes)}")
    print(f"Edges : {len(g.get_all_edges())}")
    hospitals = g.get_nodes_by_type("hospital")
    print(f"Hospitals: {[g.nodes[h]['name'] for h in hospitals]}")
# NOTE: patch applied at module level — appending extra edges is cleaner
