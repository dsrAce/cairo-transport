"""
main.py  —  Cairo Smart Transportation System
CSE112: Design and Analysis of Algorithms
Alamein International University

Run modes:
    python main.py          → runs all algorithms and prints results
    python main.py --gui    → launches the interactive PyQt5 GUI
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(__file__))
random.seed(42)

from data.cairo_data import build_cairo_graph
from algorithims.mst      import kruskal_mst
from algorithims.dijkstra  import shortest_path, cached_shortest_path, clear_cache
from algorithims.astar     import emergency_route
from algorithims.dp        import (road_maintenance_knapsack, MAINTENANCE_ROADS,
                                   transit_scheduling, TRANSIT_ROUTES)
from algorithims.greedy    import optimize_all_intersections, analyze_greedy_optimality

LINE = "─" * 62


def header(title):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


# ── 1. Kruskal's MST ─────────────────────────────────────────────────────────
def run_mst(graph):
    header("INFRASTRUCTURE DESIGN  —  Kruskal's MST (Modified)")

    edges, total_cost, critical_connected = kruskal_mst(graph)

    total_critical = len([n for n, m in graph.nodes.items() if m["is_critical"]])
    print(f"  MST edges         : {len(edges)}")
    print(f"  Total cost        : {total_cost:.2f} km")
    print(f"  Critical connected: {len(critical_connected)} / {total_critical}")
    print()

    for u, v, w in edges:
        name_u = graph.nodes[u]["name"]
        name_v = graph.nodes[v]["name"]
        is_crit = graph.nodes[u]["is_critical"] or graph.nodes[v]["is_critical"]
        pop_u, pop_v = graph.nodes[u]["population"], graph.nodes[v]["population"]
        is_high_pop  = pop_u >= 200_000 and pop_v >= 200_000

        tag = "  [CRITICAL]"  if is_crit     else \
              "  [HIGH-POP]"  if is_high_pop else ""
        print(f"  {name_u}  ↔  {name_v}   {w:.2f} km{tag}")


# ── 2. Dijkstra Routing ──────────────────────────────────────────────────────
def run_dijkstra(graph):
    header("TRAFFIC ROUTING  —  Dijkstra's Shortest Path")

    # Test three representative routes across different parts of Cairo
    route_pairs = [
        ("1",  "5",  "Maadi → Heliopolis"),
        ("2",  "8",  "Nasr City → Giza"),
        ("11", "14", "Shubra → Al Rehab"),
    ]
    for src, dst, label in route_pairs:
        print(f"  Route: {label}")
        for tod in ["normal", "morning_rush", "evening_rush"]:
            path, cost = shortest_path(graph, src, dst, tod)
            names = " → ".join(graph.nodes[n]["name"] for n in path)
            print(f"    [{tod:>14}]  cost={cost:6.2f}  |  {names}")
        print()


def run_road_closure(graph):
    header("ROAD CLOSURE  —  Alternate Route Planning")

    blocked = [("2", "3"), ("5", "11")]       # Close two major roads
    blocked_names = [
        f"{graph.nodes[u]['name']} — {graph.nodes[v]['name']}"
        for u, v in blocked
    ]
    print(f"  Simulating closed roads: {blocked_names}\n")

    for src, dst in [("2", "1"), ("11", "4")]:
        normal_path, normal_cost = shortest_path(graph, src, dst, "normal")
        detour_path, detour_cost = shortest_path(graph, src, dst, "normal", blocked)

        print(f"  {graph.nodes[src]['name']} → {graph.nodes[dst]['name']}")
        print(f"    Normal : {normal_cost:.2f} km  |  {' → '.join(graph.nodes[n]['name'] for n in normal_path)}")
        print(f"    Detour : {detour_cost:.2f} km  |  {' → '.join(graph.nodes[n]['name'] for n in detour_path)}")
        print(f"    Extra  : +{detour_cost - normal_cost:.2f} km")
        print()


def run_memoization(graph):
    header("MEMOIZATION  —  Cached Route Planning")

    clear_cache()
    queries = [
        ("1", "5"), ("2", "8"), ("11", "14"),
        ("1", "5"), ("2", "8"), ("11", "14"),   # Repeated — should be cache hits
    ]
    for src, dst in queries:
        (path, cost), from_cache = cached_shortest_path(graph, src, dst, "normal")
        status = "CACHE HIT  ✓" if from_cache else "computed"
        print(f"  {graph.nodes[src]['name']:20s} → {graph.nodes[dst]['name']:20s}  "
              f"cost={cost:6.2f}  [{status}]")


# ── 3. A* Emergency Routing ──────────────────────────────────────────────────
def run_astar(graph):
    header("EMERGENCY ROUTING  —  A* Search")

    incidents = [
        ("2",  "Nasr City"),
        ("7",  "6th October City"),
        ("11", "Shubra"),
    ]
    for node_id, label in incidents:
        print(f"  Incident @ {label}:")
        for tod in ["normal", "morning_rush"]:
            path, cost, hosp = emergency_route(graph, node_id, tod)
            names = " → ".join(graph.nodes[n]["name"] for n in path)
            print(f"    [{tod:>12}]  → {graph.nodes[hosp]['name']}  ({cost:.2f} km)")
            print(f"               {names}")
        print()


# ── 4. Dynamic Programming ───────────────────────────────────────────────────
def run_dp():
    header("ROAD MAINTENANCE  —  0/1 Knapsack DP")

    budget = 35
    selected, benefit, cost, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, budget)
    utilization = round(cost / budget * 100, 1)

    print(f"  Budget      : {budget} M EGP")
    print(f"  Spent       : {cost} M EGP  ({utilization}% utilization)")
    print(f"  Max benefit : {benefit}")
    print()
    for name in selected:
        print(f"  ✓  {name}")
    for r in MAINTENANCE_ROADS:
        if r["name"] not in selected:
            print(f"  ✗  {r['name']}")

    header("TRANSIT SCHEDULING  —  Weighted Interval DP")

    routes, value = transit_scheduling(TRANSIT_ROUTES)
    total_possible = sum(r["value"] for r in TRANSIT_ROUTES)
    coverage = round(value / total_possible * 100, 1)

    print(f"  Passengers served : {value} / {total_possible}  ({coverage}% coverage)")
    print()
    for name in routes:
        print(f"  ✓  {name}")


# ── 5. Greedy Signal Optimisation ────────────────────────────────────────────
def run_greedy():
    header("TRAFFIC SIGNALS  —  Greedy Optimization (Evening Rush)")

    emergency = {"intersection": "Tahrir Square", "direction": "Qasr El-Aini St."}
    results = optimize_all_intersections("evening_rush", emergency=emergency)

    for r in results:
        symbol = "🚨" if "EMERGENCY" in r["reason"] else "●"
        print(f"\n  {symbol}  {r['intersection']}")
        print(f"     GREEN  → {r['green_light']}  ({r['vehicles_served']} vehicles)")
        print(f"     Reason: {r['reason']}")

    print(f"\n  {'─'*40}")
    print("  Optimality Analysis — 500 simulations")
    print(f"  {'─'*40}")
    analysis = analyze_greedy_optimality(500)
    print(f"  Greedy optimal       : {analysis['greedy_optimal_pct']}%")
    print(f"  Starvation overrides : {analysis['starvation_override_pct']}%")
    print(f"  Emergency preemptions: {analysis['emergency_pct']}%")


# ── 6. Visualisations ────────────────────────────────────────────────────────
def run_visualizations():
    header("VISUALIZATIONS  —  Generating Charts")
    try:
        from visualization.visualizer import draw_all
        draw_all()
    except Exception as err:
        print(f"  Could not generate charts: {err}")


# ── 7. GUI ────────────────────────────────────────────────────────────────────
def launch_gui():
    header("GUI  —  Launching Interactive Interface")
    try:
        from PyQt5.QtWidgets import QApplication
        from gui import MainWindow, style_app
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        app.setStyleSheet(style_app())
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except ImportError:
        print("  PyQt5 is not installed.")
        print("  Install it with:  pip install PyQt5")
    except Exception as err:
        print(f"  GUI error: {err}")
        raise


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Cairo Smart Transportation System                     ║")
    print("║   CSE112 — Design and Analysis of Algorithms            ║")
    print("║   Alamein International University                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if "--gui" in sys.argv or "-g" in sys.argv:
        launch_gui()
    else:
        graph = build_cairo_graph()
        print(f"\n  Graph loaded: {len(graph.nodes)} nodes, {len(graph.get_all_edges())} edges")

        run_mst(graph)
        run_dijkstra(graph)
        run_road_closure(graph)
        run_memoization(graph)
        run_astar(graph)
        run_dp()
        run_greedy()
        run_visualizations()

        print(f"\n{LINE}")
        print("  All algorithms completed successfully.")
        print(f"  To launch the interactive GUI, run:  python main.py --gui")
        print(LINE)
