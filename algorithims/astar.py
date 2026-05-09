"""
astar.py  —  A* search for emergency vehicle routing in Cairo

A* improves on Dijkstra by using a heuristic function h(n) that estimates
the remaining distance from node n to the goal.  This guides the search
toward the destination, exploring fewer nodes in practice.

Heuristic used: Euclidean distance between the (x, y) grid coordinates
stored on each node.  This is admissible (never over-estimates) because
actual road distances are always >= straight-line distance.

Time complexity:  O(b^d) best case, O(V log V) worst case
Space complexity: O(V)
"""

import heapq
import math


def heuristic(graph, node, goal):
    """
    Straight-line (Euclidean) distance between two nodes.
    Used as the A* h(n) — always admissible for road networks.
    """
    x1, y1 = graph.nodes[node]["x"], graph.nodes[node]["y"]
    x2, y2 = graph.nodes[goal]["x"], graph.nodes[goal]["y"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def astar(graph, start, end, time_of_day="normal"):
    """
    Find the shortest route from start to end using A*.

    The priority queue stores (f_cost, node) where:
        f_cost = g_cost (actual cost so far) + h_cost (estimated remaining)

    Returns (path_list, total_g_cost).
    """
    rush = {"morning_rush": 2.5, "evening_rush": 2.8, "normal": 1.0}
    multiplier = rush.get(time_of_day, 1.0)

    # g_cost[n] = best known actual cost to reach n
    g_cost = {node: float("inf") for node in graph.adj}
    g_cost[start] = 0

    prev    = {node: None for node in graph.adj}
    visited = set()

    # Priority queue: (f = g + h, node)
    pq = [(heuristic(graph, start, end), start)]

    while pq:
        f, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == end:
            break   # Found the optimal path to our destination

        for v, distance, base_traffic in graph.adj[u]:
            if v in visited:
                continue

            travel_cost  = distance * base_traffic * multiplier
            tentative_g  = g_cost[u] + travel_cost

            if tentative_g < g_cost[v]:
                g_cost[v] = tentative_g
                prev[v]   = u
                f_cost    = tentative_g + heuristic(graph, v, end)
                heapq.heappush(pq, (f_cost, v))

    # Reconstruct path by following the predecessor chain
    path, current = [], end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    if not path or path[0] != start:
        return [], float("inf")     # No path found

    return path, g_cost[end]


def emergency_route(graph, incident_location, time_of_day="normal"):
    """
    Dispatch an emergency vehicle from incident_location to the nearest hospital.
    Checks all hospital nodes and returns the one with the lowest A* cost.

    Returns (best_path, best_cost, best_hospital_id).
    """
    hospitals = graph.get_nodes_by_type("hospital")

    best_path     = []
    best_cost     = float("inf")
    best_hospital = None

    for hospital_id in hospitals:
        path, cost = astar(graph, incident_location, hospital_id, time_of_day)
        if cost < best_cost:
            best_cost     = cost
            best_path     = path
            best_hospital = hospital_id

    return best_path, best_cost, best_hospital


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.cairo_data import build_cairo_graph

    g = build_cairo_graph()
    for loc in ["2", "7", "11"]:
        for tod in ["normal", "morning_rush"]:
            path, cost, hosp = emergency_route(g, loc, tod)
            names = [g.nodes[n]["name"] for n in path]
            print(f"[{tod}] Incident @ {g.nodes[loc]['name']} → {g.nodes[hosp]['name']}: {cost:.2f}")
            print(f"  Route: {' → '.join(names)}")
