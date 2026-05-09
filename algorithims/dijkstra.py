"""
dijkstra.py  —  Dijkstra's shortest path algorithm for Cairo routing

Three public functions:
  shortest_path()        — plain Dijkstra with optional road closure
  cached_shortest_path() — memoized wrapper (avoids repeating the same query)
  clear_cache()          — invalidate the cache (call when graph changes)

Time complexity:  O((V + E) log V)  — binary-heap priority queue
Space complexity: O(V)              — distance + predecessor arrays
"""

import heapq

# Simple in-memory cache: (start, end, time_of_day) → (path, cost)
_route_cache = {}

# How much rush hour slows traffic down
RUSH_MULTIPLIERS = {
    "morning_rush": 2.5,   # 7–10 am weekdays
    "evening_rush": 2.8,   # 4–8 pm weekdays (worse than morning in Cairo)
    "normal":       1.0,
}


def dijkstra(graph, start, end=None, time_of_day="normal", blocked_edges=None):
    """
    Core Dijkstra implementation.

    Parameters
    ----------
    graph        : Graph object
    start        : starting node ID
    end          : if given, stop early once we pop this node
    time_of_day  : "normal" | "morning_rush" | "evening_rush"
    blocked_edges: list of (u, v) pairs representing closed roads

    Returns
    -------
    dist : dict  — shortest distance from start to every reachable node
    prev : dict  — predecessor map for path reconstruction
    """
    multiplier = RUSH_MULTIPLIERS.get(time_of_day, 1.0)

    # Normalise blocked edges into a frozenset of sorted tuples for O(1) lookup
    blocked = {tuple(sorted(e)) for e in blocked_edges} if blocked_edges else set()

    dist = {node: float("inf") for node in graph.adj}
    prev = {node: None         for node in graph.adj}
    dist[start] = 0

    # Priority queue stores (current_cost, node_id)
    pq = [(0, start)]

    while pq:
        current_cost, u = heapq.heappop(pq)

        # Stale entry — we already found a better path to u
        if current_cost > dist[u]:
            continue

        # Early exit if we only need the route to one destination
        if end and u == end:
            break

        for v, distance, base_traffic in graph.adj[u]:
            if tuple(sorted([u, v])) in blocked:
                continue                    # Road is closed

            travel_cost = distance * base_traffic * multiplier
            new_dist    = current_cost + travel_cost

            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


def reconstruct_path(prev, start, end):
    """Walk the predecessor map backwards to build the full route list."""
    path    = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path if (path and path[0] == start) else []


def shortest_path(graph, start, end, time_of_day="normal", blocked_edges=None):
    """
    Find the shortest route from start to end.

    Returns (path_list, total_cost).
    path_list is empty and cost is inf if no route exists.
    """
    dist, prev = dijkstra(graph, start, end, time_of_day, blocked_edges)
    path       = reconstruct_path(prev, start, end)
    return path, dist[end]


def cached_shortest_path(graph, start, end, time_of_day="normal"):
    """
    Memoised version of shortest_path — identical calls skip re-computation.
    This is the memoization requirement from the project spec.

    Returns ((path, cost), was_cache_hit).
    """
    key = (start, end, time_of_day)
    if key in _route_cache:
        return _route_cache[key], True         # Cache hit

    result = shortest_path(graph, start, end, time_of_day)
    _route_cache[key] = result
    return result, False                       # Cache miss — computed fresh


def clear_cache():
    """Wipe the route cache (needed if edge weights change at runtime)."""
    _route_cache.clear()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.cairo_data import build_cairo_graph

    g = build_cairo_graph()
    for tod in ["normal", "morning_rush", "evening_rush"]:
        path, cost = shortest_path(g, "1", "5", tod)
        names = [g.nodes[n]["name"] for n in path]
        print(f"[{tod:>14}]  cost={cost:.2f}  |  {' → '.join(names)}")
