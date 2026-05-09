"""
graph.py  —  Weighted directed graph for Cairo's road network
Each node stores location metadata; each edge stores distance + a
base traffic multiplier so we can simulate rush-hour conditions.
"""


class Graph:
    def __init__(self):
        # adj[node_id] = list of (neighbor_id, distance_km, base_traffic_multiplier)
        self.adj   = {}
        # nodes[node_id] = dict of metadata
        self.nodes = {}

    # ------------------------------------------------------------------ #
    def add_node(self, node_id, name, population=0,
                 is_critical=False, critical_type=None, x=0, y=0):
        """Register a location (district, hospital, facility …)."""
        self.adj[node_id]   = []
        self.nodes[node_id] = {
            "name":          name,
            "population":    population,
            "is_critical":   is_critical,
            "critical_type": critical_type,
            "x": x,
            "y": y,
        }

    # ------------------------------------------------------------------ #
    def add_edge(self, u, v, distance_km, base_traffic=1.0):
        """Add an undirected road between u and v."""
        self.adj[u].append((v, distance_km, base_traffic))
        self.adj[v].append((u, distance_km, base_traffic))

    # ------------------------------------------------------------------ #
    def get_edge_weight(self, u, v, time_of_day="normal"):
        """
        Return the effective travel cost for edge (u→v).
        Rush-hour multipliers are applied on top of the base traffic factor.
        """
        rush = {"morning_rush": 2.5, "evening_rush": 2.8, "normal": 1.0}
        multiplier = rush.get(time_of_day, 1.0)

        for neighbor, dist, base_traffic in self.adj[u]:
            if neighbor == v:
                return dist * base_traffic * multiplier
        return float("inf")

    # ------------------------------------------------------------------ #
    def get_all_edges(self):
        """Return a de-duplicated list of (u, v, effective_weight) tuples."""
        seen  = set()
        edges = []
        for u in self.adj:
            for v, dist, base_traffic in self.adj[u]:
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v, dist * base_traffic))
        return edges

    # ------------------------------------------------------------------ #
    def get_nodes_by_type(self, critical_type):
        """Return all node IDs whose critical_type matches the given value."""
        return [
            nid for nid, meta in self.nodes.items()
            if meta["critical_type"] == critical_type
        ]
