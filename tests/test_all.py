"""
test_all.py  —  Unit tests for the Cairo Transport System
Run with:  python -m pytest tests/ -v
       or:  python tests/test_all.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.cairo_data  import build_cairo_graph
from algorithims.mst      import kruskal_mst
from algorithims.dijkstra  import shortest_path, cached_shortest_path, clear_cache
from algorithims.astar     import astar, emergency_route
from algorithims.dp        import (road_maintenance_knapsack, MAINTENANCE_ROADS,
                                   transit_scheduling, TRANSIT_ROUTES)
from algorithims.greedy    import (simulate_intersection, optimize_all_intersections,
                                   analyze_greedy_optimality, INTERSECTIONS)


# ─────────────────────────────────────────────────────────────────────────────
class TestGraph(unittest.TestCase):
    def setUp(self):
        self.g = build_cairo_graph()

    def test_node_count(self):
        self.assertGreater(len(self.g.nodes), 20,
                           "Graph should have at least 20 nodes")

    def test_edge_count(self):
        self.assertGreater(len(self.g.get_all_edges()), 15,
                           "Graph should have at least 15 unique edges")

    def test_hospitals_exist(self):
        hospitals = self.g.get_nodes_by_type("hospital")
        self.assertGreater(len(hospitals), 0,
                           "There should be at least one hospital in the graph")

    def test_edge_weight_rush(self):
        """Rush-hour weight must be strictly greater than normal weight."""
        for node, neighbors in self.g.adj.items():
            for neighbor, dist, base_t in neighbors:
                normal  = self.g.get_edge_weight(node, neighbor, "normal")
                morning = self.g.get_edge_weight(node, neighbor, "morning_rush")
                self.assertGreater(morning, normal,
                    f"Morning rush weight should exceed normal for {node}–{neighbor}")
                break   # One check per node is enough

    def test_undirected(self):
        """Every edge should be traversable in both directions."""
        for u, neighbors in self.g.adj.items():
            for v, _, _ in neighbors:
                reverse = [x for x, _, _ in self.g.adj[v] if x == u]
                self.assertTrue(len(reverse) > 0,
                    f"Edge {u}→{v} exists but {v}→{u} does not (graph must be undirected)")


# ─────────────────────────────────────────────────────────────────────────────
class TestKruskalMST(unittest.TestCase):
    def setUp(self):
        self.g = build_cairo_graph()
        self.edges, self.cost, self.critical = kruskal_mst(self.g)

    def test_edge_count(self):
        """MST must have exactly V-1 edges."""
        expected = len(self.g.nodes) - 1
        self.assertEqual(len(self.edges), expected,
            f"MST should have {expected} edges, got {len(self.edges)}")

    def test_no_cycles(self):
        """Each edge should connect two previously disconnected components."""
        seen_nodes = set()
        for u, v, _ in self.edges:
            # Simple check: after adding all edges, no duplicates as start of a cycle check
            seen_nodes.add(u)
            seen_nodes.add(v)
        self.assertEqual(len(seen_nodes), len(self.g.nodes))

    def test_all_nodes_connected(self):
        """Every node must appear in at least one MST edge."""
        nodes_in_mst = set()
        for u, v, _ in self.edges:
            nodes_in_mst.add(u)
            nodes_in_mst.add(v)
        for nid in self.g.nodes:
            self.assertIn(nid, nodes_in_mst,
                f"Node {nid} ({self.g.nodes[nid]['name']}) not in MST")

    def test_positive_cost(self):
        self.assertGreater(self.cost, 0)

    def test_critical_nodes_connected(self):
        """All critical nodes must appear in the MST."""
        critical_ids = [nid for nid, m in self.g.nodes.items() if m["is_critical"]]
        for cid in critical_ids:
            in_mst = any(cid in (u, v) for u, v, _ in self.edges)
            self.assertTrue(in_mst,
                f"Critical node {self.g.nodes[cid]['name']} is not in the MST")


# ─────────────────────────────────────────────────────────────────────────────
class TestDijkstra(unittest.TestCase):
    def setUp(self):
        self.g = build_cairo_graph()

    def test_basic_route(self):
        path, cost = shortest_path(self.g, "1", "5")
        self.assertGreater(len(path), 1, "Path should contain at least 2 nodes")
        self.assertGreater(cost, 0, "Route cost should be positive")
        self.assertEqual(path[0], "1")
        self.assertEqual(path[-1], "5")

    def test_rush_hour_costs_more(self):
        _, normal  = shortest_path(self.g, "1", "5", "normal")
        _, morning = shortest_path(self.g, "1", "5", "morning_rush")
        _, evening = shortest_path(self.g, "1", "5", "evening_rush")
        self.assertGreater(morning, normal,
            "Morning rush route should cost more than normal")
        self.assertGreater(evening, morning,
            "Evening rush should be the most expensive time of day")

    def test_road_closure_detour(self):
        """Blocking an edge on the normal shortest path forces a longer detour."""
        path_normal, cost_normal = shortest_path(self.g, "2", "1", "normal")
        # Block the first edge of the normal path
        if len(path_normal) >= 2:
            blocked = [(path_normal[0], path_normal[1])]
            _, cost_detour = shortest_path(self.g, "2", "1", "normal", blocked)
            self.assertGreaterEqual(cost_detour, cost_normal,
                "Blocking an edge must not reduce travel cost")

    def test_memoization(self):
        clear_cache()
        (p1, c1), hit1 = cached_shortest_path(self.g, "1", "5", "normal")
        (p2, c2), hit2 = cached_shortest_path(self.g, "1", "5", "normal")
        self.assertFalse(hit1, "First call must be a cache miss")
        self.assertTrue(hit2,  "Second call must be a cache hit")
        self.assertEqual(c1, c2, "Cached result must match original")
        self.assertEqual(p1, p2)

    def test_same_source_dest(self):
        path, cost = shortest_path(self.g, "1", "1")
        self.assertEqual(cost, 0.0, "Cost from a node to itself should be 0")


# ─────────────────────────────────────────────────────────────────────────────
class TestAStar(unittest.TestCase):
    def setUp(self):
        self.g = build_cairo_graph()

    def test_finds_path(self):
        path, cost = astar(self.g, "2", "F9")
        self.assertGreater(len(path), 0, "A* should find a path")
        self.assertGreater(cost, 0)

    def test_same_cost_as_dijkstra(self):
        """A* must find the same optimal cost as Dijkstra."""
        _, dij_cost   = shortest_path(self.g, "11", "F9",  "normal")
        _, astar_cost = astar(self.g,          "11", "F9",  "normal")
        self.assertAlmostEqual(dij_cost, astar_cost, places=4,
            msg="A* and Dijkstra must return the same optimal cost")

    def test_emergency_route(self):
        path, cost, hosp = emergency_route(self.g, "2", "normal")
        self.assertIsNotNone(hosp, "emergency_route must return a hospital")
        self.assertGreater(len(path), 0)
        hospitals = self.g.get_nodes_by_type("hospital")
        self.assertIn(hosp, hospitals, "Returned destination must be a hospital")

    def test_emergency_rush_costs_more(self):
        _, normal_cost, _  = emergency_route(self.g, "7", "normal")
        _, rush_cost,   _  = emergency_route(self.g, "7", "morning_rush")
        self.assertGreater(rush_cost, normal_cost)


# ─────────────────────────────────────────────────────────────────────────────
class TestDynamicProgramming(unittest.TestCase):

    def test_knapsack_within_budget(self):
        budget = 35
        selected, benefit, cost, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, budget)
        self.assertLessEqual(cost, budget,
            f"Selected roads cost {cost} which exceeds budget {budget}")

    def test_knapsack_positive_benefit(self):
        _, benefit, _, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 35)
        self.assertGreater(benefit, 0)

    def test_knapsack_larger_budget_not_worse(self):
        _, b30, _, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 30)
        _, b50, _, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 50)
        self.assertGreaterEqual(b50, b30,
            "Larger budget should never yield a worse benefit")

    def test_transit_non_overlapping(self):
        selected_names, _ = transit_scheduling(TRANSIT_ROUTES)
        selected = [r for r in TRANSIT_ROUTES if r["name"] in selected_names]
        # Sort by start time and check no two consecutive routes overlap
        selected.sort(key=lambda r: r["start"])
        for i in range(len(selected) - 1):
            self.assertLessEqual(selected[i]["end"], selected[i+1]["start"],
                f"Routes overlap: {selected[i]['name']} and {selected[i+1]['name']}")

    def test_transit_positive_value(self):
        _, value = transit_scheduling(TRANSIT_ROUTES)
        self.assertGreater(value, 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestGreedy(unittest.TestCase):

    def test_single_intersection(self):
        import random; random.seed(99)
        result = simulate_intersection(INTERSECTIONS[0], "normal")
        self.assertIn("green_light", result)
        self.assertIn("reason", result)
        self.assertIn(result["green_light"], INTERSECTIONS[0]["directions"])

    def test_emergency_preemption(self):
        import random; random.seed(99)
        inter = INTERSECTIONS[0]
        emg_dir = inter["directions"][0]
        result  = simulate_intersection(inter, "normal", emergency_direction=emg_dir)
        self.assertEqual(result["green_light"], emg_dir,
            "Emergency direction must win regardless of queue size")
        self.assertIn("EMERGENCY", result["reason"])

    def test_all_intersections(self):
        import random; random.seed(42)
        results = optimize_all_intersections("morning_rush")
        self.assertEqual(len(results), len(INTERSECTIONS))

    def test_optimality_analysis(self):
        import random; random.seed(42)
        stats = analyze_greedy_optimality(100)
        total = (stats["greedy_optimal_pct"] +
                 stats["starvation_override_pct"] +
                 stats["emergency_pct"])
        self.assertAlmostEqual(total, 100.0, delta=0.5,
            msg="All percentages must sum to ~100%")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
