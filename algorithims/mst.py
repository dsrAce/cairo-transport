"""
mst.py  —  Kruskal's Minimum Spanning Tree (modified for Cairo)

Modification: critical facilities (hospitals, government buildings) and
high-population areas are sorted to the front of the edge list so they
get connected first, even if a slightly cheaper edge exists elsewhere.

Time complexity:  O(E log E)  — dominated by the sort step
Space complexity: O(V)        — the Union-Find structure
"""


# ── Union-Find (Disjoint Set Union) ──────────────────────────────────────────
class UnionFind:
    """
    Keeps track of which nodes are already connected.
    Uses path compression + union-by-rank for near-O(1) operations.
    """
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank   = {n: 0  for n in nodes}

    def find(self, x):
        # Path compression: point every node directly to the root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Merge the sets containing x and y. Returns False if already same set."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False                    # Already connected — would create a cycle
        # Attach smaller tree under the larger tree
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


# ── Main algorithm ────────────────────────────────────────────────────────────
def kruskal_mst(graph):
    """
    Build the minimum spanning tree for Cairo's road network.

    Returns:
        mst_edges        — list of (u, v, weight) tuples in the MST
        total_cost       — sum of all edge weights
        critical_connected — list of critical node IDs that appear in the MST
    """
    all_edges = graph.get_all_edges()   # [(u, v, weight), …]

    # Tag which nodes are critical or high-population
    critical_nodes   = {nid for nid, m in graph.nodes.items() if m["is_critical"]}
    high_pop_nodes   = {nid for nid, m in graph.nodes.items() if m["population"] >= 200_000}

    def sort_key(edge):
        u, v, weight = edge
        touches_critical = (u in critical_nodes) or (v in critical_nodes)
        both_high_pop    = (u in high_pop_nodes)  and (v in high_pop_nodes)

        # Tier 0: connects a critical facility → always consider first
        # Tier 1: connects two high-population areas → second priority
        # Tier 2: everything else → sort purely by weight
        if touches_critical:
            tier = 0
        elif both_high_pop:
            tier = 1
        else:
            tier = 2
        return (tier, weight)

    all_edges.sort(key=sort_key)

    uf         = UnionFind(graph.adj.keys())
    mst_edges  = []
    total_cost = 0.0

    for u, v, weight in all_edges:
        if uf.union(u, v):          # Only add if it doesn't create a cycle
            mst_edges.append((u, v, weight))
            total_cost += weight
            if len(mst_edges) == len(graph.adj) - 1:
                break               # MST is complete (V-1 edges)

    # Report which critical nodes ended up connected
    critical_connected = [
        nid for nid in critical_nodes
        if any(nid in (u, v) for u, v, _ in mst_edges)
    ]

    return mst_edges, total_cost, critical_connected


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.cairo_data import build_cairo_graph

    g = build_cairo_graph()
    edges, cost, critical = kruskal_mst(g)

    print(f"MST edges        : {len(edges)}")
    print(f"Total cost       : {cost:.2f} km")
    print(f"Critical connected: {len(critical)} / {len([n for n,m in g.nodes.items() if m['is_critical']])}")
    for u, v, w in edges:
        tag = " [CRITICAL]" if g.nodes[u]["is_critical"] or g.nodes[v]["is_critical"] else ""
        print(f"  {g.nodes[u]['name']}  ↔  {g.nodes[v]['name']}   {w:.2f} km{tag}")
