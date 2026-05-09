"""
visualizer.py  —  Static chart generation for Cairo Transport System
Produces 9 PNG files in output_charts/ covering all major algorithm outputs.
Called automatically by main.py, or import draw_all() directly.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
random.seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from data.cairo_data  import build_cairo_graph
from algorithims.mst      import kruskal_mst
from algorithims.dijkstra  import shortest_path
from algorithims.astar     import emergency_route
from algorithims.dp        import (road_maintenance_knapsack, MAINTENANCE_ROADS,
                                   transit_scheduling, TRANSIT_ROUTES)
from algorithims.greedy    import optimize_all_intersections, analyze_greedy_optimality

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_charts")

# ── Colour palette (matches GUI) ─────────────────────────────────────────────
BG   = "#08090c"
BG2  = "#0e1117"
BG3  = "#141821"
BOR  = "#1e2535"
C_MST  = "#00e5b4"
C_DIJ  = "#4da6ff"
C_AST  = "#ff6b6b"
C_DP   = "#ffcc44"
C_GRD  = "#c77dff"
TXT    = "#e4e9f2"
TXT2   = "#5d6b82"

NODE_COLORS = {
    "hospital":   "#ff6b6b",
    "government": "#ffcc44",
    None:         "#4da6ff",
}


def _fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG2)
    ax.set_facecolor(BG3)
    for spine in ax.spines.values():
        spine.set_color(BOR)
    ax.tick_params(colors=TXT2, labelsize=8)
    ax.xaxis.label.set_color(TXT2)
    ax.yaxis.label.set_color(TXT2)
    ax.title.set_color(TXT)
    return fig, ax


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG2)
    plt.close(fig)
    print(f"  Saved: {name}")


def _node_positions(graph):
    """Return {node_id: (px, py)} scaled to a reasonable coordinate space."""
    xs = [m["x"] for m in graph.nodes.values()]
    ys = [m["y"] for m in graph.nodes.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max(max_x - min_x, 0.001)
    ry = max(max_y - min_y, 0.001)
    return {
        nid: ((m["x"] - min_x) / rx * 10, (m["y"] - min_y) / ry * 6)
        for nid, m in graph.nodes.items()
    }


# ── Chart 1: Full Network ─────────────────────────────────────────────────────
def chart_full_network(graph):
    fig, ax = _fig()
    pos = _node_positions(graph)

    drawn = set()
    for u in graph.adj:
        for v, dist, _ in graph.adj[u]:
            key = tuple(sorted([u, v]))
            if key in drawn:
                continue
            drawn.add(key)
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=BOR, linewidth=0.8, alpha=0.7, zorder=1)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{dist:.1f}", fontsize=5, color=TXT2, ha="center", zorder=2)

    for nid, (px, py) in pos.items():
        meta  = graph.nodes[nid]
        color = NODE_COLORS.get(meta["critical_type"], NODE_COLORS[None])
        size  = 120 if meta["is_critical"] else 60
        ax.scatter(px, py, s=size, color=color, zorder=3, edgecolors=BG2, linewidths=1)
        ax.text(px, py + 0.22, meta["name"].split()[0][:8],
                fontsize=6, color=TXT, ha="center", va="bottom", zorder=4)

    legend = [
        mpatches.Patch(color=C_DIJ,  label="District"),
        mpatches.Patch(color=C_AST,  label="Hospital"),
        mpatches.Patch(color=C_DP,   label="Facility / Govt"),
    ]
    ax.legend(handles=legend, fontsize=8, facecolor=BG3, labelcolor=TXT,
              loc="lower right", framealpha=0.9)
    ax.set_title("Cairo Road Network — All Nodes & Edges", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "01_full_network.png")


# ── Chart 2: MST ──────────────────────────────────────────────────────────────
def chart_mst(graph):
    fig, ax = _fig()
    pos = _node_positions(graph)
    mst_edges, total_cost, _ = kruskal_mst(graph)
    mst_set = {tuple(sorted([u, v])): (u, v) for u, v, _ in mst_edges}

    drawn = set()
    for u in graph.adj:
        for v, _, _ in graph.adj[u]:
            key = tuple(sorted([u, v]))
            if key in drawn:
                continue
            drawn.add(key)
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if key in mst_set:
                is_crit = graph.nodes[u]["is_critical"] or graph.nodes[v]["is_critical"]
                col = C_AST if is_crit else C_MST
                ax.plot([x1, x2], [y1, y2], color=col, linewidth=2.2, zorder=2)
            else:
                ax.plot([x1, x2], [y1, y2], color=BOR, linewidth=0.6, alpha=0.4, zorder=1)

    for nid, (px, py) in pos.items():
        meta  = graph.nodes[nid]
        color = NODE_COLORS.get(meta["critical_type"], NODE_COLORS[None])
        size  = 110 if meta["is_critical"] else 55
        ax.scatter(px, py, s=size, color=color, zorder=3, edgecolors=BG2, linewidths=1)
        ax.text(px, py + 0.22, meta["name"].split()[0][:8],
                fontsize=6, color=TXT, ha="center", va="bottom", zorder=4)

    legend = [
        Line2D([0], [0], color=C_AST, linewidth=2, label="Critical edge"),
        Line2D([0], [0], color=C_MST, linewidth=2, label="MST edge"),
        Line2D([0], [0], color=BOR,   linewidth=1, label="Non-MST edge"),
    ]
    ax.legend(handles=legend, fontsize=8, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    ax.set_title(f"Kruskal's MST  —  {len(mst_edges)} edges, total {total_cost:.1f} km", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "02_mst.png")


# ── Chart 3: Shortest Paths ───────────────────────────────────────────────────
def chart_shortest_paths(graph):
    pairs = [("1","5"), ("2","8"), ("11","14"), ("1","4")]
    tods  = ["normal", "morning_rush", "evening_rush"]
    cols  = [C_MST, C_DP, C_AST]
    fig, ax = _fig()
    w = 0.25
    for i, (tod, col) in enumerate(zip(tods, cols)):
        costs = [shortest_path(graph, s, d, tod)[1] for s, d in pairs]
        xs    = [xi + (i - 1) * w for xi in range(len(pairs))]
        ax.bar(xs, costs, w, label=tod.replace("_", " ").title(),
               color=col, edgecolor=BOR, linewidth=0.5, alpha=0.9)
    ax.set_xticks(list(range(len(pairs))))
    ax.set_xticklabels(
        [f"{graph.nodes[s]['name']}\n→ {graph.nodes[d]['name']}" for s, d in pairs],
        fontsize=8, color=TXT2
    )
    ax.set_ylabel("Travel Cost (km × traffic)", color=TXT2)
    ax.set_title("Rush Hour Impact on Dijkstra's Shortest Paths", fontsize=12)
    ax.legend(fontsize=8, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "03_shortest_paths.png")


# ── Chart 4: Emergency Routing ────────────────────────────────────────────────
def chart_emergency(graph):
    fig, ax = _fig()
    pos = _node_positions(graph)

    incidents = ["2", "7", "11"]
    colors    = ["#ff9f43", "#ee5a24", "#e55039"]

    drawn = set()
    for u in graph.adj:
        for v, _, _ in graph.adj[u]:
            key = tuple(sorted([u, v]))
            if key in drawn: continue
            drawn.add(key)
            x1, y1 = pos[u]; x2, y2 = pos[v]
            ax.plot([x1,x2],[y1,y2], color=BOR, linewidth=0.6, alpha=0.4, zorder=1)

    for inc, col in zip(incidents, colors):
        path, cost, hosp = emergency_route(graph, inc, "morning_rush")
        for i in range(len(path)-1):
            x1,y1 = pos[path[i]]; x2,y2 = pos[path[i+1]]
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.8), zorder=3)
        label = f"{graph.nodes[inc]['name']} → {graph.nodes[hosp]['name']} ({cost:.1f}km)"
        ax.plot([], [], color=col, linewidth=2, label=label)

    for nid, (px, py) in pos.items():
        meta  = graph.nodes[nid]
        color = NODE_COLORS.get(meta["critical_type"], NODE_COLORS[None])
        size  = 130 if meta["critical_type"] == "hospital" else (70 if meta["is_critical"] else 45)
        ax.scatter(px, py, s=size, color=color, zorder=4, edgecolors=BG2, linewidths=1)
        if meta["critical_type"] == "hospital":
            ax.text(px, py+0.25, "🏥", fontsize=10, ha="center", zorder=5)

    ax.legend(fontsize=7, facecolor=BG3, labelcolor=TXT, framealpha=0.9, loc="lower right")
    ax.set_title("A* Emergency Routing — Nearest Hospital (Morning Rush)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "04_emergency_routing.png")


# ── Chart 5: Knapsack ─────────────────────────────────────────────────────────
def chart_knapsack():
    selected, benefit, cost, _ = road_maintenance_knapsack(MAINTENANCE_ROADS, 35)
    fig, ax = _fig()
    names   = [r["name"][:18] for r in MAINTENANCE_ROADS]
    bens    = [r["benefit"]   for r in MAINTENANCE_ROADS]
    costs_  = [r["cost"]      for r in MAINTENANCE_ROADS]
    colors  = [C_MST if r["name"] in selected else BOR for r in MAINTENANCE_ROADS]
    bars = ax.bar(range(len(names)), bens, color=colors, edgecolor=BOR, linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=7, color=TXT2)
    ax.set_ylabel("Benefit Score", color=TXT2)
    ax.set_title(f"0/1 Knapsack — Road Maintenance  (Budget=35, Benefit={benefit}, Spent={cost})", fontsize=11)
    for i, (b, c) in enumerate(zip(bens, costs_)):
        ax.text(i, b + 0.5, f"c={c}", ha="center", fontsize=6, color=TXT2)
    legend = [mpatches.Patch(color=C_MST, label="Selected"),
              mpatches.Patch(color=BOR,   label="Not selected")]
    ax.legend(handles=legend, fontsize=8, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "05_dp_knapsack.png")


# ── Chart 6: Transit Scheduling ───────────────────────────────────────────────
def chart_transit():
    selected, val = transit_scheduling(TRANSIT_ROUTES)
    fig, ax = _fig(10, 5)
    sorted_r = sorted(TRANSIT_ROUTES, key=lambda r: r["start"])
    for i, r in enumerate(sorted_r):
        col = C_DP if r["name"] in selected else BOR
        ax.barh(i, r["end"]-r["start"], left=r["start"],
                color=col, edgecolor=BOR, height=0.65, linewidth=0.5)
        short = r["name"].split("—")[-1].strip() if "—" in r["name"] else r["name"]
        ax.text(r["start"]+0.1, i, f" {short[:20]}", va="center", fontsize=7, color=TXT)
    ax.set_yticks(range(len(sorted_r)))
    ax.set_yticklabels([r["name"][:16] for r in sorted_r], fontsize=7, color=TXT2)
    ax.set_xlabel("Hour of Day", color=TXT2)
    ax.set_xlim(5, 24)
    ax.set_title(f"Weighted Interval Scheduling — {val} passengers served", fontsize=12)
    legend = [mpatches.Patch(color=C_DP, label="Selected"),
              mpatches.Patch(color=BOR,  label="Not selected")]
    ax.legend(handles=legend, fontsize=8, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "06_transit_scheduling.png")


# ── Chart 7: Rush Hour Comparison ─────────────────────────────────────────────
def chart_rush_hour_comparison(graph):
    route = ("1", "5")
    src_name = graph.nodes[route[0]]["name"]
    dst_name = graph.nodes[route[1]]["name"]
    hours = list(range(24))
    tod_map = {h: ("morning_rush" if 7 <= h < 10
                   else "evening_rush" if 16 <= h < 20
                   else "normal") for h in hours}
    costs = [shortest_path(graph, *route, tod_map[h])[1] for h in hours]
    colors = [C_AST if tod_map[h]=="evening_rush"
              else C_DP if tod_map[h]=="morning_rush"
              else C_MST for h in hours]
    fig, ax = _fig()
    ax.bar(hours, costs, color=colors, edgecolor=BOR, linewidth=0.3)
    ax.set_xlabel("Hour of Day", color=TXT2)
    ax.set_ylabel("Travel Cost (km × traffic)", color=TXT2)
    ax.set_title(f"24-Hour Cost Profile — {src_name} → {dst_name}", fontsize=12)
    legend = [mpatches.Patch(color=C_MST, label="Normal"),
              mpatches.Patch(color=C_DP,  label="Morning Rush"),
              mpatches.Patch(color=C_AST, label="Evening Rush")]
    ax.legend(handles=legend, fontsize=8, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "07_rush_hour_comparison.png")


# ── Chart 8: Greedy Optimality ────────────────────────────────────────────────
def chart_greedy():
    analysis = analyze_greedy_optimality(500)
    vals   = [analysis["greedy_optimal_pct"],
              analysis["starvation_override_pct"],
              analysis["emergency_pct"]]
    labels = ["Greedy Optimal", "Starvation Override", "Emergency Preemption"]
    colors = [C_MST, C_DP, C_AST]
    fv = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]
    fig, ax = _fig(7, 5)
    if fv:
        vs, ls, cs = zip(*fv)
        _, _, autos = ax.pie(vs, labels=ls, colors=cs, autopct="%1.1f%%",
                              startangle=140, textprops={"color": TXT, "fontsize": 9})
        for a in autos:
            a.set_color(BG)
            a.set_fontweight("bold")
    ax.set_title("Greedy Signal Optimality — 500 Simulations", fontsize=12)
    fig.tight_layout()
    _save(fig, "08_greedy_analysis.png")


# ── Chart 9: Road Closure ─────────────────────────────────────────────────────
def chart_road_closure(graph):
    blocked = [("2", "3"), ("5", "11")]
    pairs   = [("2", "1"), ("11", "4")]
    fig, ax = _fig()
    labels, normal_costs, detour_costs = [], [], []
    for s, d in pairs:
        normal_path, nc = shortest_path(graph, s, d, "normal")
        detour_path, dc = shortest_path(graph, s, d, "normal", blocked)
        labels.append(f"{graph.nodes[s]['name']}\n→ {graph.nodes[d]['name']}")
        normal_costs.append(nc)
        detour_costs.append(dc)
    x = range(len(labels))
    w = 0.35
    ax.bar([xi - w/2 for xi in x], normal_costs, w, color=C_MST,
           label="Normal Route", edgecolor=BOR, linewidth=0.5)
    ax.bar([xi + w/2 for xi in x], detour_costs, w, color=C_AST,
           label="Detour Route", edgecolor=BOR, linewidth=0.5)
    for xi, (nc, dc) in enumerate(zip(normal_costs, detour_costs)):
        ax.text(xi, max(nc, dc) + 0.5, f"+{dc-nc:.1f}", ha="center", fontsize=9, color=TXT2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=TXT2)
    ax.set_ylabel("Travel Cost (km)", color=TXT2)
    ax.set_title("Road Closure — Normal vs Detour Route Cost", fontsize=12)
    ax.legend(fontsize=9, facecolor=BG3, labelcolor=TXT, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "09_road_closure.png")


# ── Master draw function ──────────────────────────────────────────────────────
def draw_all():
    print("  Generating visualizations …")
    graph = build_cairo_graph()
    chart_full_network(graph)
    chart_mst(graph)
    chart_shortest_paths(graph)
    chart_emergency(graph)
    chart_knapsack()
    chart_transit()
    chart_rush_hour_comparison(graph)
    chart_greedy()
    chart_road_closure(graph)
    print(f"  All charts saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    draw_all()
