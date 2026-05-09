# Cairo Smart Transportation System
### Abdelrahman Mahmoud · CSE112 — Design and Analysis of Algorithms · Alamein International University

A complete transportation management system for Greater Cairo implementing five core algorithms, ML-based traffic prediction, an interactive PyQt5 GUI, and a deployable web visualizer.

---

## Project Structure

```
cairo-transport/
├── main.py                  ← CLI entry point (run all algorithms)
├── gui.py                   ← Interactive PyQt5 GUI (python main.py --gui)
├── requirements.txt         ← Python dependencies
├── Dockerfile               ← Container definition
├── docker-compose.yml       ← One-command Docker run
│
├── data/
│   └── cairo_data.py        ← Official CSE112 dataset (25 nodes, 35 edges)
├── graph/
│   └── graph.py             ← Weighted graph data structure
├── algorithims/
│   ├── mst.py               ← Kruskal's MST (modified — critical node priority)
│   ├── dijkstra.py          ← Dijkstra + road closure + memoization cache
│   ├── astar.py             ← A* emergency routing (Euclidean heuristic)
│   ├── dp.py                ← 0/1 Knapsack + Weighted Interval Scheduling
│   └── greedy.py            ← Traffic signal optimizer + emergency preemption
├── visualization/
│   └── visualizer.py        ← Generates 9 static analysis charts (matplotlib)
├── tests/
│   └── test_all.py          ← 28 unit tests (all passing)
├── output_charts/           ← Pre-generated PNG charts
└── index.html               ← Self-contained web visualizer (open in browser)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run all algorithms (CLI)
```bash
python main.py
```

### 3. Launch the interactive GUI
```bash
python main.py --gui
```

### 4. Run tests
```bash
python tests/test_all.py
```

### 5. Docker (run identically on any machine)
```bash
docker compose up --build
# → Open http://localhost:8080 for the web visualizer
```

---

## Algorithms Implemented

### A. Kruskal's MST (Modified)
**File:** `algorithims/mst.py`

Builds the minimum spanning tree connecting all 25 nodes. Critical facilities (hospitals, government buildings) are assigned to tier-0 in the sort order and connected before any other edges, even if a cheaper option exists elsewhere.

- **Time:** O(E log E)
- **Space:** O(V)
- **Result:** 24 edges, all 10 critical nodes connected

### B. Dijkstra's Shortest Path
**File:** `algorithims/dijkstra.py`

Standard Dijkstra with three modifications:
1. **Time-dependent weights** — rush-hour multipliers (×2.5 morning, ×2.8 evening) applied to all edge weights
2. **Road closure support** — any edge can be blocked at query time; the algorithm finds the best detour automatically
3. **Memoization cache** — `cached_shortest_path()` stores results so repeated queries skip re-computation

- **Time:** O((V + E) log V)
- **Space:** O(V)

### C. A\* Search
**File:** `algorithims/astar.py`

Used for emergency vehicle routing. The Euclidean distance between node coordinates is used as the admissible heuristic h(n). The `emergency_route()` function checks all hospitals and returns the nearest one.

- **Advantage over Dijkstra:** Explores significantly fewer nodes for directed queries
- **Heuristic:** `h(n) = sqrt((x₂-x₁)² + (y₂-y₁)²)` — always admissible

### D. Dynamic Programming
**File:** `algorithims/dp.py`

Two sub-problems:

**0/1 Knapsack — Road Maintenance:**
Given a budget and a list of maintenance projects, choose the subset maximising total benefit. DP table has O(n × W) states; traceback reconstructs the selection.

**Weighted Interval Scheduling — Transit:**
Choose the non-overlapping set of bus/metro routes maximising total daily passengers. Standard DP on intervals sorted by finish time.

### E. Greedy Traffic Signals
**File:** `algorithims/greedy.py`

At each intersection, the lane with the largest queue gets the green light. Two overrides take priority:
1. **Emergency preemption** — ambulance/fire truck direction always wins
2. **Starvation prevention** — any lane waiting ≥ max_wait_cap cycles is forced to get a turn

Optimality analysis over 500 simulations reveals when the greedy choice is optimal vs. when overrides fire.

---

## Data Source

All node and edge data taken directly from the official **CSE112-Project Provided Data** PDF:
- 15 districts/neighbourhoods + 10 important facilities = **25 nodes**
- 28 existing roads + 7 hospital/facility connections = **35 edges**
- Road condition scores (1–10) converted to traffic multipliers: `1 + (10 - condition) × 0.1`
- Public transport routes: 3 metro lines + 10 bus routes (used in DP transit scheduling)

---

## ML Traffic Prediction (Bonus)

**File:** `generate_ml_data.py`
**Live demo:** `index.html` tab "ML Prediction"

Two scikit-learn models trained on 168 temporal samples (24 hours × 7 days):
- `GradientBoostingRegressor` — predicts traffic multiplier (continuous)
- `RandomForestClassifier` — predicts congestion level (low / medium / high)

Features: `hour`, `day_of_week`, `is_weekend`, `sin(2π·hour/24)`, `cos(2π·hour/24)`

The cyclic hour encoding lets the model understand that 23:00 and 00:00 are adjacent.

---

## Algorithm Complexity Summary

| Algorithm | Time Complexity | Space |
|---|---|---|
| Kruskal's MST | O(E log E) | O(V) |
| Dijkstra | O((V+E) log V) | O(V) |
| A\* Search | O(b^d) best / O(V log V) worst | O(V) |
| DP Knapsack | O(n × W) | O(n × W) |
| Interval Scheduling | O(n log n) | O(n) |
| Greedy Signals | O(d) per intersection | O(1) |

---

## Deployment

### GitHub Pages (web visualizer)
```bash
git init && git add .
git commit -m "Cairo transport system"
git remote add origin https://github.com/6sangoku9/cairo-transport.git
git push -u origin main
# Settings → Pages → main / root → Save
# Live at: https:/6sangoku9.github.io/cairo-transport
```

### Vercel
```bash
npm i -g vercel && vercel --prod
```

### Docker
```bash
docker compose up --build   # → http://localhost:8080
```

---

## Test Results

```
Ran 28 tests in 0.010s — OK

TestGraph            (5 tests)  ✓
TestKruskalMST       (5 tests)  ✓
TestDijkstra         (5 tests)  ✓
TestAStar            (4 tests)  ✓
TestDynamicProgramming (5 tests) ✓
TestGreedy           (4 tests)  ✓
```
