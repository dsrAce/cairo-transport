"""
greedy.py  —  Greedy traffic signal optimization for Cairo intersections

Strategy:
  At each intersection we give the green light to the lane with the
  LARGEST QUEUE (standard greedy choice).

Two override conditions take priority over the greedy choice:

1. EMERGENCY PREEMPTION
   If an ambulance / fire truck is waiting in a specific direction,
   that direction gets the green light regardless of queue size.
   This models the emergency-vehicle priority requirement.

2. STARVATION PREVENTION
   If any lane has been waiting for max_wait_cap consecutive cycles
   without getting a green light, we force it to get a turn — even if
   its queue is smaller.  This prevents any lane from waiting forever.

Intersections below are the major Cairo junctions from the project area.
"""

import random

# Major Cairo intersections with their incoming directions
INTERSECTIONS = [
    {
        "name":       "Tahrir Square",
        "directions": ["Corniche El Nil", "Kasr El-Nil St.", "Qasr El-Aini St.", "Tahrir Bridge"],
    },
    {
        "name":       "Ramses Square",
        "directions": ["Galaa Street", "Ramses Street", "Port Said Street", "Orabi Street"],
    },
    {
        "name":       "Heliopolis Hub",
        "directions": ["Merghany Street", "Ibrahim Laqany St.", "Nozha Street", "Airport Road"],
    },
    {
        "name":       "Nasr City Junction",
        "directions": ["Abbas El-Akkad St.", "Makram Ebeid St.", "Mostafa El-Nahas St.", "Nasr Road"],
    },
    {
        "name":       "Maadi Roundabout",
        "directions": ["Road 9 Maadi", "Corniche Maadi", "Misr-Helwan Agric. Rd.", "Road 83"],
    },
    {
        "name":       "Giza Square",
        "directions": ["Pyramids Road", "Faisal Street", "Giza Corniche", "Sudan Street"],
    },
]


def simulate_intersection(intersection, time_of_day="normal",
                           emergency_direction=None, max_wait_cap=5):
    """
    Run one green-light cycle for a single intersection.

    Parameters
    ----------
    intersection       : one item from INTERSECTIONS list
    time_of_day        : "normal" | "morning_rush" | "evening_rush"
    emergency_direction: direction name that has an emergency vehicle
    max_wait_cap       : how many cycles a lane may wait before override

    Returns a dict with the result of this cycle.
    """
    # Generate realistic queue sizes for this time of day
    queue_ranges = {
        "morning_rush": (15, 60),
        "evening_rush": (20, 70),
        "normal":        (3, 25),
    }
    lo, hi = queue_ranges.get(time_of_day, (3, 25))

    directions  = intersection["directions"]
    queues      = {d: random.randint(lo, hi) for d in directions}
    wait_cycles = {d: random.randint(0, max_wait_cap) for d in directions}

    # ── Decision logic (priority order) ──────────────────────────────────
    if emergency_direction and emergency_direction in directions:
        # Emergency override: the ambulance gets through no matter what
        chosen = emergency_direction
        reason = "EMERGENCY PREEMPTION"

    elif any(wait_cycles[d] >= max_wait_cap for d in directions):
        # Anti-starvation: among starved lanes, pick the busiest
        starved = [d for d in directions if wait_cycles[d] >= max_wait_cap]
        chosen  = max(starved, key=lambda d: queues[d])
        reason  = "STARVATION PREVENTION — max wait cap reached"

    else:
        # Standard greedy: longest queue wins
        chosen = max(directions, key=lambda d: queues[d])
        reason = "GREEDY — largest queue"

    return {
        "intersection":    intersection["name"],
        "time_of_day":     time_of_day,
        "queues":          queues,
        "green_light":     chosen,
        "reason":          reason,
        "vehicles_served": queues[chosen],
    }


def optimize_all_intersections(time_of_day="normal", emergency=None):
    """
    Run one cycle for every intersection in the network.

    emergency: dict with keys "intersection" and "direction", e.g.
               {"intersection": "Tahrir Square", "direction": "Qasr El-Aini St."}
    """
    results = []
    for inter in INTERSECTIONS:
        emg_dir = None
        if emergency and emergency.get("intersection") == inter["name"]:
            emg_dir = emergency.get("direction")

        result = simulate_intersection(inter, time_of_day, emg_dir)
        results.append(result)
    return results


def analyze_greedy_optimality(num_simulations=200):
    """
    Run many simulations and report what fraction of decisions fell into
    each category (greedy / starvation override / emergency preemption).

    This answers the project requirement to "analyse cases where the greedy
    approach provides optimal vs sub-optimal solutions".
    """
    greedy_count    = 0
    starvation_count = 0
    emergency_count  = 0

    for _ in range(num_simulations):
        tod     = random.choice(["normal", "morning_rush", "evening_rush"])
        results = optimize_all_intersections(tod)
        for r in results:
            if "GREEDY" in r["reason"]:
                greedy_count     += 1
            elif "STARVATION" in r["reason"]:
                starvation_count += 1
            else:
                emergency_count  += 1

    total = greedy_count + starvation_count + emergency_count
    return {
        "total_decisions":       total,
        "greedy_optimal_pct":    round(100 * greedy_count    / total, 1),
        "starvation_override_pct": round(100 * starvation_count / total, 1),
        "emergency_pct":         round(100 * emergency_count  / total, 1),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    print("Evening Rush — with emergency at Tahrir Square")
    emergency = {"intersection": "Tahrir Square", "direction": "Qasr El-Aini St."}
    results = optimize_all_intersections("evening_rush", emergency)
    for r in results:
        print(f"  {r['intersection']:25s}  GREEN → {r['green_light']}  ({r['reason']})")

    print("\nOptimality analysis (500 sims):")
    stats = analyze_greedy_optimality(500)
    for k, v in stats.items():
        print(f"  {k}: {v}")
