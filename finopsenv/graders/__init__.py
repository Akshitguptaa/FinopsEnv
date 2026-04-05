from __future__ import annotations

import math
from typing import Optional

from ..simulation.constants import (
    W_EFFICIENCY, W_SLA, W_CARBON, W_CHURN,
    TASK_CONFIGS,
)

class RewardShaper:
    # real time RL rewards for each step 
    # this class generates a continuous signal (efficiency vs. SLA vs. carbon) to help agents learn.

    def __init__(self, task_id: int):
        self.task_id = task_id
        cfg = TASK_CONFIGS[task_id]
        self._budget = cfg["budget_usd"]
        self._step_cost_baseline: Optional[float] = None

    def set_step_cost_baseline(self, baseline: float) -> None:
        self._step_cost_baseline = max(baseline, 1e-6)

    def compute(
        self,
        step_cost_usd: float,
        dropped_requests: int,
        total_rps: int,
        step_carbon_kg: float,
        provision_event: bool,
        terminate_event: bool,
    ) -> float:
        if self._step_cost_baseline:
            r_eff = (self._step_cost_baseline - step_cost_usd) / self._step_cost_baseline
        else:
            r_eff = 0.0

        drop_frac = dropped_requests / max(total_rps, 1)
        r_sla = drop_frac * (1 + drop_frac * 3)

        r_carbon = 0.0
        if self.task_id == 3 and step_carbon_kg > 0:
            carbon_per_rps = step_carbon_kg / max(total_rps, 1)
            r_carbon = math.exp(-carbon_per_rps * 5000)

        r_churn = 1.0 if (provision_event or terminate_event) else 0.0

        return float(
            W_EFFICIENCY * r_eff
            - W_SLA * r_sla
            + W_CARBON * r_carbon
            - W_CHURN * r_churn
        )


def grade_task1(state: dict) -> float:
    # Task 1: Basic Auto-Scaling
    # Agent must maintain SLAs while finding the absolute minimum number of instances 
    # to handle the static load. Any dropped requests instantly fail the run (score 0).

    if state["dropped_requests"] > 0:
        return 0.0
    if state["terminate_events"] == 0:
        return 0.0

    # Minimum viable fleet: ceil(1000 RPS / 250 per node / 0.85 target CPU) = 5 nodes
    optimal_nodes = 5
    optimal_cost_per_step = optimal_nodes * 0.096 * (5 / 60)
    optimal_total = optimal_cost_per_step * state["max_steps"]

    worst_cost = state.get("initial_cost_reference", optimal_total * 2.5)
    agent_cost = state["total_cost_usd"]

    if agent_cost >= worst_cost:
        return 0.1
    if agent_cost <= optimal_total:
        return 1.0

    raw = (worst_cost - agent_cost) / (worst_cost - optimal_total)
    return round(float(min(1.0, max(0.1, raw))), 4)


def grade_task2(state: dict) -> float:
    # Task 2: Spot Instance Optimization
    # Agent must leverage cheap spot instances to beat typical on-demand costs, 
    # while surviving random hardware evictions without massively failing SLAs.

    total_rps = state.get("total_rps_served", 0) + state.get("total_rps_dropped", 0)
    if total_rps == 0:
        return 0.0

    drop_fraction = state["total_rps_dropped"] / max(total_rps, 1)
    if drop_fraction > 0.001:
        return 0.0

    heuristic_cost = state.get("heuristic_cost_reference", 1.0)
    agent_cost = state["total_cost_usd"]

    if agent_cost >= heuristic_cost:
        return 0.0

    raw = (heuristic_cost - agent_cost) / heuristic_cost
    return round(float(min(1.0, max(0.0, raw))), 4)


def grade_task3(state: dict) -> float:
    # Task 3: Carbon Aware Traffic Routing
    # Agent must dynamically shift traffic between global regions to 'chase the sun',
    # utilizing data centers where sustainable energy output is currently highest, 
    # minimizing total carbon emissions beneath the cap.

    if state["total_cost_usd"] > state["budget_total_usd"]:
        return 0.0

    max_carbon = state.get("max_carbon_kg", 50.0)
    if state["total_carbon_kg"] > max_carbon:
        return 0.0

    total_rps = state.get("total_rps_served", 0) + state.get("total_rps_dropped", 0)
    drop_fraction = state["total_rps_dropped"] / max(total_rps, 1)
    if drop_fraction > 0.001:
        return 0.0

    s_cost   = float(max(0.0, min(1.0, 1.0 - state["total_cost_usd"] / state["budget_total_usd"])))
    s_carbon = float(max(0.0, min(1.0, 1.0 - state["total_carbon_kg"] / max_carbon)))
    s_sla    = float(max(0.0, min(1.0, 1.0 - drop_fraction * 1000)))

    # Weighted harmonic mean - cost 40%, carbon 35%, SLA 25%
    weights = [0.40, 0.35, 0.25]
    subs    = [s_cost, s_carbon, s_sla]
    eps     = 1e-9
    hm = sum(weights) / sum(w / max(s, eps) for w, s in zip(weights, subs))

    return round(float(min(1.0, max(0.0, hm))), 4)


GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}
