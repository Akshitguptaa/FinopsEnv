from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple

from .simulation import (
    TrafficGenerator, PricingEngine, CarbonGrid, ClusterSimulator,
    TASK_CONFIGS, REGIONS,
)
from .schemas.action import FinOpsAction
from .schemas.observation import FinOpsObservation, FinOpsState, RegionMetrics
from .graders import GRADERS, RewardShaper
from .simulation.constants import MINUTES_PER_STEP


class FinOpsEnv:
    def __init__(self) -> None:
        self._ready   = False
        self._task_id = 1
        self._seed    = 42
        self._sim_step   = 0
        self._sim_minute = 0
        self._done       = False
        self._termination_reason: Optional[str] = None
        self._total_rps_served  = 0
        self._total_rps_dropped = 0
        self._last_provision = False
        self._last_terminate = False

        self._rng:            Optional[np.random.Generator] = None
        self._traffic:        Optional[TrafficGenerator]    = None
        self._pricing:        Optional[PricingEngine]       = None
        self._carbon:         Optional[CarbonGrid]          = None
        self._cluster:        Optional[ClusterSimulator]    = None
        self._reward_shaper:  Optional[RewardShaper]        = None
        self._task_cfg:       Optional[Dict]                = None
        self._step_cost_baseline         = 0.0
        self._heuristic_cost_reference   = 0.0
        self._initial_cost_reference     = 0.0

    def reset(self, seed: int = 42, task_id: int = 1) -> FinOpsObservation:
        # Resets to the initial state for the requested task.
        # re-initialise all background engines  
        # managing their random seeds 
        
        assert task_id in (1, 2, 3), f"task_id must be 1, 2, or 3, got {task_id}"

        self._seed       = seed
        self._task_id    = task_id
        self._task_cfg   = TASK_CONFIGS[task_id]
        self._sim_step   = 0
        self._sim_minute = 0
        self._done       = False
        self._termination_reason = None
        self._total_rps_served   = 0
        self._total_rps_dropped  = 0
        self._last_provision     = False
        self._last_terminate     = False

        cfg       = self._task_cfg
        max_steps = cfg["max_steps"]

        self._rng     = np.random.default_rng(seed)
        self._pricing = PricingEngine(rng=np.random.default_rng(seed + 1), enable_spot=cfg["enable_spot"])
        self._carbon  = CarbonGrid(rng=np.random.default_rng(seed + 2),    enable_carbon=cfg["enable_carbon"])
        self._traffic = TrafficGenerator(rng=np.random.default_rng(seed + 3), base_rps=cfg["base_rps"], mode=cfg["traffic_mode"])
        self._traffic.seed_bursts(max_steps)

        self._cluster = ClusterSimulator(
            rng=np.random.default_rng(seed + 4),
            pricing_engine=self._pricing,
            initial_nodes=dict(cfg["initial_nodes"]),
            failure_enabled=(task_id >= 2),
            enable_egress=(task_id == 3),
        )

        initial_node_count       = sum(cfg["initial_nodes"].values())
        self._step_cost_baseline = initial_node_count * 0.096 * (MINUTES_PER_STEP / 60.0)

        self._reward_shaper = RewardShaper(task_id)
        self._reward_shaper.set_step_cost_baseline(self._step_cost_baseline)

        heuristic_nodes = int(initial_node_count * 1.2)
        self._heuristic_cost_reference = heuristic_nodes * 0.096 * (MINUTES_PER_STEP / 60.0) * max_steps
        self._initial_cost_reference   = initial_node_count * 0.096 * (MINUTES_PER_STEP / 60.0) * max_steps

        self._ready = True
        return self._build_observation()

    def step(self, action: FinOpsAction) -> Tuple[FinOpsObservation, float, bool, Dict]:
        # Submits an action to the environment 
        # the cluster simulation to tick forward to accrue costs, handle RPS, and 

        if not self._ready or self._done:
            raise RuntimeError("Episode not started. Call reset() first.")

        self._last_provision = False
        self._last_terminate = False
        info: Dict[str, Any] = {"action_result": "ok"}

        if action.action_type == "provision_node":
            region  = action.region or "us-east-1"
            tier    = action.instance_tier or "standard"
            billing = action.billing_model or "on-demand"
            nid = self._cluster.provision_node(region, tier, billing)
            info["provisioned_node_id"] = nid
            self._last_provision = True

        elif action.action_type == "terminate_node":
            if action.node_id:
                success = self._cluster.terminate_node(action.node_id)
                info["terminate_success"] = success
                self._last_terminate = success
            else:
                info["action_result"] = "missing node_id"

        elif action.action_type == "migrate_traffic":
            src = action.source_region or "us-east-1"
            dst = action.target_region or "eu-west-1"
            pct = action.percentage if action.percentage is not None else 0.25
            info["migrate_success"] = self._cluster.migrate_traffic(src, dst, pct)

        self._pricing.step()

        sim_hour = (self._sim_minute // 60) % 24
        total_rps, origin_weights = self._traffic.step(self._sim_minute)

        tick = self._cluster.tick(
            total_rps=total_rps,
            origin_weights=origin_weights,
            carbon_grid=self._carbon,
            sim_hour=sim_hour,
        )

        self._total_rps_served  += sum(m["rps_served"] for m in tick["region_metrics"].values())
        self._total_rps_dropped += tick["dropped_requests"]

        # generate dense reward signals.
        reward = self._reward_shaper.compute(
            step_cost_usd=tick["step_cost_usd"],
            dropped_requests=tick["dropped_requests"],
            total_rps=max(total_rps, 1),
            step_carbon_kg=tick["step_carbon_kg"],
            provision_event=self._last_provision,
            terminate_event=self._last_terminate,
        )

        self._sim_step   += 1
        self._sim_minute += MINUTES_PER_STEP

        cfg       = self._task_cfg
        max_steps = cfg["max_steps"]

        if self._sim_step >= max_steps:
            self._done = True
            self._termination_reason = "max_steps_reached"

        if self._cluster.total_cost_usd > cfg["budget_usd"]:
            self._done = True
            self._termination_reason = "budget_exceeded"

        if self._task_id == 3:
            if self._cluster.total_carbon_kg > cfg.get("max_carbon_kg", 50.0):
                self._done = True
                self._termination_reason = "carbon_cap_exceeded"

        info.update({
            "sim_step":           self._sim_step,
            "sim_hour":           sim_hour,
            "reward":             reward,
            "termination_reason": self._termination_reason,
            "tick":               tick,
        })

        return self._build_observation(), reward, self._done, info

    def state(self) -> FinOpsState:
        # internal state 

        if not self._ready:
            raise RuntimeError("Episode not started. Call reset() first.")
        cfg = self._task_cfg
        return FinOpsState(
            sim_step=self._sim_step,
            task_id=self._task_id,
            seed=self._seed,
            max_steps=cfg["max_steps"],
            done=self._done,
            termination_reason=self._termination_reason,
            total_cost_usd=round(self._cluster.total_cost_usd, 4),
            total_carbon_kg=round(self._cluster.total_carbon_kg, 6),
            total_rps_served=self._total_rps_served,
            total_rps_dropped=self._total_rps_dropped,
            sla_violations=self._cluster.sla_violations,
            dropped_requests=self._cluster.dropped_requests,
            provision_events=self._cluster.provision_events,
            terminate_events=self._cluster.terminate_events,
            budget_total_usd=cfg["budget_usd"],
            max_carbon_kg=cfg.get("max_carbon_kg"),
            nodes=self._cluster.get_node_list(),
            routing_weights=dict(self._cluster.routing_weights),
        )

    def grade(self) -> float:
        # 0.0 to 1.0

        s = self.state()
        grader_input = {
            "total_cost_usd":           s.total_cost_usd,
            "total_carbon_kg":          s.total_carbon_kg,
            "sla_violations":           s.sla_violations,
            "dropped_requests":         s.dropped_requests,
            "terminate_events":         s.terminate_events,
            "provision_events":         s.provision_events,
            "total_rps_served":         s.total_rps_served,
            "total_rps_dropped":        s.total_rps_dropped,
            "budget_total_usd":         s.budget_total_usd,
            "max_steps":                s.max_steps,
            "max_carbon_kg":            s.max_carbon_kg,
            "heuristic_cost_reference": self._heuristic_cost_reference,
            "initial_cost_reference":   self._initial_cost_reference,
        }
        return GRADERS[self._task_id](grader_input)

    def _build_observation(self) -> FinOpsObservation:
        cfg      = self._task_cfg
        sim_hour = (self._sim_minute // 60) % 24
        max_steps = cfg["max_steps"]

        regional_metrics = {}
        for region in REGIONS:
            active = sum(1 for n in self._cluster.nodes.values() if n.active and n.region == region)
            regional_metrics[region] = RegionMetrics(active_nodes=active)

        return FinOpsObservation(
            sim_step=self._sim_step,
            sim_hour=sim_hour,
            sim_minute=self._sim_minute,
            global_rps=cfg["base_rps"],
            regional_metrics=regional_metrics,
            spot_prices=self._pricing.get_snapshot(),
            carbon_intensity=self._carbon.snapshot(sim_hour),
            routing_weights=dict(self._cluster.routing_weights),
            budget_consumed_usd=round(self._cluster.total_cost_usd, 4),
            budget_total_usd=cfg["budget_usd"],
            active_nodes=self._cluster.get_node_list(),
            task_id=self._task_id,
            max_steps=max_steps,
            steps_remaining=max(0, max_steps - self._sim_step),
        )
