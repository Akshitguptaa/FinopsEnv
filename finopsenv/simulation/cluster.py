from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .constants import (
    REGIONS, INSTANCE_TIERS, RPS_PER_STANDARD_NODE,
    CPU_CRITICAL_THRESHOLD, MINUTES_PER_STEP,
    EGRESS_COST_PER_GB, GB_PER_1000_RPS,
    INTRA_REGION_LATENCY_MS,
)


@dataclass
class Node:
    node_id:      str
    region:       str
    tier:         str
    billing:      str
    capacity:     float
    active:       bool  = True
    eviction_prob: float = 0.0


class ClusterSimulator:
    # all active compute instances across global regions,
    # accruing exact costs, computing available RPS capacity, and registering 
    # spot instance evictions when enabled

    NODE_CAPACITY = {
        "standard":    1.0,
        "high-memory": 2.0,
        "compute":     4.0,
    }
    SPOT_EVICTION_PROB = 0.004

    def __init__(
        self,
        rng: np.random.Generator,
        pricing_engine,
        initial_nodes: Dict[str, int],
        failure_enabled: bool = True,
        enable_egress: bool = False,
    ):
        self.rng = rng
        self.pricing = pricing_engine
        self.failure_enabled = failure_enabled
        self.enable_egress = enable_egress

        self.nodes: Dict[str, Node] = {}
        self._pending_boot: List[tuple[int, Node]] = []
        self._step = 0

        for region, count in initial_nodes.items():
            for _ in range(count):
                self._create_node(region, "standard", "on-demand")

        self.total_cost_usd:  float = 0.0
        self.total_carbon_kg: float = 0.0
        self.sla_violations:  int   = 0
        self.dropped_requests: int  = 0
        self.provision_events: int  = 0
        self.terminate_events: int  = 0

        # Weight traffic proportionally to where nodes actually exist at startup.
        # Sending traffic to empty regions from step 0 would cause instant SLA failures.
        populated_count = sum(initial_nodes.get(r, 0) for r in REGIONS)
        if populated_count > 0:
            self.routing_weights: Dict[str, float] = {
                r: initial_nodes.get(r, 0) / populated_count for r in REGIONS
            }
        else:
            self.routing_weights: Dict[str, float] = {r: 1.0 / len(REGIONS) for r in REGIONS}

    def _create_node(self, region: str, tier: str, billing: str) -> Node:
        nid = str(uuid.uuid4())[:8]
        cap = self.NODE_CAPACITY.get(tier, 1.0)
        eviction = self.SPOT_EVICTION_PROB if billing == "spot" else 0.0
        node = Node(node_id=nid, region=region, tier=tier,
                    billing=billing, capacity=cap, eviction_prob=eviction)
        self.nodes[nid] = node
        return node

    def _active_capacity(self, region: str) -> float:
        return sum(n.capacity for n in self.nodes.values() if n.active and n.region == region)

    def provision_node(self, region: str, tier: str = "standard", billing: str = "on-demand") -> Optional[str]:
        if region not in REGIONS:
            return None
        if tier not in INSTANCE_TIERS:
            tier = "standard"
        if billing not in ("on-demand", "spot"):
            billing = "on-demand"
        node = self._create_node(region, tier, billing)
        node.active = False
        self._pending_boot.append((self._step, node))
        self.provision_events += 1
        return node.node_id

    def terminate_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if node is None or not node.active:
            return False
        node.active = False
        self.terminate_events += 1
        return True

    def migrate_traffic(self, source: str, target: str, percentage: float) -> bool:
        if source not in REGIONS or target not in REGIONS or source == target:
            return False
        percentage = float(np.clip(percentage, 0.0, 1.0))
        shift = self.routing_weights[source] * percentage
        self.routing_weights[source] -= shift
        self.routing_weights[target] += shift
        total = sum(self.routing_weights.values())
        for r in REGIONS:
            self.routing_weights[r] /= total
        return True

    def tick(
        self,
        total_rps: int,
        origin_weights: Dict[str, float],
        carbon_grid,
        sim_hour: int,
    ) -> Dict:        
        # Logic Flow:
        # 1. Activates pending nodes that have finished their boot delay
        # 2. Randomly evicts Spot instances if failure mechanics are enabled
        # 3. Distributes incoming traffic (RPS) based on the agent's current routing weights
        # 4. Accrues CPU utilization, calculating dropped traffic and latency penalties
        # 5. Computes total financial cost and carbon footprint for this timestep
        
        # Activate nodes whose boot delay has elapsed
        remaining = []
        for ready, node in self._pending_boot:
            if self._step >= ready:
                node.active = True
            else:
                remaining.append((ready, node))
        self._pending_boot = remaining

        # Spot evictions — random hardware failures on spot instances
        if self.failure_enabled:
            for node in list(self.nodes.values()):
                if node.active and node.billing == "spot" and self.rng.random() < node.eviction_prob:
                    node.active = False
                    self.terminate_events += 1

        per_region_rps: Dict[str, float] = {
            region: total_rps * self.routing_weights[region]
            for region in REGIONS
        }

        step_dropped = 0
        region_metrics: Dict[str, Dict] = {}
        for region in REGIONS:
            cap     = self._active_capacity(region)
            max_rps = cap * RPS_PER_STANDARD_NODE
            rps_in  = per_region_rps[region]
            overflow = max(0.0, rps_in - max_rps)
            step_dropped += int(overflow)

            cpu = min(1.0, rps_in / max_rps) if max_rps > 0 else 1.0
            queue_depth = max(0, int(overflow * (MINUTES_PER_STEP * 60)))
            avg_latency = INTRA_REGION_LATENCY_MS * (1 + max(0, cpu - 0.6) * 4)

            region_metrics[region] = {
                "active_nodes":    sum(1 for n in self.nodes.values() if n.active and n.region == region),
                "cpu_utilisation": round(cpu, 4),
                "rps_served":      int(rps_in - overflow),
                "rps_dropped":     int(overflow),
                "queue_depth":     queue_depth,
                "avg_latency_ms":  round(avg_latency, 1),
            }

            if cpu > CPU_CRITICAL_THRESHOLD:
                self.sla_violations += 1

        self.dropped_requests += step_dropped

        step_cost   = 0.0
        step_carbon = 0.0
        for node in self.nodes.values():
            if not node.active:
                continue
            hourly = self.pricing.hourly_cost(node.region, node.tier, node.billing)
            step_cost   += hourly * (MINUTES_PER_STEP / 60.0)
            ci           = carbon_grid.intensity(node.region, sim_hour)
            step_carbon += carbon_grid.co2_kg_per_node_hour(ci) * (MINUTES_PER_STEP / 60.0)

        if self.enable_egress:
            for src in REGIONS:
                for dst in REGIONS:
                    if src == dst:
                        continue
                    cross_rps = per_region_rps[dst] * origin_weights.get(src, 0)
                    gb = cross_rps / 1000.0 * GB_PER_1000_RPS * MINUTES_PER_STEP
                    step_cost += gb * EGRESS_COST_PER_GB

        self.total_cost_usd  += step_cost
        self.total_carbon_kg += step_carbon
        self._step += 1

        return {
            "region_metrics":   region_metrics,
            "step_cost_usd":    round(step_cost, 4),
            "step_carbon_kg":   round(step_carbon, 6),
            "dropped_requests": step_dropped,
        }

    def get_node_list(self) -> List[Dict]:
        return [
            {
                "node_id": n.node_id,
                "region":  n.region,
                "tier":    n.tier,
                "billing": n.billing,
                "active":  n.active,
            }
            for n in self.nodes.values()
        ]
