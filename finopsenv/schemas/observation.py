from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openenv.core.env_server import Observation, State


class RegionMetrics(BaseModel):
    active_nodes:    int   = 0
    cpu_utilisation: float = 0.0
    rps_served:      int   = 0
    rps_dropped:     int   = 0
    queue_depth:     int   = 0
    avg_latency_ms:  float = 0.0


class FinOpsObservation(Observation):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    sim_step:            int                      = 0
    sim_hour:            int                      = 0
    sim_minute:          int                      = 0
    global_rps:          int                      = 0
    regional_metrics:    Dict[str, RegionMetrics]  = Field(default_factory=dict)
    spot_prices:         Dict[str, Dict[str, float]] = Field(default_factory=dict)
    carbon_intensity:    Dict[str, float]          = Field(default_factory=dict)
    routing_weights:     Dict[str, float]          = Field(default_factory=dict)
    budget_consumed_usd: float                    = 0.0
    budget_total_usd:    float                    = 0.0
    active_nodes:        List[Dict[str, Any]]     = Field(default_factory=list)
    task_id:             int                      = 1
    max_steps:           int                      = 0
    steps_remaining:     int                      = 0


class FinOpsState(State):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    task_id:            int
    seed:               int
    sim_step:           int
    max_steps:          int
    done:               bool
    termination_reason: str | None

    total_cost_usd:   float
    total_carbon_kg:  float
    total_rps_served: int
    total_rps_dropped: int
    sla_violations:   int
    dropped_requests: int
    provision_events: int
    terminate_events: int

    budget_total_usd: float
    max_carbon_kg:    float | None

    nodes:           List[Dict[str, Any]] = Field(default_factory=list)
    routing_weights: Dict[str, float]     = Field(default_factory=dict)
