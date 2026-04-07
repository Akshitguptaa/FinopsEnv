"""FinOpsEnv client — the user-facing Python SDK for interacting with the environment.

Users import this client and connect to a running FinOpsEnv server:

    from finopsenv.client import FinOpsEnvClient

    with FinOpsEnvClient(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset(seed=42, task_id=1)
        result = env.step(FinOpsAction(action_type="noop"))
        print(result.observation.budget_consumed_usd)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .schemas.action import FinOpsAction
from .schemas.observation import FinOpsObservation, FinOpsState, RegionMetrics


class FinOpsEnvClient(EnvClient[FinOpsAction, FinOpsObservation, FinOpsState]):
    """WebSocket client for the FinOpsEnv environment."""

    def _step_payload(self, action: FinOpsAction) -> dict:
        """Serialize a FinOpsAction into the dict sent over the wire."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FinOpsObservation]:
        """Parse a step/reset response into a typed StepResult."""
        obs_data = payload.get("observation", payload)

        # Build RegionMetrics from nested dicts
        raw_metrics = obs_data.get("regional_metrics", {})
        regional_metrics = {}
        for region, m in raw_metrics.items():
            if isinstance(m, dict):
                regional_metrics[region] = RegionMetrics(**m)
            else:
                regional_metrics[region] = m

        obs = FinOpsObservation(
            sim_step=obs_data.get("sim_step", 0),
            sim_hour=obs_data.get("sim_hour", 0),
            sim_minute=obs_data.get("sim_minute", 0),
            global_rps=obs_data.get("global_rps", 0),
            regional_metrics=regional_metrics,
            spot_prices=obs_data.get("spot_prices", {}),
            carbon_intensity=obs_data.get("carbon_intensity", {}),
            routing_weights=obs_data.get("routing_weights", {}),
            budget_consumed_usd=obs_data.get("budget_consumed_usd", 0.0),
            budget_total_usd=obs_data.get("budget_total_usd", 0.0),
            active_nodes=obs_data.get("active_nodes", []),
            task_id=obs_data.get("task_id", 1),
            max_steps=obs_data.get("max_steps", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FinOpsState:
        """Parse a state response into a typed FinOpsState."""
        return FinOpsState(
            task_id=payload.get("task_id", 1),
            seed=payload.get("seed", 42),
            sim_step=payload.get("sim_step", 0),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 0),
            done=payload.get("done", False),
            termination_reason=payload.get("termination_reason"),
            total_cost_usd=payload.get("total_cost_usd", 0.0),
            total_carbon_kg=payload.get("total_carbon_kg", 0.0),
            total_rps_served=payload.get("total_rps_served", 0),
            total_rps_dropped=payload.get("total_rps_dropped", 0),
            sla_violations=payload.get("sla_violations", 0),
            dropped_requests=payload.get("dropped_requests", 0),
            provision_events=payload.get("provision_events", 0),
            terminate_events=payload.get("terminate_events", 0),
            budget_total_usd=payload.get("budget_total_usd", 0.0),
            max_carbon_kg=payload.get("max_carbon_kg"),
            nodes=payload.get("nodes", []),
            routing_weights=payload.get("routing_weights", {}),
        )
