from __future__ import annotations
from typing import Literal, Optional
from pydantic import ConfigDict, Field
from openenv.core.env_server import Action


class FinOpsAction(Action):
    action_type: Literal["provision_node", "terminate_node", "migrate_traffic", "noop"] = Field(
        description="Which action to execute this step."
    )

    region: Optional[str] = Field(
        default=None,
        description="Target region for provisioning (e.g. 'us-east-1').",
    )
    instance_tier: Optional[str] = Field(
        default="standard",
        description="Node tier: 'standard', 'high-memory', or 'compute'.",
    )
    billing_model: Optional[Literal["on-demand", "spot"]] = Field(
        default="on-demand",
        description="Billing model for the new node.",
    )

    node_id: Optional[str] = Field(
        default=None,
        description="8-char node ID to terminate.",
    )

    source_region: Optional[str] = Field(
        default=None,
        description="Region to shift traffic from.",
    )
    target_region: Optional[str] = Field(
        default=None,
        description="Region to shift traffic to.",
    )
    percentage: Optional[float] = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction (0-1) of source region's traffic to migrate.",
    )

    model_config = {"extra": "ignore"}
