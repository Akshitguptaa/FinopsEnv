from .constants import TASK_CONFIGS, REGIONS
from .cluster import ClusterSimulator

import math
import numpy as np

from .constants import (
    TRAFFIC_DIURNAL_AMPLITUDE, TRAFFIC_DIURNAL_PEAK_HOUR,
    TRAFFIC_BURST_MULTIPLIER,
    OU_THETA, OU_MU, OU_SIGMA,
    HOURLY_COST, INSTANCE_TIERS,
    CARBON_BASELINE, CARBON_AMPLITUDE, CARBON_SOLAR_PEAK_HOUR,
    SERVER_POWER_KW,
)


class TrafficGenerator:
    def __init__(self, rng: np.random.Generator, base_rps: int, mode: str = "constant") -> None:
        self.rng      = rng
        self.base_rps = base_rps
        self.mode     = mode
        self.bursts   = []

    def seed_bursts(self, max_steps: int):
        if self.mode == "spiky":
            for _ in range(max_steps // 20):
                start = self.rng.integers(0, max_steps - 5)
                duration = self.rng.integers(2, 6)
                self.bursts.append((start, start + duration))

    def step(self, sim_minute: int) -> tuple[int, dict[str, float]]:
        current_step = sim_minute // 5
        rps = self.base_rps

        if self.mode == "diurnal":
            hour = (sim_minute // 60) % 24
            envelope = 1.0 + TRAFFIC_DIURNAL_AMPLITUDE * math.sin(
                2 * math.pi * (hour - TRAFFIC_DIURNAL_PEAK_HOUR) / 24
            )
            rps = int(self.base_rps * envelope)
        elif self.mode == "spiky":
            for start, end in self.bursts:
                if start <= current_step <= end:
                    rps = int(self.base_rps * TRAFFIC_BURST_MULTIPLIER)
                    break

        hour = (sim_minute // 60) % 24
        if 13 <= hour <= 21:
            origins = {"us-east-1": 0.6, "eu-west-1": 0.2, "ap-southeast-1": 0.2}
        elif 5 <= hour <= 12:
            origins = {"us-east-1": 0.2, "eu-west-1": 0.6, "ap-southeast-1": 0.2}
        else:
            origins = {"us-east-1": 0.2, "eu-west-1": 0.2, "ap-southeast-1": 0.6}

        return rps, origins


class PricingEngine:
    def __init__(self, rng: np.random.Generator, enable_spot: bool) -> None:
        self.rng         = rng
        self.enable_spot = enable_spot
        self._spot_fractions: dict = {
            r: {t: OU_MU for t in INSTANCE_TIERS}
            for r in REGIONS
        }

    def step(self) -> None:
        if not self.enable_spot:
            return
        for region in self._spot_fractions:
            for tier in INSTANCE_TIERS:
                p  = self._spot_fractions[region][tier]
                dp = OU_THETA * (OU_MU - p) + OU_SIGMA * self.rng.standard_normal()
                self._spot_fractions[region][tier] = float(np.clip(p + dp, 0.05, 0.95))

    def spot_fraction(self, region: str, tier: str) -> float:
        if not self.enable_spot:
            return 1.0
        return self._spot_fractions.get(region, {}).get(tier, OU_MU)

    def hourly_cost(self, region: str, tier: str, billing: str) -> float:
        base = HOURLY_COST.get(region, {}).get(tier, 0.096)
        if billing == "spot" and self.enable_spot:
            return base * self.spot_fraction(region, tier)
        return base

    def get_snapshot(self) -> dict:
        if not self.enable_spot:
            return {}
        return {r: dict(tiers) for r, tiers in self._spot_fractions.items()}


class CarbonGrid:
    def __init__(self, rng: np.random.Generator, enable_carbon: bool) -> None:
        self.rng           = rng
        self.enable_carbon = enable_carbon

    def intensity(self, region: str, hour: int) -> float:
        if not self.enable_carbon:
            return 0.0
        base = CARBON_BASELINE.get(region, 200)
        amp  = CARBON_AMPLITUDE.get(region, 50)
        ci   = base - amp * math.cos(2 * math.pi * (hour - CARBON_SOLAR_PEAK_HOUR) / 24) + self.rng.normal(0, 5)
        return float(max(0.0, ci))

    def snapshot(self, hour: int) -> dict:
        if not self.enable_carbon:
            return {r: 0.0 for r in REGIONS}
        return {r: round(self.intensity(r, hour), 1) for r in REGIONS}

    def co2_kg_per_node_hour(self, intensity_g_per_kwh: float) -> float:
        if not self.enable_carbon:
            return 0.0
        return SERVER_POWER_KW * intensity_g_per_kwh / 1000.0


__all__ = [
    "TrafficGenerator",
    "PricingEngine",
    "CarbonGrid",
    "ClusterSimulator",
    "TASK_CONFIGS",
    "REGIONS",
]
