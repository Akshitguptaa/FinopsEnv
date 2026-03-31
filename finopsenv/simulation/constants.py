REGIONS = ["us-east-1", "eu-west-1", "ap-southeast-1"]

INSTANCE_TIERS = ["standard", "high-memory", "compute"]

# On-demand hourly price per region and tier (USD)
HOURLY_COST: dict = {
    "us-east-1":      {"standard": 0.096, "high-memory": 0.192, "compute": 0.384},
    "eu-west-1":      {"standard": 0.112, "high-memory": 0.224, "compute": 0.448},
    "ap-southeast-1": {"standard": 0.128, "high-memory": 0.256, "compute": 0.512},
}

SPOT_FRACTION_MEAN = 0.35

# Max RPS a single standard node can serve without saturating
RPS_PER_STANDARD_NODE = 250

MINUTES_PER_STEP = 5

EGRESS_COST_PER_GB = 0.09
GB_PER_1000_RPS    = 0.036

INTRA_REGION_LATENCY_MS      = 12.0
CROSS_REGION_LATENCY_PENALTY = 80.0
SLA_MAX_LATENCY_MS           = 200.0

CPU_CRITICAL_THRESHOLD = 0.90

W_EFFICIENCY = 1.00
W_SLA        = 5.00
W_CARBON     = 0.40
W_CHURN      = 0.15

# Carbon intensity baseline per region (gCO2/kWh) — based on real grid mix data
CARBON_BASELINE: dict = {
    "us-east-1":      210,
    "eu-west-1":      90,
    "ap-southeast-1": 380,
}

# How much the intensity drops at solar peak (midday)
CARBON_AMPLITUDE: dict = {
    "us-east-1":      60,
    "eu-west-1":      40,
    "ap-southeast-1": 80,
}

CARBON_SOLAR_PEAK_HOUR = 13

SERVER_POWER_KW = 0.25

TRAFFIC_DIURNAL_AMPLITUDE = 0.55
TRAFFIC_DIURNAL_PEAK_HOUR = 14
TRAFFIC_BURST_PROBABILITY = 0.05
TRAFFIC_BURST_MULTIPLIER  = 2.5

OU_THETA = 0.15
OU_MU    = 0.35
OU_SIGMA = 0.04

TASK_CONFIGS: dict = {
    1: {
        "name":          "Static Workload Rightsizing",
        "max_steps":     50,
        "base_rps":      1000,
        "traffic_mode":  "constant",
        "enable_spot":   False,
        "enable_carbon": False,
        "regions":       ["us-east-1"],
        "initial_nodes": {"us-east-1": 12},
        "budget_usd":    50.0,
        "max_carbon_kg": None,
        "failure_enabled": False,
    },
    2: {
        "name":          "Dynamic Spot-Market Orchestration",
        "max_steps":     288,
        "base_rps":      2000,
        "traffic_mode":  "spiky",
        "enable_spot":   True,
        "enable_carbon": False,
        "regions":       ["us-east-1", "eu-west-1"],
        "initial_nodes": {"us-east-1": 8, "eu-west-1": 0},
        "budget_usd":    500.0,
        "max_carbon_kg": None,
        "failure_enabled": True,
    },
    3: {
        "name":          "Carbon-Aware Multi-Region Failover",
        "max_steps":     288,
        "base_rps":      3000,
        "traffic_mode":  "diurnal",
        "enable_spot":   True,
        "enable_carbon": True,
        "regions":       REGIONS,
        "initial_nodes": {"us-east-1": 6, "eu-west-1": 4, "ap-southeast-1": 2},
        "budget_usd":    800.0,
        "max_carbon_kg": 50.0,
        "failure_enabled": True,
    },
}
