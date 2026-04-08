---
title: FinOpsEnv
emoji: 🍵
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# FinOpsEnv: Carbon-Aware Multi-Cloud Orchestration

FinOpsEnv is a high-fidelity simulation environment designed to train autonomous SRE (Site Reliability Engineering) agents in cost optimization and carbon-aware workload management. It simulates the complexities of real-world infrastructure management, including fluctuating spot market prices, regional carbon intensity variations, and strict Service Level Agreements (SLAs).

## Motivation
As cloud infrastructure grows, the dual challenge of managing spiraling costs (FinOps) and reducing environmental impact (GreenOps) has become a critical human task. FinOpsEnv provides a standardized OpenAI-compatible interface for agents to learn optimal provisioning and traffic-shaping strategies that balance performance, cost, and sustainability.

## Mathematical Foundations

### 1. Step Reward Function
The environment provides a continuous reward signal $R_t$ at each step, calculated as a weighted sum of efficiency, performance, sustainability, and stability components:

$$
R_t = (w_{e} \cdot r_{\text{eff}}) - (w_{s} \cdot r_{\text{SLA}}) + (w_{c} \cdot r_{\text{carbon}}) - (w_{h} \cdot r_{\text{churn}})
$$

**Weights:** $w_{e}=1.0$, $w_{s}=5.0$, $w_{c}=0.4$ (Task 3 only), $w_{h}=0.15$.

**Efficiency ($r_{\text{eff}}$):** Savings relative to the initial step baseline.

$$
r_{\text{eff}} = \frac{\text{Cost}_{\text{baseline}} - \text{Cost}_t}{\text{Cost}_{\text{baseline}}}
$$

**SLA Penalty ($r_{\text{SLA}}$):** A quadratic-scaled penalty based on the fraction of dropped requests ($f_{\text{drop}}$).

$$
r_{\text{SLA}} = f_{\text{drop}} \cdot (1 + 3 \cdot f_{\text{drop}})
$$

**Carbon Reward ($r_{\text{carbon}}$):** Encourages minimizing the carbon footprint per request served.

$$
r_{\text{carbon}} = \exp\left(-5000 \cdot \frac{\text{Carbon}_t}{\text{Requests}_{\text{total}}}\right)
$$

**Churn Penalty ($r_{\text{churn}}$):** A binary penalty ($1.0$) applied if any `provision_node` or `terminate_node` action is executed during the step.

### 2. Simulation Mechanics

**CPU Utilization ($\rho$):** The ratio of incoming RPS to the total active node capacity in a region.

$$
\rho = \min\left(1.0, \frac{\text{RPS}_{\text{incoming}}}{\sum (\text{Node}_{\text{capacity}} \cdot \text{RPS}_{\text{unit}})}\right)
$$

**Latency ($L$):** Base intra-region latency ($12\text{ms}$) increases linearly by $400\%$ once $\rho$ exceeds the $60\%$ threshold.

$$
L = L_{\text{base}} \cdot (1 + 4 \cdot \max(0, \rho - 0.6))
$$

**Operational Costs:** Total cost includes hourly instance rates (On-Demand or Spot) and egress costs ($0.09\text{/GB}$) if cross-region traffic migration is enabled.

### 3. Task Grading
Final scores $S \in [0, 1]$ are determined by specific constraints for each task:

**Task 1 (Rightsizing):** Score is based on savings relative to an optimal 5-node fleet. $S=0$ if any requests are dropped or if the agent never terminates a node.

$$
S = \max\left(0.1, \frac{\text{Cost}_{\text{initial}} - \text{Cost}_{\text{agent}}}{\text{Cost}_{\text{initial}} - \text{Cost}_{\text{optimal}}}\right)
$$

**Task 2 (Spot Optimization):** Score measures improvement over a naive heuristic baseline. $S=0$ if the total drop rate exceeds $0.1\%$.

$$
S = \frac{\text{Cost}_{\text{heuristic}} - \text{Cost}_{\text{agent}}}{\text{Cost}_{\text{heuristic}}}
$$

**Task 3 (Global Orchestration):** A weighted harmonic mean of three sub-scores: Cost ($s_1$), Carbon ($s_2$), and SLA ($s_3$), where $s_{\text{sla}} = 1 - 1000 \cdot f_{\text{drop}}$. $S=0$ if the drop rate exceeds $0.1\%$ or if budgets are exceeded.

$$
S = \frac{\sum w_i}{\sum \frac{w_i}{s_i}} \text{ with weights } [0.40, 0.35, 0.25]
$$

## Action Space
The environment uses a `discrete-parametric` action space:
* **provision_node**: Requires `region`, `instance_tier`, and `billing_model`.
* **terminate_node**: Shuts down a specific node via `node_id`.
* **migrate_traffic**: Shifts traffic `percentage` between `source_region` and `target_region`.
* **noop**: No action taken.

## Setup and Usage

### Prerequisites
* [uv](https://docs.astral.sh/uv/)
* Docker

### Local Development
1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment variables:**
   Add your API keys (`HF_TOKEN`, `API_BASE_URL`).
   ```bash
   cp .env.example .env
   ```

3. **Start the environment server:**
   ```bash
   uv run server
   ```

### Evaluation
To run the baseline agent and verify scores across tasks:
```bash
uv run python inference.py
```

## License
MIT License. See [LICENSE](LICENSE) for more information.
