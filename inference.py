from __future__ import annotations
import json
import os
import sys
import time
from typing import Dict, Optional, List

import requests
from openai import OpenAI

class Config:
    def __init__(self):
        self._load_env()
        self.ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860").rstrip("/")
        self.MAX_EPISODE_SECONDS = 1800
        self.API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        self.HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        
        if not self.HF_TOKEN:
            print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
            sys.exit(1)
            
        self.SUCCESS_SCORE_THRESHOLD = 0.1
        self.MAX_CONVERSATION_WINDOW = 8

    def _load_env(self):
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class Logger:
    @staticmethod
    def log_start(task_id: int, model_name: str):
        task_name = f"Task_{task_id}"
        print(f"[START] task={task_name} env=finops-env model={model_name}", flush=True)

    @staticmethod
    def log_step(step: int, action: dict, reward: float, done: bool, error: Optional[str] = None):
        action_str = json.dumps(action).replace(" ", "")
        err_str = error if error else "null"
        done_str = "true" if done else "false"
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

    @staticmethod
    def log_end(success: bool, steps: int, score: float, rewards: List[float]):
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


class FinOpsEnvClient:
    def __init__(self, config: Config):
        self.config = config

    def check_health(self):
        try:
            requests.get(f"{self.config.ENV_SERVER_URL}/health", timeout=10).raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Cannot reach server at {self.config.ENV_SERVER_URL}: {e}")

    def reset(self, task_id: int, seed: int = 42) -> Dict:
        r = requests.post(
            f"{self.config.ENV_SERVER_URL}/reset",
            json={"seed": seed, "task_id": task_id, "session_id": "inference"},
        )
        r.raise_for_status()
        return r.json()["observation"]

    def step(self, action: Dict) -> Dict:
        r = requests.post(
            f"{self.config.ENV_SERVER_URL}/step",
            json={"action": action, "session_id": "inference"},
        )
        r.raise_for_status()
        return r.json()

    def get_grade(self) -> float:
        try:
            r = requests.get(f"{self.config.ENV_SERVER_URL}/grade", params={"session_id": "inference"})
            r.raise_for_status()
            return r.json()["score"]
        except requests.exceptions.HTTPError as e:
            print(f"  [Grade Error] {e}", file=sys.stderr)
            return 0.001


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.HF_TOKEN, base_url=config.API_BASE_URL)
        self.resp_format = {"type": "json_object"}

    def call(self, obs_text: str, conversation: List[Dict], system_prompt: str) -> Dict:
        conversation.append({"role": "user", "content": obs_text})
        
        if self.config.MAX_CONVERSATION_WINDOW > 0:
            trimmed = conversation[-self.config.MAX_CONVERSATION_WINDOW:]
        else:
            trimmed = [{"role": "user", "content": obs_text}]
            
        for attempt in range(8):
            try:
                if self.config.API_BASE_URL and "googleapis" in self.config.API_BASE_URL:
                    time.sleep(4)
                    
                messages = [{"role": "system", "content": system_prompt}] + trimmed

                response = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    response_format=self.resp_format,
                    max_tokens=200,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content
                
                if not raw or not raw.strip():
                    print(f"  [LLM] Empty response, retrying...", file=sys.stderr)
                    continue
                
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    cleaned = "\n".join(lines).strip()
                
                conversation.append({"role": "assistant", "content": cleaned})
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"  [LLM] JSON parse failed: {e} | raw={repr(raw[:100] if raw else 'None')}", file=sys.stderr)
                continue
            except Exception as e:
                err_str = str(e).lower()
                if "insufficient_quota" in err_str:
                    raise RuntimeError(f"API Billing Error: You have run out of credits or hit a hard quota. Message: {e}")
                if any(k in err_str for k in ["rate_limit", "429", "too many requests", "quota"]):
                    wait_time = 15 + (15 * attempt)
                    print(f"  [LLM Rate Limit] Sleeping {wait_time}s to reset quota block...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Failed to get LLM response after 8 retries.")


class PromptFormatter:
    ACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "action_type":   {"type": "string", "enum": ["provision_node", "terminate_node", "migrate_traffic", "noop"]},
            "region":        {"type": "string", "enum": ["us-east-1", "eu-west-1", "ap-southeast-1"]},
            "instance_tier": {"type": "string", "enum": ["standard", "high-memory", "compute"]},
            "billing_model": {"type": "string", "enum": ["on-demand", "spot"]},
            "node_id":       {"type": "string"},
            "source_region": {"type": "string", "enum": ["us-east-1", "eu-west-1", "ap-southeast-1"]},
            "target_region": {"type": "string", "enum": ["us-east-1", "eu-west-1", "ap-southeast-1"]},
            "percentage":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["action_type"],
        "additionalProperties": False,
    }

    BASE_PROMPT = f"""\
You are an autonomous SRE and FinOps controller managing a global cloud cluster.
Your goal is to minimize cost while keeping SLAs. You are REWARDED for terminating excess nodes.

CAPACITY MATH (use the CAPACITY CHECK line in each observation):
- Each "standard" node handles ~250 RPS, "compute" ~400 RPS, "high-memory" ~200 RPS.
- BEFORE terminating, verify: remaining_capacity > current_RPS * 1.5 (50% safety margin).
- NEVER terminate if only 2 or fewer nodes remain in a region with traffic.

OPTIMIZATION STRATEGY (follow this every step):
- If headroom > 70%: You are OVER-PROVISIONED. You MUST output action_type "terminate_node". Prioritize terminating "on-demand" nodes!
- If headroom < 40%: Cluster is tight. MUST output action_type "provision_node".
- If headroom 40-70%: Cluster is safe, so optimize CARBON and COST:
  * Check CARBON INTENSITY. If one region is 30+ gCO2/kWh greener, output "migrate_traffic" to shift 0.10 (10%) from dirtiest to greenest region.
  * If carbon is equal, but spot prices vary greatly, use "migrate_traffic" to shift to the cheapest region.
  * Only output "noop" if capacity is safe AND carbon/costs are balanced.
- Dropped requests are CATASTROPHIC (quadratic penalty). Never risk drops.

AVAILABLE ACTIONS:
1. provision_node  — Boot a new node. Include region, instance_tier, and billing_model (always prefer "spot" billing_model to reduce costs!).
2. terminate_node  — Remove a node. MUST include region, instance_tier, billing_model, and node_id.
3. migrate_traffic — Shift traffic % between regions for cost/carbon optimization.
4. noop            — Use when cluster is right-sized or stable.

DECISION PRIORITY: Avoid drops > Reduce cost > Reduce carbon.

Respond ONLY with valid JSON. No explanation text.
SCHEMA:
{json.dumps(ACTION_SCHEMA, indent=2)}
"""

    TASK_STRATEGY = {
        1: (
            "TASK 1 — STATIC RIGHTSIZING:\n"
            "The cluster is OVER-PROVISIONED. 1000 RPS needs only 5 standard nodes.\n"
            "TERMINATE one on-demand node EVERY step until exactly 5 remain.\n"
            "NEVER provision new nodes. Pick the node_id from the ACTIVE NODES list."
        ),
        2: (
            "TASK 2 — SPOT OPTIMIZATION:\n"
            "Replace on-demand with spot nodes to cut costs ~65%. Steps:\n"
            "1. If headroom < 35%, provision a spot node first for buffer.\n"
            "2. If headroom > 35% and on-demand nodes exist, terminate one on-demand.\n"
            "3. On spot eviction (node count dropped), immediately provision spot.\n"
            "4. During spike bursts, provision extra spot nodes.\n"
            "Always use billing_model='spot'."
        ),
        3: (
            "TASK 3 — CARBON-AWARE ROUTING:\n"
            "Minimize carbon + cost + drops simultaneously.\n"
            "1. eu-west-1 is greenest (baseline ~90 gCO2). Route traffic there.\n"
            "2. ap-southeast-1 is dirtiest (~380 gCO2). Minimize its traffic.\n"
            "3. Use migrate_traffic to shift 15-20% from dirty to green each step.\n"
            "4. Replace on-demand with spot everywhere.\n"
            "5. Provision in eu-west-1 to absorb shifted traffic."
        ),
    }

    @classmethod
    def get_system_prompt(cls, task_id: int) -> str:
        return cls.BASE_PROMPT + "\n\n" + cls.TASK_STRATEGY.get(task_id, "")

    @staticmethod
    def _capacity_summary(obs: Dict) -> str:
        tier_rps = {"standard": 250, "compute": 400, "high-memory": 200}
        summaries = []
        for region, m in obs.get("regional_metrics", {}).items():
            nodes = [n for n in obs.get("active_nodes", []) if n.get("active") and n["region"] == region]
            total_cap = sum(tier_rps.get(n["tier"], 250) for n in nodes)
            rps_in = m.get("rps_incoming", 0) or obs.get("global_rps", 0) * obs.get("routing_weights", {}).get(region, 0)
            headroom = ((total_cap - rps_in) / total_cap * 100) if total_cap > 0 else 0
            summaries.append(f"{region}: {len(nodes)} nodes, cap={total_cap} RPS, headroom={headroom:.0f}%")
        return " | ".join(summaries)

    @classmethod
    def format_observation(cls, obs: Dict, recent_rewards: List[float]) -> str:
        lines = [
            f"Step {obs['sim_step']}/{obs['max_steps']} | Hour {obs['sim_hour']:02d}:00 "
            f"| Global RPS: {obs['global_rps']} "
            f"| Budget: ${obs['budget_consumed_usd']:.2f}/${obs['budget_total_usd']:.0f}",
        ]

        if len(recent_rewards) >= 2:
            last5 = recent_rewards[-5:]
            trend = "improving" if last5[-1] > last5[0] else "declining" if last5[-1] < last5[0] else "stable"
            lines.append(f"REWARD TREND ({trend}): last 5 rewards = [{', '.join(f'{r:.2f}' for r in last5)}]")

        lines.append(f"CAPACITY CHECK: {cls._capacity_summary(obs)}")

        lines.append("REGIONAL METRICS:")
        for region, m in obs.get("regional_metrics", {}).items():
            lines.append(
                f"  {region}: {m['active_nodes']} nodes | CPU {m['cpu_utilisation']:.0%} "
                f"| Dropped {m['rps_dropped']} RPS | Latency {m['avg_latency_ms']:.0f}ms"
            )

        if obs.get("spot_prices"):
            lines.append("SPOT PRICES (fraction of on-demand):")
            for region, tiers in obs["spot_prices"].items():
                lines.append("  " + region + ": " + " | ".join(f"{t}={v:.2f}" for t, v in tiers.items()))

        if obs.get("task_id") == 3:
            lines.append("CARBON INTENSITY (gCO2/kWh):")
            for region, ci in obs.get("carbon_intensity", {}).items():
                lines.append(f"  {region}: {ci:.1f}")

        lines.append("ROUTING WEIGHTS:")
        for region, w in obs.get("routing_weights", {}).items():
            lines.append(f"  {region}: {w:.2%}")

        active = [n for n in obs.get("active_nodes", []) if n.get("active")]
        if active:
            lines.append("ACTIVE NODES (first 10):")
            for n in active[:10]:
                lines.append(f"  {n['node_id']} | {n['region']} | {n['tier']} | {n['billing']}")

        return "\n".join(lines)


class HeuristicController:
    TIER_RPS = {"standard": 250, "compute": 400, "high-memory": 200}
    
    def __init__(self):
        self.prev_node_count: Dict[int, int] = {}

    def is_safe_to_terminate(self, obs: Dict, action: Dict) -> bool:
        if action.get("action_type") != "terminate_node":
            return True

        target_region = action.get("region", "")
        target_node_id = action.get("node_id", "")
        active_nodes = [n for n in obs.get("active_nodes", []) if n.get("active") and n["region"] == target_region]

        if len(active_nodes) <= 3:
            return False

        remaining_cap = sum(self.TIER_RPS.get(n["tier"], 250) for n in active_nodes if n["node_id"] != target_node_id)
        rw = obs.get("routing_weights", {}).get(target_region, 0)
        region_rps = obs.get("global_rps", 0) * rw

        return remaining_cap >= region_rps * 1.25

    def get_heuristic_action(self, obs: Dict, task_id: int) -> Optional[Dict]:
        active = [n for n in obs.get("active_nodes", []) if n.get("active")]
        
        if task_id == 1:
            return self._heuristic_task1(obs, active)
        elif task_id == 2:
            return self._heuristic_task2(obs, active)
        elif task_id == 3:
            return self._heuristic_task3(obs, active)
            
        return {"action_type": "noop"}

    def _heuristic_task1(self, obs: Dict, active: list) -> Optional[Dict]:
        region = "us-east-1"
        region_nodes = [n for n in active if n["region"] == region]

        if len(region_nodes) <= 5:
            return {"action_type": "noop"}

        return None

    def _heuristic_task2(self, obs: Dict, active: list) -> Optional[Dict]:
        current_count = len(active)
        prev_count = self.prev_node_count.get(2, current_count)
        eviction_detected = current_count < prev_count
        self.prev_node_count[2] = current_count

        total_cap = sum(self.TIER_RPS.get(n["tier"], 250) for n in active)
        global_rps = obs.get("global_rps", 0)
        headroom = (total_cap - global_rps) / total_cap if total_cap > 0 else 0
        on_demand = [n for n in active if n["billing"] == "on-demand"]

        if eviction_detected or headroom < 0.15:
            weights = obs.get("routing_weights", {})
            target_region = max(weights, key=weights.get) if weights else "us-east-1"
            return {
                "action_type": "provision_node",
                "region": target_region,
                "instance_tier": "standard",
                "billing_model": "spot",
            }

        if on_demand:
            if headroom > 0.35:
                target = on_demand[0]
                return {
                    "action_type": "terminate_node",
                    "region": target["region"],
                    "instance_tier": target["tier"],
                    "billing_model": target["billing"],
                    "node_id": target["node_id"],
                }
            else:
                return {
                    "action_type": "provision_node",
                    "region": on_demand[0]["region"],
                    "instance_tier": "standard",
                    "billing_model": "spot",
                }

        if headroom > 0.50 and len(active) > 8:
            target = active[-1]
            return {
                "action_type": "terminate_node",
                "region": target["region"],
                "instance_tier": target["tier"],
                "billing_model": target["billing"],
                "node_id": target["node_id"],
            }

        return None

    def _heuristic_task3(self, obs: Dict, active: list) -> Optional[Dict]:
        current_count = len(active)
        prev_count = self.prev_node_count.get(3, current_count)
        eviction_detected = current_count < prev_count
        self.prev_node_count[3] = current_count

        total_cap = sum(self.TIER_RPS.get(n["tier"], 250) for n in active)
        global_rps = obs.get("global_rps", 0)
        headroom = (total_cap - global_rps) / total_cap if total_cap > 0 else 0

        if eviction_detected or headroom < 0.15:
            return {
                "action_type": "provision_node",
                "region": "eu-west-1",
                "instance_tier": "standard",
                "billing_model": "spot",
            }

        carbon = obs.get("carbon_intensity", {})
        weights = obs.get("routing_weights", {})
        if carbon and weights:
            regions_by_carbon = sorted(carbon.items(), key=lambda x: x[1])
            greenest = regions_by_carbon[0][0]
            dirtiest = regions_by_carbon[-1][0]
            dirty_weight = weights.get(dirtiest, 0)

            if dirty_weight > 0.10:
                green_nodes = [n for n in active if n["region"] == greenest]
                green_cap = sum(self.TIER_RPS.get(n["tier"], 250) for n in green_nodes)
                green_rps = global_rps * weights.get(greenest, 0)
                shift_amount = global_rps * dirty_weight * 0.20

                if green_cap > (green_rps + shift_amount) * 1.2:
                    return {
                        "action_type": "migrate_traffic",
                        "source_region": dirtiest,
                        "target_region": greenest,
                        "percentage": 0.20,
                    }
                else:
                    return {
                        "action_type": "provision_node",
                        "region": greenest,
                        "instance_tier": "standard",
                        "billing_model": "spot",
                    }

        on_demand = [n for n in active if n["billing"] == "on-demand"]
        if on_demand and headroom > 0.35:
            target = on_demand[0]
            return {
                "action_type": "terminate_node",
                "region": target["region"],
                "instance_tier": target["tier"],
                "billing_model": target["billing"],
                "node_id": target["node_id"],
            }

        return None

    def get_llm_fallback_action(self, obs: Dict) -> Dict:
        active = [n for n in obs.get("active_nodes", []) if n.get("active")]
        total_cap = sum(self.TIER_RPS.get(n["tier"], 250) for n in active)
        global_rps = obs.get("global_rps", 0)
        headroom = (total_cap - global_rps) / total_cap if total_cap > 0 else 0

        if headroom < 0.20:
            weights = obs.get("routing_weights", {})
            target_region = max(weights, key=weights.get) if weights else "us-east-1"
            return {
                "action_type": "provision_node",
                "region": target_region,
                "instance_tier": "standard",
                "billing_model": "spot",
            }

        if headroom > 0.60 and len(active) > 5:
            target = active[-1]
            return {
                "action_type": "terminate_node",
                "region": target["region"],
                "instance_tier": target["tier"],
                "billing_model": target["billing"],
                "node_id": target["node_id"],
            }

        return {"action_type": "noop"}


class InferenceAgent:
    def __init__(self):
        self.config = Config()
        self.env = FinOpsEnvClient(self.config)
        self.llm = LLMClient(self.config)
        self.controller = HeuristicController()
        
    def run_all(self):
        try:
            self.env.check_health()
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
            
        scores = {}
        for task_id in [1, 2, 3]:
            scores[task_id] = self._run_task(task_id)
            
        return scores
        
    def _run_task(self, task_id: int, seed: int = 42) -> float:
        self.controller.prev_node_count = {}
        recent_rewards: List[float] = []
        conversation: List[Dict] = []
        
        system_prompt = PromptFormatter.get_system_prompt(task_id)
        
        Logger.log_start(task_id, self.config.MODEL_NAME)
        obs = self.env.reset(task_id=task_id, seed=seed)
        
        step_count = 0
        start = time.time()
        rewards: List[float] = []
        
        success = False
        score_val = 0.001
        
        try:
            while True:
                if time.time() - start > self.config.MAX_EPISODE_SECONDS:
                    break
                    
                action = self.controller.get_heuristic_action(obs, task_id)
                if action is None:
                    obs_prompt = PromptFormatter.format_observation(obs, recent_rewards)
                    try:
                        action = self.llm.call(obs_prompt, conversation, system_prompt)
                    except Exception as e:
                        print(f"  [LLM Error] {type(e).__name__}: {e}", file=sys.stderr)
                        action = self.controller.get_llm_fallback_action(obs)
                        
                if not self.controller.is_safe_to_terminate(obs, action):
                    action = {"action_type": "noop"}
                    
                error_str = None
                try:
                    result = self.env.step(action)
                    obs = result["observation"]
                    done = result["done"]
                    reward = result.get("reward", 0.0)
                    rewards.append(reward)
                    recent_rewards.append(reward)
                except Exception as e:
                    error_str = str(e)
                    done = True
                    reward = 0.0
                    rewards.append(reward)
                    
                step_count += 1
                Logger.log_step(step_count, action, reward, done, error_str)
                
                if done:
                    break
                    
            raw_score = self.env.get_grade()
            success = raw_score >= self.config.SUCCESS_SCORE_THRESHOLD
            score_val = raw_score
            
        finally:
            score_val = max(0.001, min(0.999, score_val))
            Logger.log_end(success, step_count, score_val, rewards)
            
        return score_val

def main():
    agent = InferenceAgent()
    agent.run_all()

if __name__ == "__main__":
    main()
