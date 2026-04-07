from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import requests
from openai import OpenAI

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://localhost:7860").rstrip("/")
MAX_EPISODE_SECONDS = 360

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")

if not API_KEY:
    print("ERROR: API_KEY/HF_TOKEN is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

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

SYSTEM_PROMPT = f"""\
You are an autonomous SRE and FinOps controller managing a global cloud cluster.
Your goal is to minimize cost while keeping SLAs. You are REWARDED for terminating excess nodes.

CAPACITY MATH (use the CAPACITY CHECK line in each observation):
- Each "standard" node handles ~250 RPS, "compute" ~400 RPS, "high-memory" ~200 RPS.
- BEFORE terminating, verify: remaining_capacity > current_RPS * 1.3 (30% safety margin).
- NEVER terminate if only 2 or fewer nodes remain in a region with traffic.

OPTIMIZATION STRATEGY (follow this every step):
- If headroom > 50%: You are OVER-PROVISIONED. You MUST output action_type "terminate_node" and include region, instance_tier, billing_model, and node_id.
- If headroom 30-50%: Cluster is right-sized. Use "noop".
- If headroom < 30%: Cluster is tight. Consider "provision_node" if CPU > 75%.
- Once you reach a stable reward (same reward 2+ times in a row), STOP optimizing and use "noop".
- Dropped requests are CATASTROPHIC (quadratic penalty). Never risk drops.

AVAILABLE ACTIONS:
1. provision_node  — Boot a new node. Include region, instance_tier, and billing_model (use "spot" if price < 0.5).
2. terminate_node  — Remove a node. MUST include region, instance_tier, billing_model, and node_id.
3. migrate_traffic — Shift traffic % between regions for cost/carbon optimization.
4. noop            — Use when cluster is right-sized or stable.

DECISION PRIORITY: Avoid drops > Reduce cost > Reduce carbon.

Respond ONLY with valid JSON. No explanation text.
SCHEMA:
{json.dumps(ACTION_SCHEMA, indent=2)}
"""


def reset_env(task_id: int, seed: int = 42) -> Dict:
    r = requests.post(
        f"{ENV_SERVER_URL}/reset",
        json={"seed": seed, "task_id": task_id, "session_id": "inference"},
    )
    r.raise_for_status()
    return r.json()["observation"]


def step_env(action: Dict) -> Dict:
    r = requests.post(
        f"{ENV_SERVER_URL}/step",
        json={"action": action, "session_id": "inference"},
    )
    r.raise_for_status()
    return r.json()


def get_grade() -> float:
    r = requests.get(f"{ENV_SERVER_URL}/grade", params={"session_id": "inference"})
    r.raise_for_status()
    return r.json()["score"]


# Keep a rolling window of recent rewards for context injection
_recent_rewards: list = []
MAX_CONVERSATION_WINDOW = 8


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


def obs_to_prompt(obs: Dict) -> str:
    lines = [
        f"Step {obs['sim_step']}/{obs['max_steps']} | Hour {obs['sim_hour']:02d}:00 "
        f"| Global RPS: {obs['global_rps']} "
        f"| Budget: ${obs['budget_consumed_usd']:.2f}/${obs['budget_total_usd']:.0f}",
    ]

    # Inject rolling reward context so the agent knows its trajectory
    if len(_recent_rewards) >= 2:
        last5 = _recent_rewards[-5:]
        trend = "improving" if last5[-1] > last5[0] else "declining" if last5[-1] < last5[0] else "stable"
        lines.append(f"REWARD TREND ({trend}): last 5 rewards = [{', '.join(f'{r:.2f}' for r in last5)}]")

    # Capacity safety check
    lines.append(f"CAPACITY CHECK: {_capacity_summary(obs)}")

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


def call_llm(obs_text: str, conversation: list) -> Dict:
    conversation.append({"role": "user", "content": obs_text})
    # Trim conversation to last N messages to reduce token burn
    if MAX_CONVERSATION_WINDOW > 0:
        trimmed = conversation[-MAX_CONVERSATION_WINDOW:]
    else:
        trimmed = [{"role": "user", "content": obs_text}]
    # Use simple json_object which is supported by most LLM endpoint providers (vLLM, TGI, OpenAI)
    resp_format = {"type": "json_object"}

    for attempt in range(8):
        try:
            # Respect rate limits manually if configured explicitly
            if API_BASE_URL and "googleapis" in API_BASE_URL:
                time.sleep(4)
                
            # Gemma models on Google AI don't support 'system' role natively
            if trimmed and trimmed[0]["role"] == "user":
                messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + trimmed[0]["content"]}] + trimmed[1:]
            else:
                messages = [{"role": "user", "content": SYSTEM_PROMPT}] + trimmed
                
            # Gemma models don't support JSON mode natively on this endpoint yet
            if "gemma" in MODEL_NAME.lower():
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.2,
                )
            else:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    response_format=resp_format,
                    max_tokens=200,
                    temperature=0.2,
                )
            raw = response.choices[0].message.content
            
            # Handle empty/None responses (common with thinking models)
            if not raw or not raw.strip():
                print(f"  [LLM] Empty response, retrying...", file=sys.stderr)
                continue
            
            # Strip markdown code fences if Gemini wrapped the JSON
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Remove ```json ... ``` or ``` ... ```
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            
            conversation.append({"role": "assistant", "content": cleaned})
            
            parsed = json.loads(cleaned)
            print(f"  [LLM Response] {cleaned[:150]}", file=sys.stderr)
            return parsed
        except json.JSONDecodeError as e:
            print(f"  [LLM] JSON parse failed: {e} | raw={repr(raw[:100] if raw else 'None')}", file=sys.stderr)
            continue
        except Exception as e:
            err_str = str(e).lower()
            if "insufficient_quota" in err_str:
                raise RuntimeError(f"API Billing Error: You have run out of credits or hit a hard quota. Message: {e}")
            if "rate_limit" in err_str or "429" in err_str or "too many requests" in err_str or "quota" in err_str:
                wait_time = 15 + (15 * attempt)
                print(f"  [LLM Rate Limit] Sleeping {wait_time}s to reset quota block...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed to get LLM response after 8 retries.")

def log_start(task_id: int, model_name: str):
    task_name = f"Task_{task_id}"
    print(f"[START] task={task_name} env=finops-env model={model_name}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: str = None):
    # Action must be a string representation, error must be "null" if None
    action_str = json.dumps(action).replace(" ", "") 
    err_str = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _is_safe_to_terminate(obs: Dict, action: Dict) -> bool:
    if action.get("action_type") != "terminate_node":
        return True  # Not a terminate action, always safe

    tier_rps = {"standard": 250, "compute": 400, "high-memory": 200}
    target_region = action.get("region", "")
    target_node_id = action.get("node_id", "")

    # Get active nodes in the target region
    active_nodes = [n for n in obs.get("active_nodes", []) if n.get("active") and n["region"] == target_region]

    # Never terminate if 3 or fewer nodes remain
    if len(active_nodes) <= 3:
        return False

    # Calculate capacity after removal
    remaining_cap = sum(
        tier_rps.get(n["tier"], 250) for n in active_nodes if n["node_id"] != target_node_id
    )

    # Get incoming RPS for this region
    rw = obs.get("routing_weights", {}).get(target_region, 0)
    region_rps = obs.get("global_rps", 0) * rw

    # Block if remaining capacity < 1.3x demand
    return remaining_cap >= region_rps * 1.3


def run_task(task_id: int, seed: int = 42) -> float:
    global _recent_rewards
    _recent_rewards = []  # Reset for each task
    log_start(task_id, MODEL_NAME)
    obs          = reset_env(task_id=task_id, seed=seed)
    conversation = []
    step_count   = 0
    start        = time.time()
    rewards      = []

    try:
        while True:
            if time.time() - start > MAX_EPISODE_SECONDS:
                break

            try:
                action = call_llm(obs_to_prompt(obs), conversation)
            except Exception as e:
                print(f"  [LLM Error] {type(e).__name__}: {e}", file=sys.stderr)
                action = {"action_type": "noop"}

            # Programmatic safety: override unsafe terminate_node → noop
            if not _is_safe_to_terminate(obs, action):
                action = {"action_type": "noop"}

            error_str = None
            try:
                result = step_env(action)
                obs        = result["observation"]
                done       = result["done"]
                reward     = result.get("reward", 0.0)
                rewards.append(reward)
                _recent_rewards.append(reward)
            except Exception as e:
                error_str = str(e)
                done = True
                reward = 0.0
                rewards.append(reward)

            step_count += 1
            log_step(step_count, action, reward, done, error_str)

            if done:
                break

        score = get_grade()
        success = True # or logic based on your scoring

    finally:
        score_val = score if 'score' in locals() else 0.0
        success_val = success if 'success' in locals() else False
        log_end(success_val, step_count, score_val, rewards)

    return score_val


def main():
    try:
        requests.get(f"{ENV_SERVER_URL}/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {ENV_SERVER_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task_id in [1, 2, 3]:
        scores[task_id] = run_task(task_id=task_id)


if __name__ == "__main__":
    main()
