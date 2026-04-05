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

USE_GROQ       = os.environ.get("USE_GROQ", "0").lower() in ("1", "true")
USE_GEMINI     = os.environ.get("USE_GEMINI", "0").lower() in ("1", "true")
USE_OLLAMA     = os.environ.get("USE_OLLAMA", "0").lower() in ("1", "true")

API_BASE_URL   = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MAX_EPISODE_SECONDS = 360

if USE_GROQ:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set. Required when USE_GROQ=1.", file=sys.stderr)
        sys.exit(1)
    MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
elif USE_GEMINI:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Required when USE_GEMINI=1.", file=sys.stderr)
        sys.exit(1)
    # Gemini 2.0 Flash is extremely fast and generous on the free tier
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
elif USE_OLLAMA:
    # Completely free, runs on your local machine if Ollama is running
    MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:3b")
    client = OpenAI(
        api_key="ollama", # required by library but ignored by ollama
        base_url="http://127.0.0.1:11434/v1"
    )
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    MODEL_NAME     = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=OPENAI_API_KEY)

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
Your goal is to minimize cost and carbon footprint while keeping SLAs (CPU below 85%).
Dropped requests are extremely costly to your score.
Over-provisioning wastes money but is safer than under-provisioning.

AVAILABLE ACTIONS:
1. provision_node  — Boot a new node. Use "spot" billing when spot prices are low, else "on-demand".
2. terminate_node  — Shut down an idle node if CPU < 40% and multiple nodes exist.
3. migrate_traffic — Shift traffic between regions to chase low carbon intensity or rebalance load.
4. noop            — Do nothing this step.

Respond ONLY with valid JSON matching the exact schema below. No explanation text.
SCHEMA:
{json.dumps(ACTION_SCHEMA, indent=2)}
"""


def reset_env(task_id: int, seed: int = 42) -> Dict:
    r = requests.post(
        f"{API_BASE_URL}/reset",
        json={"seed": seed, "task_id": task_id, "session_id": "inference"},
    )
    r.raise_for_status()
    return r.json()["observation"]


def step_env(action: Dict) -> Dict:
    r = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action, "session_id": "inference"},
    )
    r.raise_for_status()
    return r.json()


def get_grade() -> float:
    r = requests.get(f"{API_BASE_URL}/grade", params={"session_id": "inference"})
    r.raise_for_status()
    return r.json()["score"]


def obs_to_prompt(obs: Dict) -> str:
    lines = [
        f"Step {obs['sim_step']}/{obs['max_steps']} | Hour {obs['sim_hour']:02d}:00 "
        f"| Global RPS: {obs['global_rps']} "
        f"| Budget: ${obs['budget_consumed_usd']:.2f}/${obs['budget_total_usd']:.0f}",
        "REGIONAL METRICS:",
    ]
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
    # Only OpenAI strictly enforces json_schema structure natively across the board
    if USE_GROQ or USE_GEMINI or USE_OLLAMA:
        resp_format = {"type": "json_object"}
    else:
        resp_format = {
            "type": "json_schema",
            "json_schema": {"name": "FinOpsAction", "schema": ACTION_SCHEMA, "strict": True},
        }

    for attempt in range(8):
        try:
            # Respect Google Gemini Free Tier 15 RPM limit (~4 seconds per request)
            if USE_GEMINI:
                time.sleep(4)
                
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-8:],
                response_format=resp_format,
                max_tokens=200,
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": raw})
            return json.loads(raw)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 15 + (15 * attempt)
                print(f"  [LLM Rate Limit] Sleeping {wait_time}s to reset quota block...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed to get LLM response after 8 rate-limit retries.")

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
    print(f"[END] success={success_str} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def run_task(task_id: int, seed: int = 42) -> float:
    log_start(task_id, MODEL_NAME)
    obs          = reset_env(task_id=task_id, seed=seed)
    conversation = []
    step_count   = 0
    start        = time.time()
    rewards      = []

    while True:
        if time.time() - start > MAX_EPISODE_SECONDS:
            break

        try:
            action = call_llm(obs_to_prompt(obs), conversation)
        except Exception as e:
            action = {"action_type": "noop"}

        error_str = None
        try:
            result = step_env(action)
            obs        = result["observation"]
            done       = result["done"]
            reward     = result.get("reward", 0.0)
            rewards.append(reward)
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
    log_end(True, step_count, score, rewards)
    return score


def main():
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {API_BASE_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task_id in [1, 2, 3]:
        scores[task_id] = run_task(task_id=task_id)


if __name__ == "__main__":
    main()
