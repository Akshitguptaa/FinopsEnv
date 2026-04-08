"""FinOpsEnv FastAPI server.

Architecture:
  - OpenEnv's HTTPEnvServer provides the standard /ws WebSocket endpoint
    and protocol-compliant /health, /schema, /metadata, /docs endpoints.
  - Custom stateful HTTP routes (/reset, /step, /state, /grade) maintain
    per-session environment state for the inference script.
  - Both access methods share the same FinOpsEnv implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openenv.core.env_server import HTTPEnvServer

from finopsenv.env import FinOpsEnv
from finopsenv.schemas.action import FinOpsAction
from finopsenv.schemas.observation import FinOpsObservation

app = FastAPI(
    title="FinOpsEnv",
    description="Cloud FinOps & Carbon-Aware Orchestration Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env_server = HTTPEnvServer(
    FinOpsEnv,
    action_cls=FinOpsAction,
    observation_cls=FinOpsObservation,
    max_concurrent_envs=100,
)
_env_server.register_routes(app, mode="production")

_sessions: Dict[str, FinOpsEnv] = {}
_DEFAULT_SESSION = "default"


def _get_or_create(session_id: str) -> FinOpsEnv:
    if session_id not in _sessions:
        _sessions[session_id] = FinOpsEnv()
    return _sessions[session_id]


class ResetRequest(BaseModel):
    seed:       int = 42
    task_id:    int = 1
    session_id: str = _DEFAULT_SESSION


class StepRequest(BaseModel):
    action:     FinOpsAction
    session_id: str = _DEFAULT_SESSION


@app.post("/reset", tags=["FinOps Session"])
async def reset(req: Optional[ResetRequest] = None):
    # new episode for the id and task.
    if req is None:
        req = ResetRequest()

    env = _get_or_create(req.session_id)
    obs = env.reset(seed=req.seed, task_id=req.task_id)
    return {"observation": obs.model_dump(), "session_id": req.session_id}


@app.post("/step", tags=["FinOps Session"])
async def step(req: StepRequest):
    # action - provision, terminate, migrate, or noop 
    # Returns the next state observation, dense reward

    env = _get_or_create(req.session_id)
    try:
        obs = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state", tags=["FinOps Session"])
async def state_endpoint(session_id: str = _DEFAULT_SESSION):
    env = _get_or_create(session_id)
    try:
        return env.state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade", tags=["FinOps Session"])
async def grade_endpoint(session_id: str = _DEFAULT_SESSION):
    env = _get_or_create(session_id)
    try:
        return {"score": env.grade(), "session_id": session_id}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks", tags=["FinOps"])
async def list_tasks():
    from finopsenv.simulation.constants import TASK_CONFIGS
    return {"tasks": TASK_CONFIGS}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()
