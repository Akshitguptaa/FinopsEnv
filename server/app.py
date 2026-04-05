from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from finopsenv import FinOpsEnv, FinOpsAction
from finopsenv.simulation.constants import TASK_CONFIGS

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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinOpsEnv"}


@app.post("/reset")
async def reset(req: ResetRequest):
    # new episode for the id and task.

    env = _get_or_create(req.session_id)
    obs = env.reset(seed=req.seed, task_id=req.task_id)
    return {"observation": obs.model_dump(), "session_id": req.session_id}


@app.post("/step")
async def step(req: StepRequest):
    # action - provision, terminate, migrate, or noop 
    # Returns the next state observation, dense reward

    env = _get_or_create(req.session_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state_endpoint(session_id: str = _DEFAULT_SESSION):
    env = _get_or_create(session_id)
    try:
        return env.state().model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade")
async def grade_endpoint(session_id: str = _DEFAULT_SESSION):
    env = _get_or_create(session_id)
    try:
        return {"score": env.grade(), "session_id": session_id}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    return {"tasks": TASK_CONFIGS}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # commands: ['ping', 'reset', 'step', 'state', 'grade'].

    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = FinOpsEnv()
    _sessions[session_id] = env

    await websocket.send_json({"type": "connected", "session_id": session_id})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            cmd = msg.get("command", "")

            if cmd == "ping":
                await websocket.send_json({"type": "pong"})

            elif cmd == "reset":
                obs = env.reset(
                    seed=int(msg.get("seed", 42)),
                    task_id=int(msg.get("task_id", 1)),
                )
                await websocket.send_json({"type": "observation", "data": obs.model_dump()})

            elif cmd == "step":
                try:
                    action = FinOpsAction(**msg.get("action", {}))
                except (ValidationError, TypeError) as e:
                    await websocket.send_json({"type": "error", "message": f"Invalid action: {e}"})
                    continue
                try:
                    obs, reward, done, info = env.step(action)
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                safe_info = {
                    k: v for k, v in info.items()
                    if isinstance(v, (str, int, float, bool, type(None), dict, list))
                }
                await websocket.send_json({
                    "type":        "step_result",
                    "observation": obs.model_dump(),
                    "reward":      reward,
                    "done":        done,
                    "info":        safe_info,
                })

            elif cmd == "state":
                try:
                    await websocket.send_json({"type": "state", "data": env.state().model_dump()})
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif cmd == "grade":
                try:
                    await websocket.send_json({"type": "grade", "score": env.grade()})
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            else:
                await websocket.send_json({
                    "type":    "error",
                    "message": f"Unknown command '{cmd}'. Valid: reset, step, state, grade, ping.",
                })

    except WebSocketDisconnect:
        _sessions.pop(session_id, None)
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        _sessions.pop(session_id, None)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)


if __name__ == "__main__":
    main()
