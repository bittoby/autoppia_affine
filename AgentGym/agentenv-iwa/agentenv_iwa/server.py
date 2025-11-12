from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from .environment import iwa_env_server

app = FastAPI(title="AgentGym IWA Env Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_max_id = 0
_id_lock = asyncio.Lock()


class CreateResponse(BaseModel):
    env_idx: int


class ResetRequest(BaseModel):
    env_idx: int
    seed: int | None = None
    idx: int | None = None
    file: str | None = None
    task_idx: int | None = None


class StepRequest(BaseModel):
    env_idx: int
    action: str


class CloseRequest(BaseModel):
    env_idx: int


@app.get("/")
async def ok():
    return {"status": "ok"}


@app.post("/create", response_model=CreateResponse)
async def create():
    global _max_id
    async with _id_lock:
        env_idx = _max_id
        _max_id += 1
    await asyncio.to_thread(iwa_env_server.create, env_idx)
    return {"env_idx": env_idx}


@app.post("/reset")
async def reset(req: ResetRequest):
    import json, os
    project_id = None
    task = None
    if req.file:
        cfg_path = req.file if os.path.isabs(req.file) else os.path.join("./config_files", req.file)
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return {"observation": "ConfigError", "project_id": None}
        tidx = req.task_idx or 0
        # Support three shapes (preference: keep each task object same as single-task structure):
        # 1) List[ {"project_id": ..., "task": {...}} ]
        # 2) { "tasks": [ {"project_id": ..., "task": {...}}, ... ] }
        # 3) Single-task dict {"project_id": ..., "task": {...}} (for convenience)
        if isinstance(cfg, list):
            if tidx < 0 or tidx >= len(cfg):
                return {"observation": "ConfigError", "project_id": None}
            item = cfg[tidx]
            if not isinstance(item, dict):
                return {"observation": "ConfigError", "project_id": None}
            project_id = item.get("project_id")
            task = item.get("task")
        elif isinstance(cfg, dict) and "tasks" in cfg:
            tasks = cfg.get("tasks") or []
            if not isinstance(tasks, list) or tidx < 0 or tidx >= len(tasks):
                return {"observation": "ConfigError", "project_id": None}
            item = tasks[tidx]
            if not isinstance(item, dict):
                return {"observation": "ConfigError", "project_id": None}
            project_id = item.get("project_id")
            task = item.get("task")
        else:
            project_id = cfg.get("project_id") if isinstance(cfg, dict) else None
            task = cfg.get("task") if isinstance(cfg, dict) else None
    else:
        if req.idx is None:
            return {"observation": "ConfigError", "project_id": None}
        cfg_path = os.path.join("./config_files", f"{req.idx}.json")
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return {"observation": "ConfigError", "project_id": None}
        project_id = cfg.get("project_id") if isinstance(cfg, dict) else None
        task = cfg.get("task") if isinstance(cfg, dict) else None

    if not project_id or not task:
        return {"observation": "ConfigError", "project_id": None}
    options = {"project_id": project_id, "task": task}
    obs, info = await asyncio.to_thread(iwa_env_server.reset, req.env_idx, req.seed, options)
    return {"observation": obs, "project_id": options.get("project_id"), "url": info.get("url")}


@app.get("/observation")
async def observation(env_idx: int):
    obs = await asyncio.to_thread(iwa_env_server.observation, env_idx)
    return {"observation": obs}


@app.post("/step")
async def step(req: StepRequest):
    obs, reward, terminated, truncated, info = await asyncio.to_thread(iwa_env_server.step, req.env_idx, req.action)
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


@app.post("/close")
async def close(req: CloseRequest):
    ret = await asyncio.to_thread(iwa_env_server.close, req.env_idx)
    return ret
