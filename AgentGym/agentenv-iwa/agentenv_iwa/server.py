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
    idx: int


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
    # Load config file ./config_files/{idx}.json expected to contain {"project_id": str, "task": {...}}
    import json, os
    cfg_path = os.path.join("./config_files", f"{req.idx}.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return {"observation": "ConfigError", "project_id": None}
    options = {"project_id": cfg.get("project_id"), "task": cfg.get("task")}
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
