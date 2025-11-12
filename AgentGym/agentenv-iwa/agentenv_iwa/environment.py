import asyncio
import json
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, TypedDict

import sys
# Ensure the nested autoppia_iwa package is importable when running from agentenv-iwa/
# This adds: <repo>/AgentGym/agentenv-iwa/autoppia_iwa to sys.path so that
# `import autoppia_iwa...` resolves to <repo>/AgentGym/agentenv-iwa/autoppia_iwa/autoppia_iwa
_PKG_ROOT = Path(__file__).resolve().parents[1] / "autoppia_iwa"
if str(_PKG_ROOT) not in sys.path:
    sys.path.append(str(_PKG_ROOT))

from autoppia_iwa.entrypoints.benchmark.task_generation import get_projects_by_ids
from autoppia_iwa.src.bootstrap import AppBootstrap
from autoppia_iwa.src.data_generation.domain.classes import Task
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.demo_webs.demo_webs_service import BackendDemoWebService
from autoppia_iwa.src.evaluation.classes import EvaluatorConfig
from autoppia_iwa.src.evaluation.evaluator.evaluator import ConcurrentEvaluator
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.execution.browser_executor import PlaywrightBrowserExecutor
from playwright.async_api import async_playwright


class IWASubprocessCommand(TypedDict, total=False):
    cmd: str
    data: Any


async def _construct_observation(page) -> str:
    try:
        html = await page.content()
        url = page.url
        return f"URL: {url}\n\n{html}"
    except Exception as e:
        return f"ObservationError: {e}"


async def _evaluate(project_id: str, task: Task, actions: list[BaseAction], should_record: bool = False) -> float:
    projects = get_projects_by_ids(demo_web_projects, [project_id])
    if not projects:
        return 0.0
    project = projects[0]
    evaluator_config = EvaluatorConfig(enable_grouping_tasks=False, chunk_size=1, should_record_gif=should_record)
    evaluator = ConcurrentEvaluator(project, evaluator_config)
    from autoppia_iwa.src.web_agents.classes import TaskSolution

    task_solution = TaskSolution(task_id=task.id, actions=actions, web_agent_id="0")
    eval_result = await evaluator.evaluate_single_task_solution(task, task_solution)
    return float(getattr(eval_result, "final_score", 0.0))


def iwa_entrypoint(pipe: Connection):
    AppBootstrap()

    running = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    state: dict[str, Any] = {
        "browser": None,
        "context": None,
        "page": None,
        "project_id": None,
        "task": None,
        "actions": [],
        "backend": None,
    }

    async def _reset(seed: int | None, options: dict[str, Any] | None):
        # options should include {"project_id": str, "task": {...}}
        await _close()
        if not options:
            raise ValueError("reset requires options with project_id and task")
        project_id = options.get("project_id")
        task_dict = options.get("task")
        if not project_id or not task_dict:
            raise ValueError("options must contain project_id and task")
        task = Task.deserialize(task_dict) if hasattr(Task, "deserialize") else Task(**task_dict)
        state["project_id"] = project_id
        state["task"] = task
        state["actions"] = []

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(task.url)
        state["_pw"] = playwright
        state["browser"] = browser
        state["context"] = context
        state["page"] = page
        project = get_projects_by_ids(demo_web_projects, [project_id])[0]
        state["backend"] = BackendDemoWebService(project)
        await state["backend"].reset_database(web_agent_id="0")
        obs = await _construct_observation(page)
        return obs, {"url": task.url}

    async def _step(action_text: str):
        # Expect action_text as JSON dict with fields for BaseAction factory, or the string "stop"
        if action_text.strip().lower() == "stop" or action_text.strip().lower().startswith("stop"):
            reward = await _eval()
            return await _construct_observation(state["page"]), reward, True, False, {}
        try:
            action_data = json.loads(action_text)
        except Exception:
            # Try to wrap simple schema {"type":"NavigateAction","url":"..."} from raw lines
            return "Invalid action JSON", 0.0, False, False, {"error": "invalid_action"}
        action = BaseAction.create_action(action_data)
        if action is None:
            return "Unsupported action", 0.0, False, False, {"error": "unsupported_action"}
        executor = PlaywrightBrowserExecutor(browser_config=None, page=state["page"], backend_demo_webs_service=state["backend"])
        res = await executor.execute_single_action(action, web_agent_id="0", iteration=len(state["actions"]) + 1, is_web_real=False, should_record=False)
        state["actions"].append(action)
        obs = await _construct_observation(state["page"])
        return obs, 0.0, False, False, {"last_action_success": res.successfully_executed}

    async def _eval():
        try:
            return await _evaluate(state["project_id"], state["task"], state["actions"], should_record=False)
        except Exception:
            return 0.0

    async def _close():
        try:
            if state.get("backend"):
                try:
                    await state["backend"].close()
                except Exception:
                    pass
            if state.get("context"):
                await state["context"].close()
            if state.get("browser"):
                await state["browser"].close()
            if state.get("_pw"):
                await state["_pw"].stop()
        finally:
            state["browser"] = None
            state["context"] = None
            state["page"] = None
            state["backend"] = None

    while running:
        data: IWASubprocessCommand = pipe.recv()
        cmd = data.get("cmd")
        try:
            if cmd == "close":
                loop.run_until_complete(_close())
                pipe.send({"closed": True})
                running = False
            elif cmd == "reset":
                seed = data.get("data", {}).get("seed")
                options = data.get("data", {}).get("options")
                ret = loop.run_until_complete(_reset(seed, options))
                pipe.send(ret)
            elif cmd == "step":
                action = data.get("data", {}).get("action")
                ret = loop.run_until_complete(_step(action))
                pipe.send(ret)
            elif cmd == "observation":
                if state["page"]:
                    ret = loop.run_until_complete(_construct_observation(state["page"]))
                else:
                    ret = ""
                pipe.send(ret)
            elif cmd == "eval":
                score = loop.run_until_complete(_eval())
                pipe.send(score)
            else:
                pipe.send({"error": f"unknown_cmd:{cmd}"})
        except Exception as e:
            pipe.send({"error": str(e)})


class IwaEnvServer:
    def __init__(self) -> None:
        self.envs: dict[int, multiprocessing.Process] = {}
        self.send: dict[int, Connection] = {}
        self.recv: dict[int, Connection] = {}
        self.state: dict[int, dict[str, Any]] = {}

    def create(self, env_idx: int) -> int:
        (self.send[env_idx], self.recv[env_idx]) = multiprocessing.Pipe()
        p = multiprocessing.Process(target=iwa_entrypoint, args=(self.recv[env_idx],))
        p.start()
        self.envs[env_idx] = p
        self.state[env_idx] = {"actions": []}
        return env_idx

    def _cmd(self, env_idx: int, cmd: str, **data):
        if env_idx not in self.send:
            raise RuntimeError(f"env_idx {env_idx} not found")
        self.send[env_idx].send({"cmd": cmd, "data": data})
        ret = self.send[env_idx].recv()
        if isinstance(ret, dict) and ret.get("error"):
            raise RuntimeError(ret["error"])
        return ret

    def reset(self, env_idx: int, seed: int | None, options: dict[str, Any]):
        obs, info = self._cmd(env_idx, "reset", seed=seed, options=options)
        self.state[env_idx]["actions"] = []
        return obs, info

    def step(self, env_idx: int, action_text: str):
        obs, reward, terminated, truncated, info = self._cmd(env_idx, "step", action=action_text)
        return obs, reward, terminated, truncated, info

    def observation(self, env_idx: int) -> str:
        return self._cmd(env_idx, "observation")

    def close(self, env_idx: int):
        if env_idx in self.envs:
            try:
                self._cmd(env_idx, "close")
            finally:
                try:
                    self.envs[env_idx].terminate()
                except Exception:
                    pass
                self.envs.pop(env_idx, None)
                self.send.pop(env_idx, None)
                self.recv.pop(env_idx, None)
                self.state.pop(env_idx, None)
        return {"closed": True}


iwa_env_server = IwaEnvServer()
