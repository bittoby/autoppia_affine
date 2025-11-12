from typing import Any, Mapping, Dict
import re
import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class IwaEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": (
                "You are an autonomous agent operating a browser on interactive web apps (IWA). "
                "You will receive a task prompt and current page HTML. "
                "Respond with exactly one JSON action inside triple backticks. "
                "Example actions: {\"type\": \"NavigateAction\", \"url\": \"http://...\"}, "
                "{\"type\": \"ClickAction\", \"selector\": {\"type\": \"xpathSelector\", \"value\": \"//button[text()='Submit']\"}}, "
                "{\"type\": \"TypeAction\", \"selector\": {\"type\": \"attributeValueSelector\", \"attribute\": \"id\", \"value\": \"input-id\"}, \"text\": \"hello\"}. "
                "Finish with `stop` to trigger evaluation."
            )
        }),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(self, env_server_base: str, data_len: int, *args, timeout: int = 600, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()["env_idx"]

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(f"{self.env_server_base}/{path}", params={"env_idx": self.env_id}, timeout=self.timeout)
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        response = self._get("observation")
        return response.get("observation", "")

    def step(self, action: str) -> StepOutput:
        # action should be inside triple backticks
        _action = re.findall(r"```(.*?)```", action, re.DOTALL)
        if len(_action) == 0:
            return StepOutput(
                state="Your action must be a single JSON object inside triple backticks.",
                reward=0.0,
                done=False,
            )
        response = self._post("step", {"action": _action[0]})
        reward = response["reward"] if response.get("terminated") else 0.0
        return StepOutput(
            state=response.get("observation", ""),
            reward=reward,
            done=bool(response.get("terminated", False)),
        )

    def reset(self, idx: int) -> Dict[str, Any]:
        response = self._post("reset", {"seed": 0, "idx": idx})
        return response

    def close(self):
        response = self._post("close", {})
        return response


class IwaTask(BaseTask):
    env_client_cls = IwaEnvClient
    env_name = "IWA"

    def __init__(self, client_args: Mapping[str, Any] | Mapping[str, Any], n_clients: int, *args, **kwargs):
        super().__init__(client_args, n_clients, *args, **kwargs)
