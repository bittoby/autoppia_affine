# Train an IWA WebAgent with AgentGym-RL (Step-by-step)

This doc walks you from zero to PPO training of a web agent against the Interactive Web Apps (IWA) environment.

- Repo paths used below:
  - Env server: `AgentGym/agentenv-iwa/`
  - RL trainer: `AgentGym-RL/AgentGym-RL/`

## 1) Prerequisites

- Python 3.10+
- A virtual environment for AgentGym-RL (recommended)
- Playwright Chromium browsers
- Autoppia IWA demo web running locally (e.g., at `http://localhost:8008/`)
- Hugging Face model accessible locally or via internet (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)

Suggested venv (from `AgentGym-RL/AgentGym-RL`):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2) Install env and deps

- Install the `agentenv` package (imported by AgentGym-RL):
```bash
pip install -e /home/chan/Agentenv_workspace/AgentGym/agentenv
python -c "import agentenv; print(agentenv.__file__)"
```
- Install Playwright browsers (from `AgentGym/agentenv-iwa/autoppia_iwa/`):
```bash
python -m playwright install --with-deps
```

## 3) Start the IWA environment server

From `AgentGym/agentenv-iwa`:
```bash
python -m agentenv_iwa.launch --host 0.0.0.0 --port 8060 --workers 1
```
Health check:
```bash
curl -s http://localhost:8060/
# {"status":"ok"}
```

## 4) Create a task configuration

Task configs are in `AgentGym/agentenv-iwa/config_files/{idx}.json`. Example `0.json`:
```json
{
  "project_id": "autoconnect",
  "task": {
    "id": "task-0001",
    "prompt": "Log in as Demo User and navigate to the profile page.",
    "url": "http://localhost:8008/",
    "tests": [
      {
        "type": "CheckEventTest",
        "event_name": "VIEW_USER_PROFILE",
        "event_criteria": {"name": "Demo User"},
        "description": "Profile page viewed"
      }
    ],
    "relevant_data": {},
    "should_record": false,
    "web_project_id": "autoconnect"
  }
}
```

## 5) Smoke test the env HTTP API

```bash
# Create env
curl -s -X POST http://localhost:8060/create
# -> {"env_idx":0}

# Reset and load task idx=0
curl -s -X POST http://localhost:8060/reset \
  -H 'Content-Type: application/json' \
  -d '{"env_idx":0,"seed":0,"idx":0}'

# Observation
curl -s "http://localhost:8060/observation?env_idx=0"

# Step (navigate)
curl -s -X POST http://localhost:8060/step \
  -H 'Content-Type: application/json' \
  -d '{"env_idx":0,"action":"```{\"type\":\"NavigateAction\",\"url\":\"http://localhost:8008/\"}```"}'

# Finish/evaluate
curl -s -X POST http://localhost:8060/step \
  -H 'Content-Type: application/json' \
  -d '{"env_idx":0,"action":"```stop```"}'
```

## 6) Configure PPO (CPU-friendly minimal setup)

Edit `AgentGym-RL/AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`:

- Point the RL client at IWA:
```yaml
actor_rollout_ref:
  agentgym:
    task_name: iwa
    env_addr: 'http://localhost:8060'
    max_retries: 10
    max_rounds: 10
    timeout: 300
  rollout:
    name: hf
    do_sample: true
    n: 1
```

- Use a small HF model for both actor and critic:
```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-0.5B-Instruct
critic:
  model:
    path: Qwen/Qwen2.5-0.5B-Instruct
    tokenizer_path: Qwen/Qwen2.5-0.5B-Instruct
```

- Keep tiny batching for CPU:
```yaml
data:
  train_batch_size: 1

actor_rollout_ref:
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 2
    ppo_micro_batch_size: null
    ppo_micro_batch_size_per_gpu: 1
    ppo_epochs: 1

critic:
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
  ppo_micro_batch_size: null
  ppo_micro_batch_size_per_gpu: 1
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
```

- Trainer runtime (single logical GPU to pass validation):
```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 1
  total_epochs: 1
  project_name: iwa_rl
  experiment_name: ppo_iwa_run1
  logger: ['console']
```

Notes:
- The validator treats world size as `nnodes * n_gpus_per_node`. Using 1 avoids divisibility assertions and does not require an actual GPU when running CPU-only.

### 6.1) Configuration modes (CPU / single-GPU / multi-GPU)

- **CPU-only (minimal)**
  ```yaml
  data:
    train_batch_size: 1
  actor_rollout_ref:
    rollout:
      name: hf
      n: 1
    actor:
      strategy: fsdp
      ppo_mini_batch_size: 2
      ppo_micro_batch_size: null
      ppo_micro_batch_size_per_gpu: 1
      ppo_epochs: 1
    model:
      path: Qwen/Qwen2.5-0.5B-Instruct
  critic:
    model:
      path: Qwen/Qwen2.5-0.5B-Instruct
      tokenizer_path: Qwen/Qwen2.5-0.5B-Instruct
    ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
    ppo_micro_batch_size: null
    ppo_micro_batch_size_per_gpu: 1
    ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  trainer:
    nnodes: 1
    n_gpus_per_node: 1  # logical 1 to pass validation; still runs on CPU
    total_epochs: 1
  ```
  - Tip: Keep `ulysses_sequence_parallel_size: 1`. If Ray/FSDP later requests GPUs, we can flip the resource pools to CPU in code.

- **Single-GPU (HF rollout)**
  ```yaml
  data:
    train_batch_size: 4
  actor_rollout_ref:
    rollout:
      name: hf
      n: 1
    actor:
      strategy: fsdp
      ppo_mini_batch_size: 4
      ppo_micro_batch_size: null
      ppo_micro_batch_size_per_gpu: 1
      ppo_epochs: 1
    model:
      path: Qwen/Qwen2.5-1.5B-Instruct  # example larger model
  critic:
    model:
      path: ${actor_rollout_ref.model.path}
      tokenizer_path: ${actor_rollout_ref.model.path}
    ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
    ppo_micro_batch_size: null
    ppo_micro_batch_size_per_gpu: 1
    ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  trainer:
    nnodes: 1
    n_gpus_per_node: 1
    total_epochs: 1
  ```
  - Validation must hold: `train_batch_size % (nnodes*n_gpus_per_node) == 0` and `ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0`.

- **Multi-GPU (HF rollout, data parallel)**
  ```yaml
  data:
    train_batch_size: 16
  actor_rollout_ref:
    rollout:
      name: hf
      n: 1
    actor:
      strategy: fsdp
      ppo_mini_batch_size: 16
      ppo_micro_batch_size: null
      ppo_micro_batch_size_per_gpu: 2   # per GPU
      ppo_epochs: 1
      ulysses_sequence_parallel_size: 1
  critic:
    ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
    ppo_micro_batch_size: null
    ppo_micro_batch_size_per_gpu: 2
  trainer:
    nnodes: 1
    n_gpus_per_node: 8   # example
    total_epochs: 1
  ```
  - Checks: `train_batch_size % (nnodes*n_gpus_per_node) == 0` and `ppo_micro_batch_size_per_gpu * sp_size >= n_gpus_per_node` (with `sp_size=1`).

- **Multi-GPU with vLLM rollout (advanced)**
  ```yaml
  actor_rollout_ref:
    rollout:
      name: vllm
      dtype: bfloat16
      gpu_memory_utilization: 0.9
      tensor_model_parallel_size: 1
      max_model_len: 32768
      max_num_batched_tokens: 8192
      ignore_eos: false
      enable_chunked_prefill: true
  ```
  - Requires CUDA and vLLM installed; ignore this mode on CPU-only hosts.
  - Keep PPO/critic batch rules as above. Ensure your vLLM model fits GPU memory.

## 7) Launch PPO training

From `AgentGym-RL/AgentGym-RL` (activate the venv first):
```bash
python -m verl.agent_trainer.main_ppo
```
Hydra loads `ppo_trainer.yaml` automatically. Checkpoints/logs will be in:
```
checkpoints/iwa_rl/ppo_iwa_run1
```

## 8) Action format (what the policy outputs)

Actions are a single JSON object wrapped in triple backticks:

- Navigate:
```
```{"type":"NavigateAction","url":"http://localhost:8008/"}```
```
- Click (XPath):
```
```{"type":"ClickAction","selector":{"type":"xpathSelector","value":"//button[text()='Login']"}}```
```
- Type by element id:
```
```{"type":"TypeAction","selector":{"type":"attributeValueSelector","attribute":"id","value":"username"},"text":"demo"}```
```
- Stop/evaluate:
```
```stop```
```

## 9) Troubleshooting

- `ModuleNotFoundError: agentenv`:
  ```bash
  pip install -e /home/chan/Agentenv_workspace/AgentGym/agentenv
  python -c "import agentenv; print(agentenv.__file__)"
  ```
- vLLM/CUDA warnings with HF rollout: harmless when `rollout.name: hf`.
- Ray/FSDP CPU caveats: the trainer is GPU-first. If a later error says Ray cannot allocate GPUs or FSDP/CUDA init fails, switch resource pools to CPU in code:
  - In `verl/agent_trainer/ppo/ray_trainer.py`, set `use_gpu=False` in `ResourcePoolManager.create_resource_pool()`.
  - Ensure your PyTorch build supports CPU-only.
- Playwright Chromium:
  ```bash
  python -m playwright install chromium
  ```
- Demo web URL mismatch: ensure the task `url` in `config_files/{idx}.json` matches the running demo web.

## 10) Where things live

- Env server: `agentenv-iwa/agentenv_iwa/`
- Env launch: `agentenv-iwa/agentenv_iwa/launch.py`
- Client mapping: `AgentGym-RL/AgentGym-RL/verl/utils/agentgym/client.py`
- PPO config: `AgentGym-RL/AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml`
- Task configs: `agentenv-iwa/config_files/{idx}.json`
