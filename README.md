# EL-TSBPO

EL-TSBPO is a two-stage training workflow for language-model agents in ALFWorld and WebShop. Successful teacher trajectories provide an SFT cold start; a task-aware root and branch allocator then collects physical rollout trees for RL. The RL implementation extends `verl-agent` and retains its `verl` package layout.

![EL-TSBPO training framework](docs/assets/figure1_training_framework.png)

## Repository layout

| Path | Contents |
| --- | --- |
| [`verl-agent/`](verl-agent/) | RL framework, ALFWorld and WebShop environments, EL-TSBPO algorithm, collectors, tests, and two training scripts |
| [`llamafactory/`](llamafactory/) | SFT configurations and dataset registration; install LlamaFactory separately |
| [`docs/assets/`](docs/assets/) | Figures from the manuscript |

The SFT datasets, checkpoints, ALFWorld game data, and WebShop product data are deliberately excluded. Local outputs are written under `verl-agent/.local/` and ignored by Git.

## Method

The rollout tree records each factual trajectory and the continuations sampled from selected states. Physical tree relationships determine trajectory credit, and state-action groups provide local comparisons. Segment balancing is an optional policy-weighting setting in the RL configuration.

![Tree credit assignment](docs/assets/figure2_tree_credit.png)

The allocator uses calibrated task and state evidence to distribute factual root and branch samples under separate budgets. Branch selection conditions on the remaining interaction horizon and allocates within a task-level mixed-outcome objective.

![Root and branch budget allocation](docs/assets/figure3_budget_allocation.png)

## Setup

Run these commands from the repository root on Linux with NVIDIA GPUs, a working driver, and Conda. The combinations below match the two benchmark environments used during development; use PyTorch and FlashAttention wheels built for your CUDA driver and Python ABI. Keep ALFWorld, WebShop, and SFT in separate environments.

```bash
git clone https://github.com/67142b/EL-TSBPO.git
cd EL-TSBPO
nvidia-smi
```

### ALFWorld RL environment

The ALFWorld stack uses Python 3.12, PyTorch 2.8.0 with CUDA 12.8, and vLLM 0.11.0. Install the matching PyTorch build first, then vLLM, FlashAttention, the text environment, and this repository:

```bash
conda create -n el-tsbpo-alfworld python=3.12 -y
conda activate el-tsbpo-alfworld
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.11.0
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
python -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2
python -m pip install -e ./verl-agent
```

FlashAttention source builds require a matching CUDA toolkit and `nvcc`; a wheel matching Python, Torch, and CUDA can be used instead. Download the TextWorld game assets and check the files required by the launcher:

```bash
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download --data-dir "$ALFWORLD_DATA" -f
test -f "$ALFWORLD_DATA/logic/alfred.pddl"
test -f "$ALFWORLD_DATA/logic/alfred.twl2"
test -d "$ALFWORLD_DATA/json_2.1.1/train"
python -c 'import torch, gymnasium, alfworld, vllm; print(torch.cuda.is_available())'
```

### WebShop RL environment

WebShop's installation script requires Python 3.10.1–3.10.x; the compatible training stack uses Python 3.10.19, PyTorch 2.6.0 with CUDA 12.4, Ray 2.43.0, and vLLM 0.8.5. Its setup script installs environment dependencies, downloads the 1,000-product catalog, and builds the corresponding search index. It also installs OpenJDK 11 when Java is unavailable.

```bash
conda create -n el-tsbpo-webshop python=3.10.19 -y
conda activate el-tsbpo-webshop
cd verl-agent/agent_system/environments/env_package/webshop/webshop
bash setup.sh -d small
cd ../../../../..
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r agent_system/environments/env_package/webshop/webshop/requirements_vllm.txt
python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install -e .
export JAVA_HOME="$CONDA_PREFIX"
test -s agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json
test -d agent_system/environments/env_package/webshop/webshop/search_engine/indexes_1k
java -version
cd ..
```

The FlashAttention wheel above is for Linux x86-64 and the listed Python/Torch/CUDA build; choose a matching wheel on other platforms. `setup.sh -d small` is repeatable and reuses complete downloads and indexes. The launcher expects this small catalog; `-d all` prepares a different catalog.

### SFT environment and models

Install upstream LlamaFactory in a separate Python 3.11 environment. The provided configurations use DeepSpeed ZeRO-2 and FlashAttention 2:

```bash
conda create -n el-tsbpo-sft python=3.11 -y
conda activate el-tsbpo-sft
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install llamafactory deepspeed
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

The SFT configurations download `Qwen/Qwen2.5-1.5B-Instruct` when it is not cached and write their outputs under `verl-agent/.local/models/`. RL also needs a frozen Qwen3-Embedding-0.6B encoder. Download it to the launchers' default location from the repository root, or point `EL_TSBPO_EMBEDDING_MODEL` to an existing local copy:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-Embedding-0.6B", local_dir="verl-agent/.local/models/Qwen3-Embedding-0.6B")'
```

## Stage 1: SFT cold start

The collectors run against an OpenAI-compatible teacher API and write only successful training trajectories. They default to `gpt-5.6-sol`; set `OPENAI_API_KEY`, `OPENAI_BASE_URL` if needed, and `OPENAI_MODEL` to override the teacher. Run each collector in its benchmark environment from the repository root, then switch to the SFT environment:

```bash
conda activate el-tsbpo-alfworld
python verl-agent/examples/prompt_agent/openai_alfworld_collector.py

conda activate el-tsbpo-webshop
export JAVA_HOME="$CONDA_PREFIX"
python verl-agent/examples/prompt_agent/openai_webshop_collector.py

conda activate el-tsbpo-sft
llamafactory-cli train llamafactory/configs/alfworld.yaml
llamafactory-cli train llamafactory/configs/webshop.yaml
```

The default output files are `llamafactory/data/data.json` and `llamafactory/data/webshop_success_data.json`. Collection makes paid API requests and is never triggered by the training scripts. See [`llamafactory/README.md`](llamafactory/README.md) for the dataset format, separate SFT environment, and training commands.

## Stage 2: RL training

After SFT, the configurations place the checkpoints at `verl-agent/.local/models/alfworld-sft` and `verl-agent/.local/models/webshop-sft`. The embedding download in Setup uses the launchers' default path; set `EL_TSBPO_EMBEDDING_MODEL` when using a different local copy. Activate the corresponding RL environment before running each benchmark:

```bash
conda activate el-tsbpo-alfworld
cd verl-agent
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/el_tsbpo_trainer/run_alfworld.sh
cd ..

conda activate el-tsbpo-webshop
export JAVA_HOME="$CONDA_PREFIX"
cd verl-agent
CUDA_VISIBLE_DEVICES=0,1 bash examples/el_tsbpo_trainer/run_webshop.sh
cd ..
```

The WebShop script requires an even number of GPUs. Set `EL_TSBPO_MODEL` to override a checkpoint and `EL_TSBPO_WORK_DIR` to relocate local training outputs. The branch semantic encoder uses the first selected GPU when its bounded headroom check passes; set `EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES` to use another GPU. The common RL defaults live in [`el_tsbpo.yaml`](verl-agent/verl/trainer/config/el_tsbpo.yaml); the scripts contain only benchmark-specific overrides. They call the trainer directly, without a `screen` session.

## Verification

In the RL environment, run the focused CPU checks:

```bash
conda activate el-tsbpo-alfworld
cd verl-agent
python -c 'import verl, el_tsbpo; print(verl.__file__); print(el_tsbpo.__file__)'
python -m pip install pytest
python -m pytest -q tests/trainer/ppo
```

The printed module paths should be inside this checkout. If either points to another editable checkout, activate the new benchmark environment and reinstall this repository there. The launchers also prepend their own repository root to `PYTHONPATH` for Ray workers. Full training requires GPU, environment assets, and model checkpoints. The repository contains no pretrained weights or collected demonstrations.

## Licenses

This repository is distributed under [Apache License 2.0](LICENSE). It includes code derived from `verl` and `verl-agent`; nested WebShop and ALFWorld environment components retain their MIT license notices. LlamaFactory is a separately installed upstream dependency. See [NOTICE](NOTICE) for attribution.
