# SFT cold start

This directory contains the two SFT configurations and dataset registration used before EL-TSBPO RL training. LlamaFactory itself is installed from its [upstream repository](https://github.com/hiyouga/LLaMA-Factory); its source code is not vendored here. The local configuration baseline was prepared against LlamaFactory `0.9.6.dev0`, which uses Python 3.11 or newer. Keep its Python environment separate from the Python 3.10 RL environment.

From the repository root, collect successful demonstrations using the two scripts in `verl-agent/examples/prompt_agent/`. The ALFWorld output must be `llamafactory/data/data.json`, and the WebShop output must be `llamafactory/data/webshop_success_data.json`. Both are ShareGPT `messages` arrays registered in `data/dataset_info.json`. Failed trajectories and validation tasks are not included in SFT. The dataset files are ignored by Git.

Install LlamaFactory and DeepSpeed in its own environment:

```bash
python -m pip install llamafactory deepspeed
```

Then train from the repository root:

```bash
llamafactory-cli train llamafactory/configs/alfworld.yaml
llamafactory-cli train llamafactory/configs/webshop.yaml
```

Both configurations initialize Qwen2.5-1.5B-Instruct and use two full-parameter SFT epochs. Their outputs match the RL launcher defaults under `verl-agent/.local/models/`. Review the model download, hardware, and teacher API costs before starting either stage.
