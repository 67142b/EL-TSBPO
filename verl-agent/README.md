# EL-TSBPO RL Framework

This directory keeps the `verl-agent` package layout. The active EL-TSBPO code is in `agent_system/multi_turn_rollout/`, `el_tsbpo/`, and the GiGPO state-group utilities in `gigpo/`. ALFWorld and WebShop environments are under `agent_system/environments/`; branch replay workers are in `agent_system/environments/branch_envs.py`.

Install this directory as a Python package with `python -m pip install -e '.[train,alfworld]'` in a Python 3.10 environment. Shared defaults are in `verl/trainer/config/el_tsbpo.yaml`. The two RL launchers are `examples/el_tsbpo_trainer/run_alfworld.sh` and `examples/el_tsbpo_trainer/run_webshop.sh`. See the [repository README](../README.md) for data preparation and the SFT stage.

The upstream `verl` and `verl-agent` portions retain their original package structure and license headers. Local checkpoints, datasets, caches, and logs are ignored by Git.
