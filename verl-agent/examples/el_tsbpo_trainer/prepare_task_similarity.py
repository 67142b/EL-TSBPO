#!/usr/bin/env python3
"""Precompute frozen task vectors before allocating training workers/GPU memory."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_task_texts(config):
    """Enumerate training instructions using the environment's own task pool."""
    if config.env.env_name.lower() == 'alfworld/alfredtwenv':
        from agent_system.environments.env_package.alfworld.envs import load_config_file
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
        from agent_system.environments.env_package.alfworld.task_texts import root_task_texts
        alf = load_config_file(str(ROOT / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'))
        base = get_environment(alf['env']['type'])(alf, train_eval='train')
        return root_task_texts(sorted(base.game_files)), 'alfworld-task-grammar-v1'
    if config.env.env_name.lower() != "webshop":
        raise ValueError("Automatic task-cache building supports WebShop and ALFWorld/AlfredTWEnv")
    web = ROOT / "agent_system/environments/env_package/webshop/webshop"
    sys.path.insert(0, str(web))
    from web_agent_site.envs import WebAgentTextEnv
    env = WebAgentTextEnv(
        seed=int(config.env.seed), observation_mode="text", num_products=1000,
        human_goals=False, auto_reset=False, retain_sessions=False,
        file_path=str(web / "data/items_shuffle_1000.json"),
        attr_path=str(web / "data/items_ins_v2_1000.json"))
    try:
        texts = {str(i): str(goal["instruction_text"]) for i, goal in enumerate(env.server.goals) if i >= 500}
    finally:
        env.close()
    return texts, f"webshop-seed{int(config.env.seed)}-synthetic-1k"


def prepare_before_training(config):
    """Use the final training config and release the encoder before Ray starts."""
    from omegaconf import OmegaConf

    if str(OmegaConf.select(config, "algorithm.adv_estimator", default="")) != "el_tsbpo":
        return False
    cfg = config.algorithm.get("el_tsbpo", {})
    backend = str(cfg.get("allocation_backend", "llm")).lower().replace("-", "_")
    if backend not in {"agent_dps", "dps", "agentdps"} or not cfg.get("agent_dps_root_similarity_enabled", False):
        return False

    print("[DPS-TASK-EMBEDDINGS] Preparing semantic root cache before Ray initialization", flush=True)
    # The child gets all resolved CLI overrides, rather than reconstructing a
    # second, potentially different Hydra configuration from shell defaults.
    with tempfile.TemporaryDirectory(prefix="agent-dps-task-cache-") as directory:
        path = Path(directory) / "training_config.yaml"
        OmegaConf.save(config, path, resolve=True)
        subprocess.run(
            [sys.executable, "-m", "examples.el_tsbpo_trainer.prepare_task_similarity", "--config-file", str(path)],
            cwd=ROOT,
            check=True,
        )
    print("[DPS-TASK-EMBEDDINGS] Cache verified; embedding process exited before training workers start", flush=True)
    return True


def main(argv=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, help="Resolved training configuration (internal preflight)")
    parser.add_argument("--device", default="cuda", help="Embedding device; CPU caches can also be used by GPU training")
    parser.add_argument("--batch-size", type=int, default=64)
    args, overrides = parser.parse_known_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.config_file is not None:
        if overrides:
            parser.error("--config-file cannot be combined with Hydra overrides")
        config = OmegaConf.load(args.config_file)
    else:
        with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
            config = compose(config_name="el_tsbpo", overrides=overrides)
    OmegaConf.resolve(config)
    cfg = config.algorithm.el_tsbpo
    if not cfg.agent_dps_root_similarity_enabled:
        return
    texts, namespace = load_task_texts(config)
    from agent_system.multi_turn_rollout.task_similarity import prepare_task_vectors, validate_task_cache
    result = prepare_task_vectors(
        texts, cfg.agent_dps_root_embedding_model_path, cfg.agent_dps_root_embedding_cache_dir,
        device=args.device, batch_size=args.batch_size, namespace=namespace)
    validate_task_cache(
        texts, cfg.agent_dps_root_embedding_model_path,
        cfg.agent_dps_root_embedding_cache_dir, namespace=namespace,
    )
    result["validated"] = True
    print("[DPS-TASK-EMBEDDINGS] " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
