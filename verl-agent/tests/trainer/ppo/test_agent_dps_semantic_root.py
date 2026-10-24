"""CPU checks for semantic-root preflight, cache reuse and bounded borrowing."""

import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.agent_dps import AgentDPSStateMemory, root_curriculum_key
from agent_system.multi_turn_rollout.task_similarity import (
    PREPROCESSING,
    TaskSimilarityPrior,
    model_identity,
    prepare_task_vectors,
    text_digest,
    validate_task_cache,
)
from examples.el_tsbpo_trainer.prepare_task_similarity import main, prepare_before_training


class SemanticRootTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.texts = {"source": "put an apple in a bowl", "target": "put the apple in the bowl", "other": "heat a mug"}
        self.namespace = "alfworld-task-grammar-v1"
        self.index = dict(
            identity=dict(model=model_identity(self.model), preprocessing=PREPROCESSING, namespace=self.namespace),
            embedding_dim=2,
            rows=[dict(task_ref=ref, text_sha256=text_digest(text)) for ref, text in self.texts.items()],
        )
        (self.cache / "index.json").write_text(json.dumps(self.index))
        np.save(self.cache / "vectors.npy", np.asarray([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32))
        self.config = OmegaConf.create({
            "algorithm": {"adv_estimator": "el_tsbpo", "el_tsbpo": {
                "allocation_backend": "agent_dps", "agent_dps_root_similarity_enabled": True,
                "agent_dps_root_embedding_model_path": str(self.model),
                "agent_dps_root_embedding_cache_dir": str(self.cache),
            }},
            "env": {"env_name": "alfworld/AlfredTWEnv", "seed": 37},
        })

    def test_cache_reuse_requires_no_encoder(self):
        validate_task_cache(self.texts, self.model, self.cache, namespace=self.namespace)
        result = prepare_task_vectors(self.texts, self.model, self.cache, device="cpu", namespace=self.namespace)
        self.assertEqual(result["misses"], 0)
        self.assertEqual(result["forward_calls"], 0)

    def test_stale_text_model_or_dataset_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing/stale"):
            validate_task_cache({"target": "different goal"}, self.model, self.cache, namespace=self.namespace)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            validate_task_cache(self.texts, self.model, self.cache, namespace="different-dataset")
        (self.model / "config.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(RuntimeError, "identity"):
            validate_task_cache(self.texts, self.model, self.cache, namespace=self.namespace)

    def test_invalid_vector_data_is_rejected(self):
        np.save(self.cache / "vectors.npy", np.zeros((3, 2), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "L2-normalized"):
            validate_task_cache(self.texts, self.model, self.cache, namespace=self.namespace)
        np.save(self.cache / "vectors.npy", np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            validate_task_cache(self.texts, self.model, self.cache, namespace=self.namespace)

    def test_shared_evidence_changes_posterior_without_pseudo_observations(self):
        memory = AgentDPSStateMemory(root_decay=.95)
        source, target = root_curriculum_key("source"), root_curriculum_key("target")
        memory.root_state_keys.update(source=source, target=target)
        for _ in range(20):
            memory.observe(source, True, 0)
        prior = TaskSimilarityPrior(self.cache, max_strength=4.)
        alpha, beta, metadata = prior.parameters(memory, "target", target, 0)
        self.assertAlmostEqual(alpha, 5.)
        self.assertAlmostEqual(beta, 1.)
        self.assertAlmostEqual(metadata["shared_mass"], 4.)
        self.assertEqual(metadata["neighbors"], 1.)
        self.assertEqual(memory.evidence_counts(target, 0), (0., 0.))
        _, _, unrelated = prior.parameters(memory, "other", root_curriculum_key("other"), 0)
        self.assertEqual(unrelated["shared_mass"], 0.)

    def test_collector_uses_shared_prior_for_candidates_and_root_quotas(self):
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

        memory = AgentDPSStateMemory(root_decay=.95)
        key = root_curriculum_key("source")
        memory.root_state_keys["source"] = key
        for _ in range(20):
            memory.observe(key, True, 0)
        cfg = self.config.algorithm.el_tsbpo
        cfg.agent_dps_root_candidate_ratio = 1.5
        collector = SimpleNamespace(
            config=OmegaConf.create({"env": {"rollout": {"n": 6}}}),
            _el_tsbpo_step=1, _agent_dps_task_sampler_restored=True,
            _get_agent_dps_memory=lambda: memory,
            _get_el_tsbpo_config=lambda: cfg,
        )
        for name in ["_agent_dps_root_parameters", "_agent_dps_root_frontier"]:
            setattr(collector, name, MethodType(getattr(TrajectoryCollector, name), collector))
        collector._agent_dps_root_candidates = None
        backend = SimpleNamespace(
            sample_root_task_refs=lambda count, excluded: ["target", "other"][:count],
            get_root_task_texts=lambda refs: {ref: self.texts[ref] for ref in refs},
            load_root_candidates=lambda refs: (["root"] * len(refs), [{"extra.gamefile": ref} for ref in refs]),
        )
        refreshed = TrajectoryCollector._refresh_agent_dps_root_candidates(collector, SimpleNamespace(envs=backend), 2)
        self.assertEqual(len(refreshed[0]), 2)
        quotas, metrics, probabilities = TrajectoryCollector._allocate_agent_dps_roots(
            collector, ["root", "root"], [{"extra.gamefile": "target"}, {"extra.gamefile": "other"}], 12,
        )
        self.assertEqual(int(quotas.sum()), 12)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(metrics["agent_dps/root/semantic_hit_fraction"], 0.)
        self.assertGreater(metrics["agent_dps/root/shared_mass_mean"], 0.)
        self.assertEqual(memory.evidence_counts(root_curriculum_key("target"), 1), (0., 0.))

    def test_preflight_passes_final_config_and_waits_for_child(self):
        def inspect_child(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertEqual(command[1:4], ["-m", "examples.el_tsbpo_trainer.prepare_task_similarity", "--config-file"])
            path = Path(command[-1])
            received = OmegaConf.load(path)
            self.assertEqual(received.env.seed, 37)
            self.assertEqual(received.algorithm.el_tsbpo.agent_dps_root_embedding_cache_dir, str(self.cache))
            captured.append(path)
            return SimpleNamespace(returncode=0)

        captured = []
        with patch("examples.el_tsbpo_trainer.prepare_task_similarity.subprocess.run", side_effect=inspect_child):
            self.assertTrue(prepare_before_training(self.config))
        self.assertFalse(captured[0].exists())

    def test_disabled_or_non_dps_training_skips_preflight(self):
        with patch("examples.el_tsbpo_trainer.prepare_task_similarity.subprocess.run") as run:
            self.config.algorithm.el_tsbpo.agent_dps_root_similarity_enabled = False
            self.assertFalse(prepare_before_training(self.config))
            self.config.algorithm.el_tsbpo.agent_dps_root_similarity_enabled = True
            self.config.algorithm.el_tsbpo.allocation_backend = "llm"
            self.assertFalse(prepare_before_training(self.config))
            self.config.algorithm.adv_estimator = "gigpo"
            self.assertFalse(prepare_before_training(self.config))
            run.assert_not_called()

    def test_config_file_cli_validates_cache_without_model_forward(self):
        config_path = self.root / "final.yaml"
        OmegaConf.save(self.config, config_path)
        with patch("examples.el_tsbpo_trainer.prepare_task_similarity.load_task_texts", return_value=(self.texts, self.namespace)):
            main(["--config-file", str(config_path), "--device", "cpu", "--batch-size", "4"])

    def test_training_preflight_runs_before_ray_initialization(self):
        from verl.trainer.main_ppo import run_ppo

        order = []
        remote = SimpleNamespace(remote=lambda: SimpleNamespace(run=SimpleNamespace(remote=lambda config: "result")))
        with patch("verl.trainer.main_ppo.prepare_before_training", side_effect=lambda config: order.append("cache")), \
             patch("verl.trainer.main_ppo.ray.is_initialized", return_value=False), \
             patch("verl.trainer.main_ppo.ray.init", side_effect=lambda **kwargs: order.append("ray")), \
             patch("verl.trainer.main_ppo.ray.remote", return_value=lambda cls: remote), \
             patch("verl.trainer.main_ppo.ray.get"):
            run_ppo(self.config)
        self.assertEqual(order, ["cache", "ray"])

    def test_failed_cache_build_prevents_ray_initialization(self):
        from verl.trainer.main_ppo import run_ppo

        with patch("examples.el_tsbpo_trainer.prepare_task_similarity.subprocess.run", side_effect=subprocess.CalledProcessError(1, "encoder")), \
             patch("verl.trainer.main_ppo.ray.init") as init:
            with self.assertRaises(subprocess.CalledProcessError):
                run_ppo(self.config)
            init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
