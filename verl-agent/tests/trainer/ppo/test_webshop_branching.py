"""CPU regressions for WebShop tree-rollout replay integration."""

import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop.projection import (
    webshop_projection,
)
from agent_system.environments.prompts.webshop import (
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
)
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.environments.branch_envs import (
    BranchWebshopEnvs,
    _format_webshop_actions,
    _format_webshop_observation,
)


class _AllocatedWebshopBackend:
    def __init__(self):
        self.received = None
        self.received_actions = None
        self._admissible = []

    def reset_allocated_sessions(self, session_ids):
        self.received = list(session_ids)
        observations = [
            f"WebShop [SEP] Instruction: [SEP] task-{session_id} [SEP] Search"
            for session_id in session_ids
        ]
        infos = [
            {
                "webshop.session_id": session_id,
                "available_actions": {
                    "has_search_bar": True,
                    "clickables": ["search"],
                },
            }
            for session_id in session_ids
        ]
        self._admissible = [
            ["search[<your query>]", "click[search]"]
            for _ in session_ids
        ]
        return observations, infos

    @property
    def get_admissible_commands(self):
        return self._admissible

    def step(self, actions):
        self.received_actions = list(actions)
        observations = [
            f"WebShop [SEP] Instruction: [SEP] task-{session_id} [SEP] Results"
            for session_id in self.received
        ]
        infos = [
            {
                "webshop.session_id": session_id,
                "available_actions": {
                    "has_search_bar": False,
                    "clickables": ["B001"],
                },
            }
            for session_id in self.received
        ]
        self._admissible = [["click[B001]"] for _ in self.received]
        return observations, [0.0] * len(actions), [False] * len(actions), infos


class TestWebshopBranching(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create({
            "env": {
                "env_name": "Webshop",
                "history_length": 5,
            }
        })

    def test_allocated_roots_are_pinned_to_sessions(self):
        backend = _AllocatedWebshopBackend()
        manager = WebshopEnvironmentManager(backend, lambda actions: (actions, [1]), self.config)

        observations, _ = manager.reset_allocated_roots([501, 777])

        self.assertEqual(backend.received, [501, 777])
        self.assertEqual(manager.replay_task_refs, ["501", "777"])
        self.assertEqual(manager.gamefile, ["501", "777"])
        self.assertEqual(len(observations["text"]), 2)

    def test_root_candidates_use_the_same_anchor_as_allocated_reset(self):
        backend = _AllocatedWebshopBackend()
        manager = WebshopEnvironmentManager(
            backend, lambda actions: (actions, [1]), self.config
        )
        raw = [
            "WebShop [SEP] Instruction: [SEP] task-501 [SEP] Search",
            "WebShop [SEP] Instruction: [SEP] task-777 [SEP] Results",
        ]
        infos = [
            {"webshop.session_id": 501, "extra.gamefile": "501"},
            {"webshop.session_id": 777, "extra.gamefile": "777"},
        ]

        canonical, returned_infos = manager.canonicalize_root_candidates(
            raw, infos
        )

        self.assertEqual(canonical, ["'Search'", "'Results'"])
        self.assertEqual(returned_infos, infos)
        # Canonicalizing a candidate pool must not overwrite the active
        # manager task state used by the live rollout.
        self.assertFalse(hasattr(manager, "tasks"))

    def test_branch_observation_and_actions_match_manager_format(self):
        raw = "WebShop [SEP] Instruction: [SEP] buy a red mug [SEP] Search [SEP] Results"
        self.assertEqual(
            _format_webshop_observation(raw, "buy a red mug"),
            "'Search' [SEP] 'Results'",
        )
        self.assertEqual(
            _format_webshop_actions({
                "has_search_bar": True,
                "clickables": ["search", "B001"],
            }),
            ["search[<your query>]", "click[B001]"],
        )

    def test_branch_prompt_uses_webshop_template(self):
        collector = SimpleNamespace(config=self.config)
        collector.build = TrajectoryCollector._build_branch_prompt.__get__(collector)
        memory = [{"observation": "old", "action": "search[mug]"}]
        prompt = collector.build(
            task="buy a red mug",
            memory_steps=memory,
            current_obs="results",
            admissible_commands=["click[B001]"],
            history_length=5,
        )
        expected = WEBSHOP_TEMPLATE.format(
            task_description="buy a red mug",
            step_count=1,
            history_length=1,
            action_history="[Observation 1: 'old', Action 1: 'search[mug]']",
            current_step=2,
            current_observation="results",
            available_actions="'click[B001]',",
        )
        self.assertEqual(prompt, expected)

        root_prompt = collector.build(
            task="buy a red mug",
            memory_steps=[],
            current_obs="search",
            admissible_commands=["search[<your query>]"],
        )
        self.assertEqual(
            root_prompt,
            WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description="buy a red mug",
                current_observation="search",
                available_actions="'search[<your query>]',",
            ),
        )

    def test_branch_projection_uses_webshop_semantics(self):
        collector = SimpleNamespace(config=self.config)
        collector.project = TrajectoryCollector._project_branch_actions.__get__(collector)
        actions, valids, details = collector.project(
            [
                "<think>query</think><action>search[red mug]</action>",
                "<think>open</think><action>click[B001]</action>",
            ],
            [
                ["search[<your query>]", "click[search]"],
                ["click[B001]"],
            ],
        )
        self.assertEqual(actions, ["search[red mug]", "click[b001]"])
        self.assertEqual(valids, [1, 1])
        self.assertEqual(details["admissible_valid"], [True, True])

    def test_projection_separates_format_and_page_admissibility(self):
        actions, valids, details = webshop_projection(
            [
                "<think>query</think><action>search[red mug]</action>",
                "<think>open</think><action>click[missing]</action>",
                "<think>query</think><action>search[blue mug]</action>tail",
                "<think>a</think><think>b</think><action>click[B001]</action>",
            ],
            [
                ["search[<your query>]"],
                ["click[B001]"],
                ["search[<your query>]"],
                ["click[B001]"],
            ],
            return_details=True,
        )

        self.assertEqual(
            actions,
            ["search[red mug]", "click[missing]", "search[blue mug]", "click[b001]"],
        )
        self.assertEqual(valids, [1, 0, 0, 0])
        self.assertEqual(details["format_valid"], [1, 1, 0, 0])
        self.assertEqual(details["admissible_valid"], [1, 0, 1, 1])

    def test_factual_rollout_uses_the_same_strict_projection(self):
        backend = _AllocatedWebshopBackend()
        manager = WebshopEnvironmentManager(
            backend, webshop_projection, self.config
        )
        manager.reset_allocated_roots([501])

        _, _, _, infos = manager.step([
            "<think>query</think><action>search[red mug]</action>tail"
        ])

        self.assertEqual(backend.received_actions, ["search[red mug]"])
        self.assertFalse(bool(infos[0]["is_action_valid"]))
        self.assertFalse(bool(infos[0]["is_action_format_valid"]))
        self.assertTrue(bool(infos[0]["is_action_admissible"]))
        self.assertEqual(infos[0]["executed_action"], "search[red mug]")

    def test_branch_pool_keeps_one_goal_order_seed(self):
        pool = BranchWebshopEnvs.__new__(BranchWebshopEnvs)
        pool.seed = 23
        pool._spawn_generation = 100
        pool.max_workers = 64
        self.assertEqual(pool._worker_seed(0), 23)
        self.assertEqual(pool._worker_seed(63), 23)


if __name__ == "__main__":
    unittest.main()
