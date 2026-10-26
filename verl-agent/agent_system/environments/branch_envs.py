# Copyright 2025 NTU / verl-agent team. Licensed under Apache-2.0.
"""Environment workers for replaying branched ALFWorld and WebShop trajectories."""

from typing import Any, Dict, List, Optional, Tuple

import ray


def _import_alfworld_deps():
    import textworld
    import textworld.gym
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredInfos,
    )
    from agent_system.environments.env_package.alfworld.envs import load_config_file

    return textworld, AlfredDemangler, AlfredInfos, load_config_file


def _import_webshop_env():

    import os
    import sys

    project_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'env_package',
        'webshop',
        'webshop',
    ))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from web_agent_site.envs import WebAgentTextEnv

    return WebAgentTextEnv


def _format_webshop_observation(observation: str, task: str) -> str:

    parts = str(observation or '').split(' [SEP] ')
    try:
        task_index = parts.index(task)
    except ValueError:
        return str(observation or '')
    return ' [SEP] '.join(f"'{part}'" for part in parts[task_index + 1:])


def _format_webshop_actions(available_actions: Dict[str, Any]) -> List[str]:
    available_actions = dict(available_actions or {})
    actions: List[str] = []
    if available_actions.get('has_search_bar', False):
        actions.append('search[<your query>]')
    actions.extend(
        f"click[{text}]" for text in available_actions.get('clickables', [])
    )
    return actions


def _format_webshop_admissible_actions(available_actions: Dict[str, Any]) -> List[str]:
    return [
        action for action in _format_webshop_actions(available_actions)
        if action.lower() != 'click[search]'
    ]


class BranchWebshopWorker:

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int,
        session_id: str,
    ):
        WebAgentTextEnv = _import_webshop_env()
        kwargs = dict(config.get('env_kwargs') or {})
        kwargs.pop('_shared_goal_order', None)
        catalog_seed = int(kwargs.pop('_goal_order_seed', seed))
        kwargs.pop('_max_steps', None)
        for name in list(kwargs):
            if name.startswith('_task_sampler_'):
                kwargs.pop(name)
        kwargs.update(seed=catalog_seed, auto_reset=False, retain_sessions=False)
        self.env = WebAgentTextEnv(**kwargs)
        self.seed = catalog_seed
        self.worker_seed = int(seed)
        self.max_steps = int(config.get('max_steps', 15))
        self.session_id = int(session_id)
        self.task = ''
        self._step_count = 0
        self._done = False
        self._last_observation = ''
        self._last_info = {}

    def configure(self, session_id: str, seed: int) -> bool:
        if int(seed) != self.worker_seed:
            raise ValueError(
                'WebShop replay seed changed inside a persistent worker: '
                f'{self.worker_seed} -> {int(seed)}'
            )
        self.session_id = int(session_id)
        return True

    @staticmethod
    def _extract_task(observation: str) -> str:
        parts = str(observation or '').split(' [SEP] ')
        if len(parts) < 3 or parts[1] != 'Instruction:':
            raise ValueError('WebShop root observation is missing its instruction')
        return parts[2]

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        observation, info = self.env.reset(session=self.session_id)
        self.task = self._extract_task(observation)
        available = self.env.get_available_actions()
        self._step_count = 0
        self._done = False
        self._last_observation = _format_webshop_observation(observation, self.task)
        self._last_info = {
            'won': False,
            'admissible_commands': _format_webshop_admissible_actions(available),
            'prompt_commands': _format_webshop_actions(available),
            'available_actions': available,
            'webshop.session_id': self.session_id,
            'webshop.seed': self.seed,
            'extra.gamefile': str(self.session_id),
        }
        return self._last_observation, dict(self._last_info)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done:
            return self._last_observation, 0.0, True, dict(self._last_info)
        observation, task_score, done, info = self.env.step(str(action))
        self._step_count += 1
        timed_out = bool(self._step_count >= self.max_steps and not done)
        done = bool(done or timed_out)
        won = bool(done and float(task_score) == 1.0)
        available = self.env.get_available_actions()
        info = dict(info or {})
        info.update({
            'won': won,
            'task_score': float(task_score),
            'executed_action': str(action),
            'available_actions': available,
            'admissible_commands': _format_webshop_admissible_actions(available),
            'prompt_commands': _format_webshop_actions(available),
            'webshop.session_id': self.session_id,
            'webshop.seed': self.seed,
            'extra.gamefile': str(self.session_id),
            'TimeLimit.truncated': timed_out,
        })
        self._done = done
        self._last_observation = _format_webshop_observation(observation, self.task)
        self._last_info = info
        reward = 10.0 if won else 0.0
        return self._last_observation, reward, done, dict(info)

    def replay(self, action_history: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        try:
            observation, info = self.reset()
            for action in action_history:
                observation, _, done, info = self.step(action)
                if done:
                    return observation, info, False
            return observation, info, True
        except Exception as exc:
            return '', {
                'won': False,
                'admissible_commands': [],
                'replay_error': str(exc),
            }, False

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self._step_count)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


class BranchAlfworldWorker:

    def __init__(self, config: Dict[str, Any], seed: int, gamefile: str):
        self.config = config
        self.max_steps = self._max_steps_from_config(config)
        self.env = None
        self.gamefile = ""
        self._configure(gamefile, seed)

    @staticmethod
    def _max_steps_from_config(config: Dict[str, Any]) -> int:

        training_method = config["general"]["training_method"]
        if training_method == "dagger":
            return int(config["dagger"]["training"]["max_nb_steps_per_episode"])
        if training_method == "dqn":
            return int(config["rl"]["training"]["max_nb_steps_per_episode"])
        raise NotImplementedError(f"Unknown training_method: {training_method}")

    def _configure(self, gamefile: str, seed: int) -> None:

        from agent_system.environments.env_package.alfworld.textworld_utils import (
            close_textworld_env, make_textworld_env,
        )

        if self.env is not None:
            close_textworld_env(self.env)
            self.env = None

        textworld, AlfredDemangler, AlfredInfos, _ = _import_alfworld_deps()


        domain_randomization = self.config["env"].get("domain_randomization", False)
        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)

        wrappers = [alfred_demangler, AlfredInfos]


        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )


        self.env = make_textworld_env(
            [gamefile],
            request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=self.max_steps,
            wrappers=wrappers,
        )
        self.env.seed(seed)
        self.gamefile = gamefile

        self._step_count = 0
        self._done = False

    def configure(self, gamefile: str, seed: int) -> bool:

        self._configure(str(gamefile), int(seed))
        return True


    def reset(self) -> Tuple[str, Dict[str, Any]]:
        obs, infos = self.env.reset()

        for k in list(infos.keys()):
            infos[k] = infos[k][0]
        infos["observation_text"] = obs[0]
        self._step_count = 0
        self._done = False
        return obs[0], infos

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done:

            return "", 0.0, True, {"won": False, "admissible_commands": []}
        obs, scores, dones, infos = self.env.step([action])


        done = bool(dones[0]) if isinstance(dones, tuple) else bool(dones)
        for k in list(infos.keys()):
            infos[k] = infos[k][0]
        infos["observation_text"] = obs[0]
        self._step_count += 1

        infos['TimeLimit.truncated'] = bool(
            infos.get('TimeLimit.truncated', False)
            or (self._step_count >= self.max_steps
                and not infos.get('won', False) and not infos.get('lost', False))
        )
        if self._step_count >= self.max_steps:
            self._done = True
            done = True
        else:
            self._done = done
        return obs[0], float(scores[0]), done, infos


    def replay(self, action_history: List[str]) -> Tuple[str, Dict[str, Any], bool]:
        try:
            obs, infos = self.reset()
        except Exception as e:
            import logging
            logging.error(f"[BranchAlfworldWorker.replay] reset failed: {e}, gamefile={self.gamefile}")
            return "", {"won": False, "admissible_commands": [], "replay_error": str(e)}, False

        for i, action in enumerate(action_history):
            try:
                obs, scores, dones, infos = self.step(action)
            except Exception as e:
                import logging
                logging.error(f"[BranchAlfworldWorker.replay] step {i+1} failed: action={action!r}, error={e}")
                return "", {"won": False, "admissible_commands": [], "replay_error": str(e)}, False
            if self._done:

                import logging
                logging.warning(
                    f"[BranchAlfworldWorker.replay] step {i+1}/{len(action_history)} ended early: "
                    f"action={action!r}, won={infos.get('won')}, step_count={self._step_count}/{self.max_steps}"
                )
                return obs, infos, False
        return obs, infos, True


    def get_admissible_commands(self) -> List[str]:


        return []

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self._step_count)

    def close(self):
        from agent_system.environments.env_package.alfworld.textworld_utils import close_textworld_env

        close_textworld_env(self.env)
        self.env = None


class BranchAlfworldEnvs:
    """Replay ALFWorld prefixes and run branch continuations."""

    def __init__(
        self,
        alf_config_path: str,
        resources_per_worker: Dict[str, Any],
        seed: int = 0,
        max_workers: int = 128,
    ):
        if not ray.is_initialized():
            ray.init()
        _, _, _, load_config_file = _import_alfworld_deps()
        self.config = load_config_file(alf_config_path)
        self.resources_per_worker = resources_per_worker
        self.seed = seed
        self.max_workers = max(1, int(max_workers))
        self._spawn_generation = 0

        self._worker_cls = ray.remote(**resources_per_worker)(BranchAlfworldWorker)


        self.workers: List[Optional[Any]] = []
        self._active: List[bool] = []
        self._gamefile_cache: List[Optional[str]] = []
        self._created_workers = 0
        self._reconfigured_workers = 0
        self._same_gamefile_reuses = 0

        self._admissible_cache: List[Optional[List[str]]] = []
        self._prompt_command_cache: List[Optional[List[str]]] = []


        self._observation_cache: List[Optional[str]] = []


    def spawn_branch(
        self,
        gamefile: str,
        action_history: List[str],
    ) -> int:
        return self.spawn_branches_parallel(
            [{"gamefile": gamefile, "action_history": action_history}]
        )[0]

    def _worker_seed(self, slot: int) -> int:
        return self.seed + int(slot) + self._spawn_generation * max(
            1, self.max_workers
        )

    def spawn_branches_parallel(
        self,
        branches: List[Dict[str, Any]],
    ) -> List[int]:
        if not branches:
            return []
        free_slots = [
            index for index, active in enumerate(self._active) if not active
        ]
        if len(branches) > self.max_workers:
            raise ValueError(
                "branch batch exceeds persistent worker-pool capacity: "
                f"requested={len(branches)}, max_workers={self.max_workers}. "
                "Chunk branch allocations by continuation count before spawning."
            )
        if len(branches) > len(free_slots) + max(
            0, self.max_workers - len(self.workers)
        ):
            raise RuntimeError(
                "branch worker pool has insufficient inactive capacity: "
                f"requested={len(branches)}, inactive={len(free_slots)}, "
                f"slots={len(self.workers)}, max_workers={self.max_workers}"
            )
        while len(free_slots) < len(branches):
            slot = len(self.workers)
            self.workers.append(None)
            self._active.append(False)
            self._gamefile_cache.append(None)
            self._admissible_cache.append(None)
            self._prompt_command_cache.append(None)
            self._observation_cache.append(None)
            free_slots.append(slot)


        remaining_slots = set(free_slots)
        assigned_slots: List[Optional[int]] = [None] * len(branches)
        for branch_index, branch in enumerate(branches):
            gamefile = str(branch["gamefile"])
            matching = next(
                (
                    slot
                    for slot in sorted(remaining_slots)
                    if self._gamefile_cache[slot] == gamefile
                ),
                None,
            )
            if matching is not None:
                assigned_slots[branch_index] = matching
                remaining_slots.remove(matching)
        for branch_index, assigned in enumerate(assigned_slots):
            if assigned is not None:
                continue
            uninitialized = next(
                (
                    slot
                    for slot in sorted(remaining_slots)
                    if self.workers[slot] is None
                ),
                None,
            )
            slot = (
                uninitialized
                if uninitialized is not None
                else min(remaining_slots)
            )
            assigned_slots[branch_index] = slot
            remaining_slots.remove(slot)

        self._spawn_generation += 1
        replay_futures = []
        for i, br in enumerate(branches):
            slot = int(assigned_slots[i])
            worker = self.workers[slot]
            gamefile = str(br["gamefile"])
            branch_seed = self._worker_seed(slot)
            if worker is None:
                worker = self._worker_cls.remote(
                    self.config, branch_seed, gamefile
                )
                self.workers[slot] = worker
                self._gamefile_cache[slot] = gamefile
                self._created_workers += 1
            elif self._gamefile_cache[slot] != gamefile:


                worker.configure.remote(gamefile, branch_seed)
                self._gamefile_cache[slot] = gamefile
                self._reconfigured_workers += 1
            else:
                self._same_gamefile_reuses += 1
            self._active[slot] = True
            self._admissible_cache[slot] = None
            self._prompt_command_cache[slot] = None
            self._observation_cache[slot] = None
            replay_futures.append((slot, worker.replay.remote(br["action_history"])))

        results = ray.get([future for _, future in replay_futures])
        indices: List[int] = []
        for i, ((worker_idx, _), (obs, infos, ok)) in enumerate(
            zip(replay_futures, results)
        ):
            if ok:
                self._admissible_cache[worker_idx] = list(
                    infos.get("admissible_commands", [])
                )
                self._prompt_command_cache[worker_idx] = list(
                    infos.get("prompt_commands", infos.get("admissible_commands", []))
                )
                self._observation_cache[worker_idx] = str(obs or "")
                indices.append(worker_idx)
            else:


                import logging
                err = infos.get("replay_error", "ended early") if infos else "unknown"
                won = infos.get("won") if infos else None
                logging.warning(
                    f"[BranchAlfworldEnvs] branch worker {worker_idx} replay failed: "
                    f"reason={err}, won={won}, gamefile={branches[i].get('gamefile', '?')}, "
                    f"action_history_len={len(branches[i].get('action_history', []))}"
                )
                self._active[worker_idx] = False
                self._admissible_cache[worker_idx] = None
                self._prompt_command_cache[worker_idx] = None
                self._observation_cache[worker_idx] = None
                indices.append(-1)
        return indices


    def submit_step_batch(
        self,
        worker_indices: List[int],
        actions: List[str],
    ) -> List[Dict[str, Any]]:
        assert len(worker_indices) == len(actions), (
            "worker_indices and actions must have same length"
        )
        requests: List[Dict[str, Any]] = []
        for worker_idx, action in zip(worker_indices, actions):
            if not self._is_alive(worker_idx):
                continue
            requests.append({
                "worker_idx": int(worker_idx),
                "action": str(action),
                "future": self.workers[worker_idx].step.remote(action),
            })
        return requests

    def wait_step_requests(
        self,
        requests: List[Dict[str, Any]],
        *,
        num_returns: int = 1,
        timeout: Optional[float] = None,
        wait_all: bool = False,
        wait_for_target: bool = False,
        drain_ready: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not requests:
            return [], []
        refs = [request["future"] for request in requests]
        target = max(1, min(int(num_returns), len(refs)))
        if wait_all or wait_for_target:


            ready_refs, pending_refs = ray.wait(
                refs, num_returns=target, timeout=timeout
            )
        elif not wait_for_target:


            ready_refs, pending_refs = ray.wait(
                refs, num_returns=1, timeout=timeout
            )
        if not ready_refs:
            return [], requests
        if drain_ready and not wait_all and pending_refs:
            extra_ready, _ = ray.wait(
                pending_refs,
                num_returns=len(pending_refs),
                timeout=0.0,
            )
            ready_refs.extend(extra_ready)
        elif not wait_all and not wait_for_target and target > 1 and pending_refs:
            extra_ready, _ = ray.wait(
                pending_refs,
                num_returns=min(target - 1, len(pending_refs)),
                timeout=0.0,
            )
            ready_refs.extend(extra_ready)
        ready_set = set(ready_refs)
        ready_requests = [
            request for request in requests if request["future"] in ready_set
        ]


        raw_results = ray.get([request["future"] for request in ready_requests])
        completed: List[Dict[str, Any]] = []
        for request, raw in zip(ready_requests, raw_results):
            obs, score, dones, info = raw
            worker_idx = int(request["worker_idx"])
            action = request["action"]


            reward = float(score)
            if isinstance(info, dict):
                info.setdefault("executed_action", action)
                admissible = info.get("admissible_commands", []) or []
            else:
                admissible = []
            self._admissible_cache[worker_idx] = list(admissible)
            self._prompt_command_cache[worker_idx] = list(
                info.get("prompt_commands", admissible) if isinstance(info, dict)
                else admissible
            )
            self._observation_cache[worker_idx] = str(obs or "")
            completed.append({
                "worker_idx": worker_idx,
                "action": action,
                "result": (obs, reward, bool(dones), info),
            })
        remaining = [
            request for request in requests if request["future"] not in ready_set
        ]
        return completed, remaining

    def step_batch(
        self,
        worker_indices: List[int],
        actions: List[str],
    ) -> List[Tuple[Optional[str], float, bool, Optional[Dict[str, Any]]]]:
        assert len(worker_indices) == len(actions), \
            "worker_indices and actions must have same length"

        futures = []
        future_positions: List[int] = []
        for i, widx in enumerate(worker_indices):
            if not self._is_alive(widx):
                continue
            futures.append(self.workers[widx].step.remote(actions[i]))
            future_positions.append(i)

        results = ray.get(futures) if futures else []

        out: List[Tuple[Optional[str], float, bool, Optional[Dict[str, Any]]]] = [
            (None, 0.0, True, None) for _ in worker_indices
        ]
        for pos, (obs, score, dones, info) in zip(future_positions, results):
            reward = float(score)
            if isinstance(info, dict):


                info.setdefault("executed_action", actions[pos])
            self._admissible_cache[worker_indices[pos]] = list(
                info.get("admissible_commands", [])
            )
            self._prompt_command_cache[worker_indices[pos]] = list(
                info.get("prompt_commands", info.get("admissible_commands", []))
            )
            self._observation_cache[worker_indices[pos]] = str(obs or "")
            out[pos] = (obs, reward, bool(dones), info)
        return out


    def get_admissible_commands_batch(
        self,
        worker_indices: List[int],
    ) -> List[List[str]]:
        out: List[List[str]] = []
        for widx in worker_indices:
            if not self._is_alive(widx):
                out.append([])
            else:
                cached = self._admissible_cache[widx]
                out.append(list(cached) if cached is not None else [])
        return out

    def get_observations_batch(self, worker_indices: List[int]) -> List[str]:

        return [
            str(self._observation_cache[widx] or "")
            if self._is_alive(widx)
            else ""
            for widx in worker_indices
        ]

    def get_prompt_commands_batch(self, worker_indices: List[int]) -> List[List[str]]:
        return [
            list(self._prompt_command_cache[widx] or [])
            if self._is_alive(widx)
            else []
            for widx in worker_indices
        ]

    def get_remaining_steps_batch(self, worker_indices: List[int]) -> List[int]:
        futures = []
        positions = []
        for i, widx in enumerate(worker_indices):
            if not self._is_alive(widx):
                continue
            futures.append(self.workers[widx].remaining_steps.remote())
            positions.append(i)
        results = ray.get(futures) if futures else []
        out = [0 for _ in worker_indices]
        for pos, rs in zip(positions, results):
            out[pos] = int(rs)
        return out


    def close_branch(self, worker_idx: int):
        if 0 <= worker_idx < len(self.workers) and self.workers[worker_idx] is not None:
            self._active[worker_idx] = False
            self._admissible_cache[worker_idx] = None
            self._prompt_command_cache[worker_idx] = None
            self._observation_cache[worker_idx] = None

    def close_all(self):
        workers = [worker for worker in self.workers if worker is not None]
        close_futures = []
        for worker in workers:
            try:
                close_futures.append(worker.close.remote())
            except Exception:
                pass
        if close_futures:
            try:
                ray.get(close_futures, timeout=30)
            except Exception:
                pass
        for i in range(len(self.workers)):
            if self.workers[i] is not None:
                try:
                    ray.kill(self.workers[i], no_restart=True)
                except Exception:
                    pass
                self.workers[i] = None
                self._active[i] = False
                self._gamefile_cache[i] = None
                self._admissible_cache[i] = None
                self._prompt_command_cache[i] = None
                self._observation_cache[i] = None


    def _is_alive(self, widx: int) -> bool:
        return (
            0 <= widx < len(self.workers)
            and self.workers[widx] is not None
            and self._active[widx]
        )

    @property
    def num_alive(self) -> int:

        return sum(
            1
            for worker in self.workers
            if worker is not None
        )

    @property
    def num_active(self) -> int:

        return sum(
            1
            for worker, active in zip(self.workers, self._active)
            if worker is not None and active
        )

    @property
    def num_total_slots(self) -> int:
        return len(self.workers)

    @property
    def num_created_workers(self) -> int:
        return self._created_workers

    @property
    def num_reconfigured_workers(self) -> int:
        return self._reconfigured_workers

    @property
    def num_same_gamefile_reuses(self) -> int:
        return self._same_gamefile_reuses


class BranchWebshopEnvs(BranchAlfworldEnvs):
    """Replay WebShop prefixes and run branch continuations."""

    def __init__(
        self,
        env_kwargs: Dict[str, Any],
        resources_per_worker: Dict[str, Any],
        seed: int = 0,
        max_workers: int = 128,
        max_steps: int = 15,
    ):
        if not ray.is_initialized():
            ray.init()
        self.config = {
            'env_kwargs': dict(env_kwargs or {}),
            'max_steps': int(max_steps),
        }
        self.env_kwargs = dict(env_kwargs or {})
        self.resources_per_worker = resources_per_worker
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.max_workers = max(1, int(max_workers))
        self._spawn_generation = 0
        self._worker_cls = ray.remote(**resources_per_worker)(BranchWebshopWorker)
        self.workers = []
        self._active = []
        self._gamefile_cache = []
        self._created_workers = 0
        self._reconfigured_workers = 0
        self._same_gamefile_reuses = 0
        self._admissible_cache = []
        self._prompt_command_cache = []
        self._observation_cache = []

    def _worker_seed(self, slot: int) -> int:


        return self.seed
