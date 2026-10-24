# Copyright 2025 NTU / verl-agent team. Licensed under Apache-2.0.
"""WebShop workers and resumable task references for Graph-GiGPO/Agent-DPS."""

import os
import sys
from functools import partial
from types import SimpleNamespace

import gym
import numpy as np
import ray

from .task_sampler import WebshopTaskSampler


def _admissible_actions(available):
    actions = ['search[<your query>]'] if available.get('has_search_bar') else []
    return actions + [f'click[{text}]' for text in available.get('clickables', [])
                      if str(text).strip().lower() != 'search']


class WebshopWorker:
    def __init__(self, seed, env_kwargs):
        project_root = os.path.join(os.path.dirname(__file__), 'webshop')
        if project_root not in sys.path:
            sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

        kwargs = dict(env_kwargs or {})
        shared = bool(kwargs.pop('_shared_goal_order', True))
        catalog_seed = int(kwargs.pop('_goal_order_seed', seed))
        self.max_steps = int(kwargs.pop('_max_steps', 0))
        kwargs.pop('_workers_per_process', None)
        shared_server = kwargs.pop('_shared_server', None)
        for name in list(kwargs):
            if name.startswith('_task_sampler_'):
                kwargs.pop(name)
        self.seed = catalog_seed if shared else int(seed)
        kwargs.update(seed=self.seed, auto_reset=False, retain_sessions=False)
        if shared_server is not None:
            kwargs['server'] = shared_server
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **kwargs)
        self._done = False
        self._step_count = 0
        self._last_observation = ''
        self._last_info = {}

    def _info(self, info=None, *, score=0.0, won=False):
        available = self.env.get_available_actions()
        result = dict(info or {})
        result.update({
            'available_actions': available,
            'admissible_commands': _admissible_actions(available),
            'task_score': float(score), 'won': bool(won),
            'webshop.session_id': int(self.env.unwrapped.session),
            'webshop.seed': self.seed,
            'extra.gamefile': str(self.env.unwrapped.session),
        })
        return result

    def reset(self, idx):
        idx = int(idx)
        if not 0 <= idx < len(self.get_goals()):
            raise ValueError(f'WebShop session out of range: {idx}')
        obs, info = self.env.reset(session=idx)
        self._step_count = 0
        self._done = False
        self._last_observation = obs
        self._last_info = self._info(info)
        return obs, dict(self._last_info)

    def step(self, action):
        if self._done:
            return self._last_observation, 0.0, True, dict(self._last_info)
        obs, score, done, info = self.env.step(str(action))
        self._step_count += 1
        timed_out = bool(self.max_steps > 0 and self._step_count >= self.max_steps and not done)
        done = bool(done or timed_out)
        won = bool(done and float(score) == 1.0)
        info = self._info(info, score=score, won=won)
        info['executed_action'] = str(action)
        info['TimeLimit.truncated'] = timed_out
        self._done, self._last_observation, self._last_info = done, obs, info
        return obs, 10.0 if won else 0.0, done, dict(info)

    def render(self, mode_for_render):
        return self.env.render(mode=mode_for_render)

    def get_available_actions(self):
        return self.env.get_available_actions()

    def get_goals(self):
        return self.env.unwrapped.server.goals

    def get_runtime_stats(self):
        """Inspect actor memory and live sessions without copying the catalog."""
        import psutil
        memory = psutil.Process().memory_full_info()
        return {
            'pid': os.getpid(), 'rss_bytes': memory.rss,
            'uss_bytes': memory.uss, 'catalog_seed': self.seed,
            'live_sessions': len(self.env.unwrapped.server.user_sessions),
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self._last_info = {}
        self._last_observation = ''
        return True


class WebshopWorkerPool:
    """Serial actor with independent episodes and one immutable catalog/index.

    Each slot owns its browser, histories, counters and session dictionary.
    Switching the server's session dictionary isolates repeated task IDs too.
    Ray executes this synchronous actor's calls serially.
    """

    def __init__(self, seed, env_kwargs, slots):
        self.workers, self.sessions = [], []
        self.server = None
        try:
            for index in range(int(slots)):
                kwargs = dict(env_kwargs)
                if self.server is not None:
                    self.server.user_sessions = {}
                    kwargs['_shared_server'] = self.server
                worker = WebshopWorker(seed + index, kwargs)
                self.workers.append(worker)
                self.server = worker.env.unwrapped.server
                self.sessions.append(self.server.user_sessions)
        except Exception:
            self.close()
            raise

    def call_slot(self, slot, method, *args, **kwargs):
        if method not in {'reset', 'step', 'render', 'get_available_actions',
                          'get_goals', 'get_runtime_stats'}:
            raise ValueError(f'Unsupported WebShop slot method: {method}')
        slot = int(slot)
        self.server.user_sessions = self.sessions[slot]
        try:
            result = getattr(self.workers[slot], method)(*args, **kwargs)
            if method == 'get_runtime_stats':
                result['logical_slots'] = len(self.workers)
            return result
        finally:
            self.sessions[slot] = self.server.user_sessions

    def get_goals(self):
        return self.server.goals

    def close(self):
        # Close borrowers before the slot owning the shared Lucene searcher.
        for index in reversed(range(len(self.workers))):
            self.server.user_sessions = self.sessions[index]
            self.workers[index].close()
        self.workers, self.sessions = [], []
        return True


class _WebshopWorkerSlot:
    """Keep the one-episode worker RPC interface for a slot in a pooled actor."""

    def __init__(self, actor, slot):
        self._actor, self._slot = actor, int(slot)

    def __getattr__(self, method):
        if method not in {'reset', 'step', 'render', 'get_available_actions',
                          'get_goals', 'get_runtime_stats'}:
            raise AttributeError(method)
        return SimpleNamespace(remote=partial(self._actor.call_slot.remote, self._slot, method))


class WebshopMultiProcessEnv(gym.Env):
    def __init__(self, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs=None):
        super().__init__()
        self._closed = False
        self._workers = []
        self._physical_workers = []
        self.env_num, self.group_n = int(env_num), int(group_n)
        if self.env_num <= 0 or self.group_n <= 0 or (not is_train and self.group_n != 1):
            raise ValueError('WebShop needs positive worker counts; validation group_n must be 1')
        self.worker_capacity = self.env_num * self.group_n
        self.num_processes = self.worker_capacity
        self.num_envs = self.worker_capacity
        self.is_train = bool(is_train)
        self._rng = np.random.RandomState(int(seed))
        self._env_kwargs = dict(env_kwargs or {'observation_mode': 'text', 'num_products': 1000,
                                              'human_goals': False})
        self._env_kwargs.setdefault('_shared_goal_order', True)
        self._env_kwargs.setdefault('_goal_order_seed', int(seed))
        workers_per_process = int(self._env_kwargs.get('_workers_per_process', 1))
        if workers_per_process <= 0:
            raise ValueError('WebShop workers_per_process must be positive')
        if not self._env_kwargs['_shared_goal_order']:
            raise ValueError('WebShop task references require shared goal ordering across workers')
        if not ray.is_initialized():
            ray.init()
        actor = ray.remote(**resources_per_worker)(
            WebshopWorkerPool if workers_per_process > 1 else WebshopWorker)
        try:
            for index in range(0, self.worker_capacity, workers_per_process):
                actor_seed = seed + index // self.group_n
                if workers_per_process == 1:
                    physical = actor.remote(actor_seed, self._env_kwargs)
                    self._workers.append(physical)
                else:
                    slots = min(workers_per_process, self.worker_capacity - index)
                    physical = actor.remote(actor_seed, self._env_kwargs, slots)
                    self._workers.extend(_WebshopWorkerSlot(physical, slot) for slot in range(slots))
                self._physical_workers.append(physical)
            goals = ray.get(self._workers[0].get_goals.remote())
            self.goal_queries = [str(goal.get('query', '')) for goal in goals]
            self.goal_instructions = [str(goal.get('instruction_text', '')) for goal in goals]
            self.goal_idxs = range(500, len(goals)) if self.is_train else range(min(500, len(goals)))
            if not self.goal_idxs:
                raise ValueError('WebShop training split is empty; use synthetic goals (human_goals=False)')
            mode = str(self._env_kwargs.get('_task_sampler_mode', 'query_stratified'))
            if mode not in {'random_pool', 'shuffled_pool', 'query_stratified'}:
                raise ValueError(f'Unknown WebShop task sampler: {mode}')
            sampler_seed = self._env_kwargs.get('_task_sampler_seed')
            # Validation pins test sessions explicitly (or samples with _rng).
            # It does not need the training-only task curriculum module.
            self._task_sampler = None
            if self.is_train:
                self._task_sampler = WebshopTaskSampler(
                    goals, self.goal_idxs, seed=int(seed if sampler_seed is None else sampler_seed),
                    query_slots=self._env_kwargs.get('_task_sampler_query_slots', 8) if mode == 'query_stratified' else 0,
                    natural_slots=self._env_kwargs.get('_task_sampler_natural_slots', 8) if mode == 'query_stratified' else 0,
                    replacement_across_batches=mode == 'random_pool',
                )
            self._last_infos = [{} for _ in range(self.num_processes)]
        except Exception:
            self.close()
            raise

    def _sessions(self, session_ids):
        result = [int(value) for value in session_ids]
        allowed = set(self.goal_idxs)
        if any(sid not in allowed for sid in result):
            raise ValueError('WebShop session is outside the current train/validation split')
        return result

    def _reset_sessions(self, session_ids):
        session_ids = self._sessions(session_ids)
        if not 0 < len(session_ids) <= self.worker_capacity:
            raise ValueError('WebShop allocated root budget exceeds worker capacity or is empty')
        results = ray.get([self._workers[index].reset.remote(sid)
                           for index, sid in enumerate(session_ids)])
        self.num_processes = self.num_envs = len(session_ids)
        self._last_infos = [dict(info or {}) for _, info in results]
        return [obs for obs, _ in results], [dict(info) for info in self._last_infos]

    def reset(self, session_ids=None):
        if session_ids is None:
            selected = (self._task_sampler.sample(self.env_num) if self.is_train else
                        self._rng.choice(list(self.goal_idxs), size=self.env_num, replace=False).tolist())
            session_ids = np.repeat(selected, self.group_n).tolist()
        elif len(session_ids) == self.env_num and self.group_n > 1:
            session_ids = np.repeat(session_ids, self.group_n).tolist()
        return self._reset_sessions(session_ids)

    def reset_allocated_sessions(self, session_ids):
        return self._reset_sessions(session_ids)

    def sample_root_task_refs(self, n_candidates, exclude_session_ids=None, exclude_gamefiles=None):
        excluded = list(exclude_session_ids or []) + list(exclude_gamefiles or [])
        # Sampling references creates no simulator sessions or actor RPCs.
        return [str(sid) for sid in self._task_sampler.sample(int(n_candidates), excluded)]

    def get_root_task_texts(self, task_refs):
        """Read pure instructions without resetting sessions or creating actors."""
        sessions = self._sessions(task_refs)
        if any(not self.goal_instructions[sid] for sid in sessions):
            raise ValueError('WebShop goal lacks a pure task instruction')
        return {str(sid): self.goal_instructions[sid] for sid in sessions}

    def load_root_candidates(self, session_ids):
        session_ids = self._sessions(session_ids)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError('WebShop root candidates must be unique')
        if len(session_ids) * self.group_n > self.worker_capacity:
            raise ValueError('Selected WebShop roots exceed task capacity')
        results = ray.get([self._workers[index * self.group_n].reset.remote(sid)
                           for index, sid in enumerate(session_ids)])
        return [obs for obs, _ in results], [dict(info or {}) for _, info in results]

    def sample_root_candidates(self, n_candidates, exclude_session_ids=None, exclude_gamefiles=None):
        return self.load_root_candidates(self.sample_root_task_refs(
            n_candidates, exclude_session_ids, exclude_gamefiles))

    def sample_fresh_root_candidates(self, n_candidates, exclude_session_ids=None, exclude_gamefiles=None):
        return self.sample_root_candidates(n_candidates, exclude_session_ids, exclude_gamefiles)

    def fresh_task_sampler_state_dict(self):
        return self._task_sampler.state_dict()

    def load_fresh_task_sampler_state_dict(self, state):
        self._task_sampler.load_state_dict(state)

    def fresh_task_sampler_metrics(self):
        return self._task_sampler.metrics()

    def step(self, actions):
        if len(actions) != self.num_processes:
            raise ValueError('WebShop actions must match the active worker count')
        return self.step_active(list(range(self.num_processes)), actions)

    def step_active(self, indices, actions):
        indices = [int(value) for value in indices]
        if len(indices) != len(actions) or len(indices) != len(set(indices)):
            raise ValueError('WebShop active indices must be unique and match actions')
        if any(index < 0 or index >= self.num_processes for index in indices):
            raise IndexError('Active WebShop worker index out of range')
        results = ray.get([self._workers[index].step.remote(action)
                           for index, action in zip(indices, actions)])
        for index, (_, _, _, info) in zip(indices, results):
            self._last_infos[index] = dict(info or {})
        return tuple([[result[column] for result in results] for column in range(4)])

    @property
    def get_admissible_commands(self):
        return [list(info.get('admissible_commands', [])) for info in self._last_infos]

    def render(self, mode='text', env_idx=None):
        if env_idx is not None:
            return ray.get(self._workers[int(env_idx)].render.remote(mode))
        return ray.get([worker.render.remote(mode) for worker in self._workers[:self.num_processes]])

    def close(self):
        if self._closed:
            return
        self._closed = True
        workers, self._physical_workers = self._physical_workers, []
        self._workers = []
        futures = []
        for worker in workers:
            try:
                futures.append(worker.close.remote())
            except Exception:
                pass
        if futures:
            try:
                ray.get(futures, timeout=30)
            except Exception:
                pass
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass

    def __del__(self):
        if hasattr(self, '_closed'):
            self.close()


def build_webshop_envs(seed, env_num, group_n, resources_per_worker,
                       is_train=True, env_kwargs=None):
    return WebshopMultiProcessEnv(seed, env_num, group_n, resources_per_worker,
                                 is_train=is_train, env_kwargs=env_kwargs)
