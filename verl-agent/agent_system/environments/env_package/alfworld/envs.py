# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from copy import copy
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
from agent_system.environments.task_sampler import ShuffledTaskPool
from agent_system.environments.env_package.alfworld.textworld_utils import close_textworld_env
from agent_system.environments.env_package.alfworld.task_texts import root_task_texts

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
        self.base_env = base_env
        self.seed = seed
        self._root_env = None
        self._root_gamefile = None
        self._active_env = self.env
        method = config['general']['training_method']
        section = 'rl' if method == 'dqn' else method
        self.max_steps = int(config[section]['training']['max_nb_steps_per_episode'])
        self._step_count = 0
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self._active_env.step(actions)
        self._step_count += 1
        won = bool(infos.get('won', [False])[0])
        lost = bool(infos.get('lost', [False])[0])
        # TextWorld folds its TimeLimit into `done`. Keep that administrative
        # deadline separate from absorbing failure for the branch likelihood.
        infos['TimeLimit.truncated'] = [bool(
            not (won or lost) and (
                infos.get('TimeLimit.truncated', [False])[0]
                or self._step_count >= self.max_steps
            )
        )]
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self, gamefile=None):
        """Reset the environment"""
        if gamefile is None:
            # Keep the original game sampler separate from pinned root
            # rollouts; candidate sampling advances it only once per round.
            self._active_env = self.env
        else:
            gamefile = str(gamefile)
            if self._root_env is None or gamefile != self._root_gamefile:
                if self._root_env is not None:
                    close_textworld_env(self._root_env)
                    self._root_env = None
                pinned_base = copy(self.base_env)
                pinned_base.game_files = [gamefile]
                pinned_base.num_games = 1
                self._root_env = pinned_base.init_env(batch_size=1)
                self._root_env.seed(self.seed)
                self._root_gamefile = gamefile
            self._active_env = self._root_env
        obs, infos = self._active_env.reset()
        self._step_count = 0
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self._active_env)
        image = image.cpu()  
        return image

    def close(self):
        """Close TextWorld environments before this Ray actor exits."""
        environments = []
        for environment in (self._root_env, self.env):
            if environment is not None and all(
                environment is not existing for existing in environments
            ):
                environments.append(environment)
        for environment in environments:
            try:
                close_textworld_env(environment)
            except Exception:
                pass
        self._root_env = None
        self._active_env = None
        self.env = None
        return True

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.worker_capacity = self.num_processes
        self.group_n = group_n
        # The Agent-DPS root phase needs explicit, replayable task identities.
        # Keep a seeded candidate pool over ALFWorld's solvable game files so
        # candidate sampling is reproducible and unique within one batch.
        self._candidate_game_files = sorted(
            str(gamefile) for gamefile in getattr(base_env, 'game_files', [])
        )
        self._candidate_rng = np.random.RandomState(int(seed))
        self._fresh_root_pool = ShuffledTaskPool(
            self._candidate_game_files,
            seed=int(seed),
        )

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        indices = list(range(self.num_processes))
        return self._step_indices(indices, actions, full_output=True)

    def step_active(self, indices, actions):
        """Step only the selected workers and return sparse results.

        The rollout manager expands these results back to the fixed worker
        layout. Keeping the Ray call sparse is important: completed ALFWorld
        episodes must not receive empty actions merely to satisfy the legacy
        vectorized API.
        """
        indices = [int(index) for index in indices]
        if len(indices) != len(actions):
            raise ValueError("active worker indices and actions must have equal length")
        if any(index < 0 or index >= self.num_processes for index in indices):
            raise IndexError("active worker index out of range")
        return self._step_indices(indices, actions, full_output=False)

    def _step_indices(self, indices, actions, full_output):
        if len(indices) != len(actions):
            raise ValueError("worker indices and actions must have equal length")

        # Send step commands to all workers
        futures = []
        for index, action in zip(indices, actions):
            future = self.workers[index].step.remote(action)
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for index, (obs, scores, dones, info) in zip(indices, results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[index] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs(indices)
        else:
            image_obs_list = None

        if not full_output:
            return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def get_root_task_texts(self, task_refs):
        """Read pure instructions without simulator resets or worker RPCs."""
        if self.multi_modal:
            raise ValueError('Semantic root allocation supports AlfredTWEnv only')
        return root_task_texts(task_refs)

    def sample_root_candidates(
        self,
        n_candidates,
        exclude_gamefiles=None,
        exclude_session_ids=None,
    ):
        """Sample unique replayable roots without an actor rollout.

        ``exclude_session_ids`` is an alias used by the shared WebShop/ALFWorld
        Agent-DPS replay path; ALFWorld resolves it against game-file paths.
        """
        count = int(n_candidates)
        if count < 0 or count * self.group_n > self.worker_capacity:
            raise ValueError("root candidate pool exceeds ALFWorld worker capacity")
        if self.multi_modal:
            raise ValueError("TRACE root allocation currently supports AlfredTWEnv only")
        excluded = {
            str(value)
            for value in (
                list(exclude_gamefiles or [])
                + list(exclude_session_ids or [])
            )
        }
        available = [
            gamefile
            for gamefile in self._candidate_game_files
            if str(gamefile) not in excluded
        ]
        if count > len(available):
            raise ValueError(
                'ALFWorld root exclusions leave too few unique game files: '
                f'requested={count}, available={len(available)}'
            )
        if count == 0:
            return [], []
        selected = self._candidate_rng.choice(
            len(available), size=count, replace=False
        )
        gamefiles = [
            available[int(index)] for index in np.atleast_1d(selected)
        ]
        results = ray.get([
            self.workers[index * self.group_n].reset.remote(gamefile=gamefile)
            for index, gamefile in enumerate(gamefiles)
        ])
        text_obs_list = []
        info_list = []
        for gamefile, (obs, info) in zip(gamefiles, results):
            info = {key: value[0] for key, value in info.items()}
            info.setdefault('extra.gamefile', gamefile)
            text_obs_list.append(obs[0])
            info_list.append(info)
        return text_obs_list, info_list

    def sample_fresh_root_candidates(
        self,
        n_candidates,
        exclude_gamefiles=None,
        exclude_session_ids=None,
    ):
        """Sample fresh roots from a shared shuffled task bag."""

        gamefiles = self.sample_root_task_refs(
            n_candidates,
            exclude_gamefiles=exclude_gamefiles,
            exclude_session_ids=exclude_session_ids,
        )
        return self.load_root_candidates(gamefiles)

    def sample_root_task_refs(
        self,
        n_candidates,
        exclude_gamefiles=None,
        exclude_session_ids=None,
    ):
        """Sample a large root candidate pool without resetting workers."""

        count = int(n_candidates)
        if count < 0:
            raise ValueError("root task candidate count must be non-negative")
        excluded = {
            str(value)
            for value in (
                list(exclude_gamefiles or [])
                + list(exclude_session_ids or [])
            )
        }
        return self._fresh_root_pool.sample(count, excluded=excluded)

    def load_root_candidates(self, gamefiles):
        """Reset only curriculum-selected ALFWorld roots."""

        gamefiles = [str(gamefile) for gamefile in gamefiles]
        count = len(gamefiles)
        if count * self.group_n > self.worker_capacity:
            raise ValueError("selected root batch exceeds ALFWorld worker capacity")
        if self.multi_modal:
            raise ValueError("TRACE root allocation currently supports AlfredTWEnv only")
        results = ray.get([
            self.workers[index * self.group_n].reset.remote(gamefile=gamefile)
            for index, gamefile in enumerate(gamefiles)
        ])
        text_obs_list = []
        info_list = []
        for gamefile, (obs, info) in zip(gamefiles, results):
            info = {key: value[0] for key, value in info.items()}
            info.setdefault('extra.gamefile', gamefile)
            text_obs_list.append(obs[0])
            info_list.append(info)
        return text_obs_list, info_list

    def fresh_task_sampler_state_dict(self):
        return self._fresh_root_pool.state_dict()

    def load_fresh_task_sampler_state_dict(self, state):
        self._fresh_root_pool.load_state_dict(state)

    def fresh_task_sampler_metrics(self):
        return self._fresh_root_pool.metrics()

    def reset(self, gamefiles=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        if gamefiles is None:
            self.num_processes = self.worker_capacity
        else:
            if not 0 < len(gamefiles) <= self.worker_capacity:
                raise ValueError("root_budget exceeds ALFWorld worker capacity")
            self.num_processes = len(gamefiles)
        self.prev_admissible_commands = [None] * self.num_processes
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for index, worker in enumerate(self.workers[:self.num_processes]):
            future = (
                worker.reset.remote()
                if gamefiles is None
                else worker.reset.remote(gamefile=gamefiles[index])
            )
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self, indices=None):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        if indices is None:
            indices = range(self.num_processes)
        futures = []
        for index in indices:
            future = self.workers[int(index)].getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        workers = list(self.workers)
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
        for worker in workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self.workers = []
        self.num_processes = 0
        self.worker_capacity = 0
        self.prev_admissible_commands = []

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
