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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        return self._initialize_observations(text_obs, image_obs, infos)

    def reset_allocated_roots(self, gamefiles):
        """Start each allocated bare rollout at its candidate task root."""
        text_obs, image_obs, infos = self.envs.reset(gamefiles=list(gamefiles))
        return self._initialize_observations(text_obs, image_obs, infos)

    def _initialize_observations(self, text_obs, image_obs, infos):
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = list(text_obs)
        self._last_image_obs = image_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids, validity = self.projection_f(
            text_actions,
            self.envs.get_admissible_commands,
            return_details=True,
        )
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_format_valid'] = to_numpy(validity['format_valid'][i])
            info['is_action_admissible'] = to_numpy(validity['admissible_valid'][i])
            # Preserve the canonical action actually sent to the environment.
            # Branch replay and snapshot identities must not depend on the raw
            # model response when projection normalizes or rejects it.
            info['executed_action'] = actions[i]

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def step_active(self, active_indices: List[int], text_actions: List[str]):
        """Step only live ALFWorld workers and expand results to batch shape."""
        active_indices = [int(index) for index in active_indices]
        if len(active_indices) != len(text_actions):
            raise ValueError("active_indices and text_actions must have equal length")
        batch_size = len(self.pre_text_obs)
        if any(index < 0 or index >= batch_size for index in active_indices):
            raise IndexError("active environment index out of range")

        action_pools = self.envs.get_admissible_commands
        active_pools = [action_pools[index] or [] for index in active_indices]
        actions, valids, validity = self.projection_f(
            list(text_actions), active_pools, return_details=True
        )
        previous_obs = [self.pre_text_obs[index] for index in active_indices]
        text_obs, image_obs, rewards, dones, infos = self.envs.step_active(
            active_indices, actions
        )

        self.memory.store_at(
            active_indices,
            {"text_obs": previous_obs, "action": actions},
        )
        for local_idx, index in enumerate(active_indices):
            self.pre_text_obs[index] = text_obs[local_idx]
            info = infos[local_idx]
            if info.get("extra.gamefile") is None:
                info["extra.gamefile"] = self.gamefile[index]
            info["is_action_valid"] = to_numpy(valids[local_idx])
            info["is_action_format_valid"] = to_numpy(
                validity["format_valid"][local_idx]
            )
            info["is_action_admissible"] = to_numpy(
                validity["admissible_valid"][local_idx]
            )
            info["executed_action"] = actions[local_idx]

        full_text_obs = self.build_text_obs(
            self.pre_text_obs, self.envs.get_admissible_commands
        )
        if image_obs is None:
            full_image_obs = None
        else:
            current_images = list(
                self._last_image_obs
                if self._last_image_obs is not None
                else [None] * batch_size
            )
            for local_idx, index in enumerate(active_indices):
                current_images[index] = image_obs[local_idx]
            self._last_image_obs = current_images
            full_image_obs = current_images

        full_rewards = np.zeros(batch_size, dtype=np.float32)
        full_dones = np.ones(batch_size, dtype=bool)
        full_infos = [
            {
                "is_action_valid": np.asarray(True),
                "is_action_format_valid": np.asarray(True),
                "is_action_admissible": np.asarray(True),
                "executed_action": "",
            }
            for _ in range(batch_size)
        ]
        for local_idx, index in enumerate(active_indices):
            full_rewards[index] = rewards[local_idx]
            full_dones[index] = dones[local_idx]
            full_infos[index] = infos[local_idx]

        return (
            {"text": full_text_obs, "image": full_image_obs, "anchor": list(self.pre_text_obs)},
            full_rewards,
            full_dones,
            full_infos,
        )
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break






class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        kwargs = kwargs or {}
        obs, infos = self.envs.reset(session_ids=kwargs.get('session_ids'))
        return self._initialize_observations(obs, infos)

    def reset_allocated_roots(self, task_refs):
        """Start allocated bare rollouts at fixed WebShop sessions."""

        obs, infos = self.envs.reset_allocated_sessions(
            [int(task_ref) for task_ref in task_refs]
        )
        return self._initialize_observations(obs, infos)

    def canonicalize_root_candidates(self, observations, infos):
        """Return root observations in the same form as a normal reset.

        ``WebshopMultiProcessEnv.sample_root_candidates`` intentionally
        returns the raw worker observation because it is also used to inspect
        the task/session.  The rollout manager, however, stores the
        post-task-prefix observation as ``anchor``.  Agent-DPS must query the
        same state key in both places; otherwise every sampled root is a new
        raw-string key and its posterior falls back to ``p=0.5``.

        This method is side-effect free with respect to the live manager state:
        ``format_obs`` needs ``self.tasks``, so we temporarily derive the
        sampled tasks and then restore the task list used by the active
        rollout.
        """

        observations = list(observations or [])
        infos = [dict(info or {}) for info in (infos or [])]
        if len(observations) != len(infos):
            raise ValueError(
                'WebShop root observations and infos must have equal length'
            )
        for info in infos:
            if not info.get('extra.gamefile') and info.get('webshop.session_id') is not None:
                info['extra.gamefile'] = str(info['webshop.session_id'])
        if not observations:
            return [], infos

        sentinel = object()
        previous_tasks = getattr(self, 'tasks', sentinel)
        try:
            self.tasks = self.extract_task(observations)
            canonical = self.format_obs(observations)
        finally:
            if previous_tasks is sentinel:
                try:
                    del self.tasks
                except AttributeError:
                    pass
            else:
                self.tasks = previous_tasks
        return canonical, infos

    def _initialize_observations(self, obs, infos):
        self.tasks = self.extract_task(obs)
        self.replay_task_refs = [
            str(info.get('webshop.session_id')) for info in infos
        ]
        # WebShop uses a session ID as its replayable task reference.
        self.gamefile = list(self.replay_task_refs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self._last_infos = [dict(info or {}) for info in infos]
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids, validity = self.projection_f(
            text_actions,
            self.envs.get_admissible_commands,
            return_details=True,
        )
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_format_valid'] = to_numpy(
                validity['format_valid'][i]
            )
            info['is_action_admissible'] = to_numpy(
                validity['admissible_valid'][i]
            )
            info['executed_action'] = actions[i]

        self._last_infos = [dict(info or {}) for info in infos]

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def step_active(self, active_indices: List[int], text_actions: List[str]):
        """Step live WebShop workers without advancing finished sessions."""
        active_indices = [int(index) for index in active_indices]
        if len(active_indices) != len(text_actions):
            raise ValueError('active_indices and text_actions must have equal length')
        batch_size = len(self.pre_text_obs)
        if any(index < 0 or index >= batch_size for index in active_indices):
            raise IndexError('active WebShop environment index out of range')

        active_pools = [
            self.envs.get_admissible_commands[index] or []
            for index in active_indices
        ]
        actions, valids, validity = self.projection_f(
            list(text_actions), active_pools, return_details=True
        )
        previous_obs = [self.pre_text_obs[index] for index in active_indices]
        text_obs, rewards, dones, infos = self.envs.step_active(
            active_indices, actions,
        )
        text_obs = self.format_obs(text_obs, indices=active_indices)

        self.memory.store_at(
            active_indices,
            {'text_obs': previous_obs, 'action': actions},
        )
        for local_idx, index in enumerate(active_indices):
            self.pre_text_obs[index] = text_obs[local_idx]
            info = dict(infos[local_idx] or {})
            info['is_action_valid'] = to_numpy(valids[local_idx])
            info['is_action_format_valid'] = to_numpy(
                validity['format_valid'][local_idx]
            )
            info['is_action_admissible'] = to_numpy(
                validity['admissible_valid'][local_idx]
            )
            info['executed_action'] = actions[local_idx]
            infos[local_idx] = info
            self._last_infos[index] = info

        full_text_obs = self.build_text_obs(self.pre_text_obs, self._last_infos)
        full_rewards = np.zeros(batch_size, dtype=np.float32)
        full_dones = np.ones(batch_size, dtype=bool)
        full_infos = [dict(info) for info in self._last_infos]
        for local_idx, index in enumerate(active_indices):
            full_rewards[index] = rewards[local_idx]
            full_dones[index] = dones[local_idx]
            full_infos[index] = infos[local_idx]

        return (
            {'text': full_text_obs, 'image': None, 'anchor': list(self.pre_text_obs)},
            full_rewards,
            full_dones,
            full_infos,
        )

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs, indices=None):
        postprocess_text_obs = []
        if indices is None:
            indices = range(len(text_obs))
        for local_idx, task_idx in enumerate(indices):
            obs = text_obs[local_idx]
            parts = obs.split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[task_idx])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = obs

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return


def make_envs(config, *, create_train: bool = True, create_val: bool = True):
    """Create train and/or validation environment managers.

    Validation environments may be created lazily so their Ray actors do not
    occupy CPU resources during rollout/training-only intervals.
    """
    if not create_train and not create_val:
        raise ValueError("At least one of create_train/create_val must be true")
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset,
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker) if create_train else None
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker) if create_val else None
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config) if _envs is not None else None
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config) if _val_envs is not None else None
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        num_products = 1000 if config.env.webshop.use_small else None
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': num_products, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path,
                    # Tree branches are restored in different Ray actors. A
                    # shared goal ordering makes ``session_id`` a stable task
                    # identity across factual, root-allocation and branch pools.
                    '_shared_goal_order': True,
                    # Root task selection is an environment concern.  Agent-
                    # DPS receives the selected roots later and only assigns
                    # their rollout quotas.
                    '_task_sampler_mode': config.env.webshop.get(
                        'task_sampler_mode', 'query_stratified'
                    ),
                    '_task_sampler_query_slots': config.env.webshop.get(
                        'task_sampler_query_slots', None
                    ),
                    '_task_sampler_natural_slots': config.env.webshop.get(
                        'task_sampler_natural_slots', None
                    ),
                    '_task_sampler_seed': config.env.webshop.get(
                        'task_sampler_seed', config.env.seed
                    ),
                    }
        # Explicit evaluation protocols may pin a catalog order separately
        # from simulator RNG seeds, and reuse a catalog across worker slots.
        if config.env.webshop.get('goal_order_seed', None) is not None:
            env_kwargs['_goal_order_seed'] = int(config.env.webshop.goal_order_seed)
        if config.env.webshop.get('workers_per_process', None) is not None:
            env_kwargs['_workers_per_process'] = int(config.env.webshop.workers_per_process)
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker) if create_train else None
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker) if create_val else None

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config) if _envs is not None else None
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config) if _val_envs is not None else None
        import time
        worker_count = (
            (config.data.train_batch_size * group_n if create_train else 0)
            + (config.data.val_batch_size if create_val else 0)
        )
        time.sleep(worker_count * 0.1) # wait for the envs to be ready
        return envs, val_envs
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
