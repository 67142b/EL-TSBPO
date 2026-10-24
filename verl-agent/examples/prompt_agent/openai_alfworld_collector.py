"""Collect ALFWorld SFT demonstrations from a teacher API."""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))


current_pythonpath = os.environ.get('PYTHONPATH', '')
if PROJECT_DIR not in current_pythonpath:
    os.environ['PYTHONPATH'] = PROJECT_DIR + ':' + current_pythonpath


if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import json
import time
import logging
import argparse
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from openai import AsyncOpenAI

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def build_alfworld_env(env_num: int, use_train: bool = True, eval_dataset: str = "eval_in_distribution"):
    """Build vectorized ALFWorld environments for data collection."""
    import ray
    

    os.environ['RAY_memory_usage_threshold'] = '0.85'
    

    if not ray.is_initialized():
        ray.init(
            runtime_env={"working_dir": PROJECT_DIR},
            ignore_reinit_error=True,
            object_store_memory=500 * 1024 * 1024,
            _memory=4 * 1024 * 1024 * 1024,
        )
    
    from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs
    alf_config_path = os.path.join(
        PROJECT_DIR, 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    )
    env_kwargs = {'eval_dataset': eval_dataset}

    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}
    envs = build_alfworld_envs(
        alf_config_path, seed=1, env_num=env_num, group_n=1,
        is_train=use_train, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker
    )
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        'env': {
            'history_length': 5,
            'env_name': 'alfworld/AlfredTWEnv',
        }
    })
    env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, config)
    

    if use_train:
        logging.info("使用训练集数据 (train)")
    else:
        logging.info(f"使用验证集数据 ({eval_dataset})")
    
    return env_manager


def parse_gamefile(gamefile: str) -> str:
    if gamefile is None:
        return "other"
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def extract_action(response_text: str) -> str:
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = response_text.find(start_tag)
    end_idx = response_text.find(end_tag)
    if start_idx != -1 and end_idx != -1:
        return response_text[start_idx + len(start_tag):end_idx].strip()
    return response_text.strip()[-30:]


class IncrementalSaver:
    """Append successful interaction turns to the SFT dataset."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.lock = asyncio.Lock()
        self._existing_count = 0

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._existing_count = len(data) if isinstance(data, list) else 0
            except (json.JSONDecodeError, IOError):
                self._existing_count = 0
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)

    @property
    def existing_count(self):
        return self._existing_count

    async def save_trajectory_steps(
        self,
        trajectory_steps: List[List[Dict]],
    ):
        async with self.lock:
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = []


            for step_messages in trajectory_steps:
                data.append({"messages": step_messages})

            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self._existing_count = len(data)


async def call_teacher_api(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    temperature: float | None = None,
    max_retries: int = 5,
) -> str:
    """Request a teacher response with bounded retries."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            request = {"model": model_name, "messages": messages, "n": 1}
            if temperature is not None:
                request["temperature"] = temperature
            response = await client.chat.completions.create(**request)
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait_time = min(2 ** attempt, 30)
            logging.warning(
                f"API 请求失败 (尝试 {attempt + 1}/{max_retries}): {e}，"
                f"{wait_time}s 后重试..."
            )
            await asyncio.sleep(wait_time)
    raise RuntimeError(f"API 请求在 {max_retries} 次重试后仍然失败")


async def run_collection(
    client: AsyncOpenAI,
    model_name: str,
    env_manager,
    saver: IncrementalSaver,
    env_num: int,
    max_steps: int,
    temperature: float,
):
    """Collect successful trajectories from the training environment."""

    kwargs = {}
    obs, infos = env_manager.reset(kwargs)


    env_dones = [False] * env_num
    env_won = [False] * env_num
    env_trajectories = [[] for _ in range(env_num)]
    env_actions = [None] * env_num
    env_gamefiles = [infos[i].get("extra.gamefile", "") for i in range(env_num)]


    step_prompts = [obs["text"][i] for i in range(env_num)]


    api_tasks = [
        call_teacher_api(client, model_name, prompt, temperature)
        for prompt in step_prompts
    ]
    api_responses = await asyncio.gather(*api_tasks, return_exceptions=True)


    for i in range(env_num):
        if isinstance(api_responses[i], Exception):
            logging.error(f"环境 {i} 第 1 步 API 请求失败: {api_responses[i]}")
            env_dones[i] = True
            continue

        response_text = api_responses[i]
        user_msg = {"role": "user", "content": step_prompts[i]}
        assistant_msg = {"role": "assistant", "content": response_text}

        env_trajectories[i].append([user_msg, assistant_msg])

        action = extract_action(response_text)
        env_actions[i] = action


    for step_idx in range(1, max_steps + 1):
        active_indices = [i for i in range(env_num) if not env_dones[i]]
        if not active_indices:
            break

        logging.info(
            f"Step {step_idx}; Dones ({sum(env_dones)}/{env_num}); "
            f"Active {len(active_indices)}"
        )


        actions = ["None"] * env_num
        for i in active_indices:
            actions[i] = env_actions[i]

        obs, rewards, dones, infos = env_manager.step(actions)


        newly_done = []
        still_active = []
        for i in active_indices:
            if dones[i]:
                env_dones[i] = True
                env_won[i] = bool(infos[i].get("won", False))
                newly_done.append(i)
            else:
                still_active.append(i)


        for i in newly_done:
            if not env_won[i]:
                continue
            task_type = parse_gamefile(env_gamefiles[i])
            await saver.save_trajectory_steps(env_trajectories[i])
            logging.info(
                f"环境 {i} 完成 | 成功: {env_won[i]} | "
                f"步数: {len(env_trajectories[i])} | 任务: {task_type} | "
                f"已收集: {saver.existing_count}"
            )

        if not still_active:
            logging.info("所有环境已完成!")
            break

        if step_idx == max_steps:
            break



        step_prompts = [obs["text"][i] for i in still_active]


        api_tasks = [
            call_teacher_api(client, model_name, prompt, temperature)
            for prompt in step_prompts
        ]
        api_responses = await asyncio.gather(*api_tasks, return_exceptions=True)


        for idx, i in enumerate(still_active):
            if isinstance(api_responses[idx], Exception):
                logging.error(
                    f"环境 {i} 第 {step_idx + 1} 步 API 请求失败: "
                    f"{api_responses[idx]}"
                )
                env_dones[i] = True
                task_type = parse_gamefile(env_gamefiles[i])
                continue

            response_text = api_responses[idx]
            user_msg = {"role": "user", "content": step_prompts[idx]}
            assistant_msg = {"role": "assistant", "content": response_text}

            env_trajectories[i].append([user_msg, assistant_msg])

            action = extract_action(response_text)
            env_actions[i] = action


    success_count = sum(env_won)
    total_count = env_num
    task_stats = defaultdict(lambda: {"success": 0, "total": 0})
    for i in range(env_num):
        task_type = parse_gamefile(env_gamefiles[i])
        task_stats[task_type]["total"] += 1
        if env_won[i]:
            task_stats[task_type]["success"] += 1

    return success_count, total_count, task_stats


async def main():
    parser = argparse.ArgumentParser(
        description="收集 ALFWorld 训练任务的成功示范"
    )
    parser.add_argument(
        "--model_name", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"),
        help="OpenAI 兼容教师模型名称"
    )
    parser.add_argument(
        "--api_base", type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI 兼容 API 地址"
    )
    parser.add_argument("--env_num", type=int, default=20, help="每次运行的环境数量")
    parser.add_argument("--test_times", type=int, default=1, help="重复运行次数")
    parser.add_argument("--max_steps", type=int, default=50, help="每个环境的最大步数")
    parser.add_argument("--temperature", type=float, default=None, help="可选采样温度")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()


    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("set OPENAI_API_KEY in the environment")
    if not args.model_name:
        parser.error("set OPENAI_MODEL or pass --model_name")


    if args.output_file is None:
        args.output_file = os.path.join(
            os.path.dirname(PROJECT_DIR), "llamafactory", "data", "data.json"
        )


    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld",
        f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.info(
        f"配置: model={args.model_name}, env_num={args.env_num}, "
        f"test_times={args.test_times}, max_steps={args.max_steps}, "
        f"output={args.output_file}"
    )


    client = AsyncOpenAI(api_key=api_key, base_url=args.api_base)
    env_manager = build_alfworld_env(args.env_num, use_train=True)
    saver = IncrementalSaver(args.output_file)

    logging.info(f"已有数据量: {saver.existing_count}")


    overall_success_rates = []
    task_success_history = defaultdict(list)

    for test_idx in range(args.test_times):
        logging.info(f"\n========== 开始第 {test_idx} 轮 ==========")
        start_time = time.time()

        success_count, total_count, task_stats = await run_collection(
            client, args.model_name, env_manager, saver,
            args.env_num, args.max_steps, args.temperature,
        )

        round_success_rate = (
            success_count / total_count if total_count > 0 else 0.0
        )
        overall_success_rates.append(round_success_rate)

        logging.info(
            f"第 {test_idx} 轮成功率: "
            f"{round_success_rate:.4f} ({success_count}/{total_count})"
        )
        for task_type, stats in task_stats.items():
            rate = (
                stats["success"] / stats["total"]
                if stats["total"] > 0 else 0.0
            )
            task_success_history[task_type].append(rate)
            logging.info(
                f"  {task_type:<35s}: {rate:.4f} "
                f"({stats['success']}/{stats['total']})"
            )

        logging.info(f"第 {test_idx} 轮耗时: {time.time() - start_time:.2f}s")


    logging.info("\n=============== 最终汇总 ===============")
    logging.info(
        f"总轮次: {args.test_times} | 每轮环境数: {args.env_num} | "
        f"总环境数: {args.env_num * args.test_times}"
    )
    if overall_success_rates:
        logging.info(
            f"总成功率: {np.mean(overall_success_rates):.4f} "
            f"\u00b1 {np.std(overall_success_rates):.4f}"
        )
    logging.info(f"总收集轨迹数: {saver.existing_count}")

    for task_type in TASKS + ["other"]:
        if task_type in task_success_history:
            rates = task_success_history[task_type]
            logging.info(
                f"  {task_type:<35s}: {np.mean(rates):.4f} "
                f"\u00b1 {np.std(rates):.4f}"
            )

    logging.info(f"数据已保存到: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
