# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs

from verl.utils.import_utils import is_sglang_available


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def use_behavior_logprobs(engine_kwargs):
    """Make returned vLLM log-probs describe the sampled distribution.

    vLLM V1 defaults to ``raw_logprobs``, computed before temperature,
    penalties, and top-k/top-p. PPO rollout probabilities must instead be the
    behavior-policy probabilities. Older supported releases do not expose the
    option and keep their existing behavior.
    """

    result = dict(engine_kwargs)
    try:
        from vllm.engine.arg_utils import EngineArgs

        if "logprobs_mode" in getattr(EngineArgs, "__dataclass_fields__", {}):
            result.setdefault("logprobs_mode", "processed_logprobs")
        else:
            # 旧版 vLLM 不暴露该参数，保持其既有行为；剔除用户/脚本显式
            # 传入的值，避免 EngineArgs 收到未知关键字而启动失败。
            result.pop("logprobs_mode", None)
    except (ImportError, AttributeError):
        pass
    return result


package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None


if package_version == "0.5.4":
    vllm_version = "0.5.4"
    from .vllm_v_0_5_4 import parallel_state
    from .vllm_v_0_5_4.llm import LLM, LLMEngine
elif package_version == "0.6.3" or package_version.startswith("0.6.3"):
    # rocm version: "0.6.3+rocmxxx"
    vllm_version = "0.6.3"
    from .vllm_v_0_6_3 import parallel_state
    from .vllm_v_0_6_3.llm import LLM, LLMEngine
elif vs.parse(package_version) >= vs.parse("0.7.0"):
    # From 0.6.6.post2 on, vllm supports SPMD inference
    # See https://github.com/vllm-project/vllm/pull/12071

    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if not is_sglang_available():
        raise ValueError(f"vllm version {package_version} not supported and SGLang also not Found. Currently supported vllm versions are 0.6.3 and 0.7.0+")

__all__ = ["LLM", "LLMEngine", "parallel_state", "use_behavior_logprobs"]
