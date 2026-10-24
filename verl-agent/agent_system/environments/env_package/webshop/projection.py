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

from typing import List, Optional
import re

def webshop_projection(
    actions: List[str],
    action_pools: Optional[List[List[str]]] = None,
    *,
    return_details: bool = False,
):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

    valids = [0] * len(actions)
    format_valids = [0] * len(actions)
    admissible_valids = [0] * len(actions)

    def normalize_command(command: str) -> str:
        return " ".join(str(command).strip().lower().split())

    def is_admissible(command: str, pool: Optional[List[str]]) -> bool:
        # Keep the legacy projection behavior for callers which cannot expose
        # the current page actions. Training managers always provide a pool.
        if pool is None:
            return bool(command)
        normalized_pool = {normalize_command(item) for item in (pool or [])}
        if command in normalized_pool:
            return True
        # WebShop exposes a search template while the policy supplies the
        # concrete query. Empty or malformed searches remain invalid.
        search = re.fullmatch(r"search\[(.+)\]", command)
        return bool(
            search
            and search.group(1).strip()
            and "search[<your query>]" in normalized_pool
        )

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        lowered = str(actions[i]).lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = lowered.find(start_tag)
        end_idx = lowered.find(end_tag, start_idx + len(start_tag))
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = lowered[-20:]
                continue

            # Extract just the content between the tags
            extracted_action = normalize_command(
                lowered[start_idx + len(start_tag):end_idx]
            )
            actions[i] = extracted_action
            pool = action_pools[i] if action_pools is not None and i < len(action_pools) else None
            admissible_valids[i] = int(is_admissible(extracted_action, pool))

        except Exception:
            actions[i] = lowered[-20:]
            continue

        # Match ALFWorld's response contract: exactly one think block followed
        # by exactly one action block, with only whitespace outside them.
        expected_tags = ("<think>", "</think>", "<action>", "</action>")
        tags = re.findall(r"</?(?:think|action)>", original_str, flags=re.IGNORECASE)
        format_valids[i] = int(
            tuple(tags) == expected_tags
            and re.fullmatch(
                r"\s*<think>.*?</think>\s*<action>.*?</action>\s*",
                original_str,
                flags=re.DOTALL | re.IGNORECASE,
            ) is not None
            and not re.search(r"[\u4e00-\u9fff]", original_str)
        )
        valids[i] = int(bool(format_valids[i] and admissible_valids[i]))

    if return_details:
        return actions, valids, {
            "format_valid": format_valids,
            "admissible_valid": admissible_valids,
        }
    return actions, valids
