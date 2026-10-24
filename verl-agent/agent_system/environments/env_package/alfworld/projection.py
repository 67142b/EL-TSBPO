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

from typing import List
import re

def alfworld_projection(
    actions: List[str],
    action_pools: List[List[str]],
    *,
    return_details: bool = False,
):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)
    format_valids = [0] * len(actions)
    admissible_valids = [0] * len(actions)

    def normalize_command(command: str) -> str:
        # ALFWorld commands are text commands; collapse incidental whitespace
        # so formatting differences do not turn a legal command into a false
        # invalid action.
        return " ".join(str(command).strip().lower().split())

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag, start_idx + len(start_tag))
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = normalize_command(
                actions[i][start_idx + len(start_tag):end_idx]
            )
            
            actions[i] = extracted_action
            pool = action_pools[i] if i < len(action_pools) and action_pools[i] else []
            normalized_action = extracted_action
            normalized_pool = {normalize_command(command) for command in pool}
            # Syntax alone is insufficient: the action must be one of the
            # commands currently admissible in this environment snapshot.
            admissible_valids[i] = int(
                bool(normalized_action and normalized_action in normalized_pool)
            )

        except:
            actions[i] = actions[i][-30:]
            continue

        # Validate the whole response, independently of command extraction.
        # The extracted first command is still sent to ALFWorld even when
        # repeated tags or trailing junk make the producing turn invalid.
        # This preserves environment rewards while charging the existing
        # local invalid-action cost to precisely this turn.
        expected_tags = ("<think>", "</think>", "<action>", "</action>")
        tags = re.findall(r"</?(?:think|action)>", original_str, flags=re.IGNORECASE)
        format_valids[i] = int(
            tuple(tags) == expected_tags
            and re.fullmatch(
                r"\s*<think>.*?</think>\s*<action>.*?</action>\s*",
                original_str, flags=re.DOTALL,
            ) is not None
            and not re.search(r'[\u4e00-\u9fff]', original_str)
        )

        valids[i] = int(bool(format_valids[i] and admissible_valids[i]))

    if return_details:
        return actions, valids, {
            "format_valid": format_valids,
            "admissible_valid": admissible_valids,
        }
    return actions, valids
