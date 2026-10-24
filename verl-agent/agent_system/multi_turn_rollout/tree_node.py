"""Tree node metadata shared by factual and branched rollouts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None = None
    depth: int = 0
    task: str = ""
    gamefile: str = ""
    action_history: list[str] = field(default_factory=list)
    memory_steps: list[dict[str, str]] = field(default_factory=list)
    branch_type: str = "natural"
    branch_group_id: str | None = None
    reward: float = 0.0
    won: bool = False
    is_leaf: bool = False
    children: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
