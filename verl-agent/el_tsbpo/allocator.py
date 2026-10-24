"""EL-TSBPO allocator implementation."""

from dataclasses import dataclass, field


@dataclass
class AnchorCandidate:
    anchor_id: str
    trajectory_id: str
    task_uid: str
    env_idx: int
    step_idx: int
    context: str
    gamefile: str
    action_history: list[str]
    memory_steps: list[dict[str, str]]
    anchor_obs: str
    admissible_commands: list[str] = field(default_factory=list)
    action_status_history: list[str] = field(default_factory=list)
    task: str = ""
    context_hash: str = ""
    decision_key: str = ""
    policy_group_key: str = ""
    env_key: str = ""
    node_id: str = ""
    parent_node_id: str = ""
    current_action: str = ""
    current_action_status: str = "legal"
    next_anchor_obs: str = ""
    is_terminal: bool = False
    trajectory_length: int = 0
