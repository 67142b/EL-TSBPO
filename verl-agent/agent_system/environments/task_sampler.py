import hashlib
import json
from typing import Any, Iterable

import numpy as np


class ShuffledTaskPool:
    """Seeded, resumable traversal over a fixed task population."""

    _SCHEMA_VERSION = 1

    def __init__(self, tasks: Iterable[Any], *, seed: int = 0) -> None:
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("ShuffledTaskPool requires at least one task")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("ShuffledTaskPool tasks must be unique")
        self.seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.excluded_skips = 0
        self._fingerprint = self._task_fingerprint(self.tasks)
        self._task_to_index = {
            task: index for index, task in enumerate(self.tasks)
        }
        self._covered_indices: set[int] = set()
        self._order = self._order_for_epoch(self.epoch)

    @staticmethod
    def _task_fingerprint(tasks: list[Any]) -> str:
        payload = json.dumps(
            [str(task) for task in tasks],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _order_for_epoch(self, epoch: int) -> list[int]:
        # Reconstructing each permutation from (seed, epoch) keeps checkpoints
        # compact while preserving the exact task sequence after resume.
        epoch_seed = (
            (self.seed & 0xFFFFFFFF) + int(epoch) * 0x9E3779B1
        ) & 0xFFFFFFFF
        rng = np.random.RandomState(epoch_seed)
        return [int(index) for index in rng.permutation(len(self.tasks))]

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.cursor = 0
        self._covered_indices.clear()
        self._order = self._order_for_epoch(self.epoch)

    def _mark_covered(self, tasks: Iterable[Any]) -> None:
        for task in tasks:
            index = self._task_to_index.get(task)
            if index is None or index in self._covered_indices:
                continue
            self._covered_indices.add(index)
            self.excluded_skips += 1

    def sample(self, count: int, *, excluded: Iterable[Any] = ()) -> list[Any]:
        count = int(count)
        if count < 0:
            raise ValueError("ShuffledTaskPool sample count must be non-negative")
        excluded_set = set(excluded)
        available = len(self.tasks) - sum(task in excluded_set for task in self.tasks)
        if count > available:
            raise ValueError(
                "ShuffledTaskPool exclusions leave too few unique tasks: "
                f"requested={count}, available={available}"
            )
        if count == 0:
            return []

        selected = []
        selected_set = set()
        # Replay roots also train the actor this round. Count them as covered
        # by the current traversal so the fresh lane does not immediately
        # revisit them while other task instances remain unseen.
        self._mark_covered(excluded_set)
        while len(selected) < count:
            if self.cursor >= len(self._order):
                self._advance_epoch()
                self._mark_covered(excluded_set | selected_set)
            task_index = self._order[self.cursor]
            self.cursor += 1
            if task_index in self._covered_indices:
                continue
            task = self.tasks[task_index]
            selected.append(task)
            selected_set.add(task)
            self._covered_indices.add(task_index)
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "task_count": len(self.tasks),
            "task_fingerprint": self._fingerprint,
            "excluded_skips": self.excluded_skips,
            "covered_indices": sorted(self._covered_indices),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state = dict(state or {})
        if int(state.get("schema_version", 0)) != self._SCHEMA_VERSION:
            raise ValueError("unsupported shuffled task-pool state schema")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("shuffled task-pool seed does not match current run")
        if int(state.get("task_count", -1)) != len(self.tasks):
            raise ValueError("shuffled task-pool task count has changed")
        if str(state.get("task_fingerprint", "")) != self._fingerprint:
            raise ValueError("shuffled task-pool population has changed")

        epoch = max(0, int(state.get("epoch", 0)))
        cursor = int(state.get("cursor", 0))
        if not 0 <= cursor <= len(self.tasks):
            raise ValueError("invalid shuffled task-pool cursor")
        self.epoch = epoch
        self.cursor = cursor
        self.excluded_skips = max(0, int(state.get("excluded_skips", 0)))
        covered_indices = {
            int(index) for index in state.get("covered_indices", [])
        }
        if any(
            index < 0 or index >= len(self.tasks)
            for index in covered_indices
        ):
            raise ValueError("invalid shuffled task-pool covered index")
        self._covered_indices = covered_indices
        self._order = self._order_for_epoch(epoch)

    def metrics(self) -> dict[str, float]:
        return {
            "agent_dps/root/fresh_pool_epoch": float(self.epoch),
            "agent_dps/root/fresh_pool_cursor": float(self.cursor),
            "agent_dps/root/fresh_pool_size": float(len(self.tasks)),
            "agent_dps/root/fresh_pool_covered": float(
                len(self._covered_indices)
            ),
            "agent_dps/root/fresh_pool_fraction": float(
                len(self._covered_indices) / max(1, len(self.tasks))
            ),
            "agent_dps/root/fresh_pool_excluded_skips": float(
                self.excluded_skips
            ),
        }
