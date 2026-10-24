"""WebShop training task sampling, independent of environment workers."""

from collections import defaultdict
from copy import deepcopy

import numpy as np


class WebshopTaskSampler:
    """Task sampler for WebShop roots.

    ``random_pool`` draws directly from all task instances. Draws are
    independent across rounds and session-unique inside one batch, so query
    families retain exactly their original frequency in expectation. The older
    no-replacement queues remain available for compatibility with existing
    launchers.
    """

    def __init__(
        self,
        goals,
        goal_indices,
        *,
        seed: int = 0,
        query_slots: int | None = None,
        natural_slots: int | None = None,
        replacement_across_batches: bool = False,
    ):
        self.goal_indices = [int(index) for index in goal_indices]
        if not self.goal_indices:
            raise ValueError('WebShop task sampler requires at least one goal')
        self._rng = np.random.RandomState(int(seed))
        self.query_slots = None if query_slots is None or int(query_slots) < 0 else int(query_slots)
        self.natural_slots = None if natural_slots is None or int(natural_slots) < 0 else int(natural_slots)
        self.replacement_across_batches = bool(replacement_across_batches)
        records = list(goals or [])
        self._goals = records
        by_query = defaultdict(list)
        for goal_index in self.goal_indices:
            goal = records[goal_index] if goal_index < len(records) else {}
            query = ' '.join(str(goal.get('query', '')).lower().split())
            by_query[query].append(goal_index)
        self._query_ids = {}
        for query, ids in by_query.items():
            values = list(ids)
            self._rng.shuffle(values)
            self._query_ids[query] = values
        self._query_order = list(self._query_ids)
        self._rng.shuffle(self._query_order)
        self._query_cursor = 0
        self._query_positions = {query: 0 for query in self._query_order}
        self._natural_ids = list(self.goal_indices)
        self._rng.shuffle(self._natural_ids)
        self._natural_cursor = 0
        self._query_epoch = 0
        self._natural_epoch = 0
        self._sampled_sessions = set()
        # Sessions are not repeated across batches until the full training
        # split has been covered once.  This is separate from the cumulative
        # metric set above so coverage can continue after each queue epoch.
        self._active_used_sessions = set()
        self._sampled_queries = set()
        self._total_sampled = 0
        self._last_batch_query_slots = 0
        self._last_batch_natural_slots = 0
        self._last_batch_revisits = 0

    @staticmethod
    def _query_for(goals, session_id: int) -> str:
        goal = goals[int(session_id)] if int(session_id) < len(goals) else {}
        return ' '.join(str(goal.get('query', '')).lower().split())

    def _next_natural(self, selected: set[int]) -> int | None:
        limit = max(1, len(self._natural_ids))
        for _ in range(limit):
            if self._natural_cursor >= len(self._natural_ids):
                self._natural_ids = list(self.goal_indices)
                self._rng.shuffle(self._natural_ids)
                self._natural_cursor = 0
                self._natural_epoch += 1
            session_id = int(self._natural_ids[self._natural_cursor])
            self._natural_cursor += 1
            if session_id not in selected and session_id not in self._active_used_sessions:
                return session_id
        return None

    def _next_query(self, selected: set[int], selected_queries: set[str]) -> int | None:
        if not self._query_order:
            return None
        # Rotate through families before reusing one within the same batch.
        for _ in range(max(1, len(self._query_order) * 2)):
            query = self._query_order[self._query_cursor % len(self._query_order)]
            self._query_cursor += 1
            if query in selected_queries and len(selected_queries) < len(self._query_order):
                continue
            ids = self._query_ids[query]
            position = self._query_positions[query]
            if position >= len(ids):
                values = list(self._query_ids[query])
                self._rng.shuffle(values)
                self._query_ids[query] = values
                ids = values
                position = 0
                self._query_positions[query] = 0
                self._query_epoch += 1
            session_id = int(ids[position])
            self._query_positions[query] = position + 1
            if session_id not in selected and session_id not in self._active_used_sessions:
                return session_id
        return self._next_natural(selected)

    def _sample_random_pool(
        self,
        count: int,
        excluded: set[int],
    ) -> list[int]:
        available = [
            session_id for session_id in self.goal_indices
            if session_id not in excluded
        ]
        if count > len(available):
            raise ValueError(
                'WebShop task sampler cannot satisfy a unique batch after '
                f'exclusions: requested={count}, available={len(available)}'
            )
        if count == 0:
            return []

        indices = self._rng.choice(len(available), size=count, replace=False)
        return [int(available[int(index)]) for index in np.atleast_1d(indices)]

    def sample(
        self,
        count: int,
        exclude_session_ids=None,
    ) -> list[int]:
        count = int(count)
        if count < 0:
            raise ValueError('WebShop task sample count must be non-negative')
        if count == 0:
            return []
        excluded = {int(value) for value in (exclude_session_ids or [])}
        if self.replacement_across_batches:
            result = self._sample_random_pool(count, excluded)
            self._last_batch_query_slots = len(result)
            self._last_batch_natural_slots = 0
            self._last_batch_revisits = sum(
                session_id in self._sampled_sessions for session_id in result
            )
            self._total_sampled += len(result)
            self._sampled_sessions.update(result)
            for session_id in result:
                self._sampled_queries.add(
                    self._query_for(self._goals, session_id)
                )
            return result
        if len(self._active_used_sessions) + count > len(self.goal_indices):
            self._active_used_sessions.clear()
            self._query_epoch += 1
            self._natural_epoch += 1
        if self.query_slots is None and self.natural_slots is None:
            query_count = count // 2
            natural_count = count - query_count
        else:
            query_count = max(0, int(self.query_slots or 0))
            natural_count = max(0, int(self.natural_slots or 0))
            if query_count + natural_count < count:
                natural_count += count - query_count - natural_count
            elif query_count + natural_count > count:
                natural_count = max(0, count - query_count)
                query_count = count - natural_count
        selected = set(excluded)
        selected_queries = set()
        result = []
        for _ in range(query_count):
            session_id = self._next_query(selected, selected_queries)
            if session_id is None:
                break
            selected.add(session_id)
            self._active_used_sessions.add(session_id)
            selected_queries.add(self._query_for(self._goals, session_id))
            result.append(session_id)
        for _ in range(natural_count):
            session_id = self._next_natural(selected)
            if session_id is None:
                break
            selected.add(session_id)
            self._active_used_sessions.add(session_id)
            result.append(session_id)
        # A custom slot configuration may leave capacity after one lane was
        # exhausted. Fill it from the natural queue while preserving uniqueness.
        while len(result) < count:
            session_id = self._next_natural(selected)
            if session_id is None:
                break
            selected.add(session_id)
            self._active_used_sessions.add(session_id)
            result.append(session_id)
        self._total_sampled += len(result)
        self._last_batch_revisits = sum(
            session_id in self._sampled_sessions for session_id in result
        )
        self._sampled_sessions.update(result)
        for session_id in result:
            self._sampled_queries.add(self._query_for(self._goals, session_id))
        self._last_batch_query_slots = min(query_count, len(result))
        self._last_batch_natural_slots = max(0, len(result) - self._last_batch_query_slots)
        return result

    def state_dict(self) -> dict:
        """Save the draw order as well as coverage for exact checkpoint resume."""
        fields = (
            '_query_ids', '_query_order', '_query_cursor', '_query_positions',
            '_natural_ids', '_natural_cursor', '_query_epoch', '_natural_epoch',
            '_total_sampled', '_last_batch_query_slots',
            '_last_batch_natural_slots', '_last_batch_revisits',
        )
        state = {name: deepcopy(getattr(self, name)) for name in fields}
        for name in ('_sampled_sessions', '_active_used_sessions', '_sampled_queries'):
            state[name] = sorted(getattr(self, name))
        algorithm, keys, position, has_gauss, cached_gaussian = self._rng.get_state()
        return {
            'schema_version': 1,
            'goal_indices': list(self.goal_indices),
            'query_slots': self.query_slots,
            'natural_slots': self.natural_slots,
            'replacement_across_batches': self.replacement_across_batches,
            'rng_state': [algorithm, keys.tolist(), position, has_gauss, cached_gaussian],
            'state': state,
        }

    def load_state_dict(self, state: dict) -> None:
        state = dict(state or {})
        if state.get('schema_version') != 1:
            raise ValueError('unsupported WebShop task sampler state schema')
        for name in ('goal_indices', 'query_slots', 'natural_slots', 'replacement_across_batches'):
            if state.get(name) != getattr(self, name):
                raise ValueError(f'WebShop task sampler {name} has changed')
        values = deepcopy(state['state'])
        if set(values) != set(self.state_dict()['state']):
            raise ValueError('incomplete WebShop task sampler state')
        if set(values['_query_ids']) != set(self._query_ids):
            raise ValueError('WebShop task sampler query families have changed')
        for query, ids in values['_query_ids'].items():
            if sorted(ids) != sorted(self._query_ids[query]):
                raise ValueError('WebShop task sampler query population has changed')
        algorithm, keys, position, has_gauss, cached_gaussian = state['rng_state']
        rng = np.random.RandomState()
        rng.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position,
                       has_gauss, cached_gaussian))
        for name in ('_sampled_sessions', '_active_used_sessions', '_sampled_queries'):
            values[name] = set(values[name])
        self.__dict__.update(values)
        self._rng = rng

    def metrics(self) -> dict[str, float]:
        return {
            'webshop/task_sampler/total_sampled_tasks': float(self._total_sampled),
            'webshop/task_sampler/unique_sessions_seen': float(len(self._sampled_sessions)),
            'webshop/task_sampler/unique_query_families_seen': float(len(self._sampled_queries)),
            'webshop/task_sampler/query_family_count': float(len(self._query_order)),
            'webshop/task_sampler/goal_count': float(len(self.goal_indices)),
            'webshop/task_sampler/query_family_coverage': float(len(self._sampled_queries) / max(1, len(self._query_order))),
            'webshop/task_sampler/goal_coverage': float(len(self._sampled_sessions) / max(1, len(self.goal_indices))),
            'webshop/task_sampler/query_slots': float(self._last_batch_query_slots),
            'webshop/task_sampler/natural_slots': float(self._last_batch_natural_slots),
            'webshop/task_sampler/replacement_across_batches': float(
                self.replacement_across_batches
            ),
            'webshop/task_sampler/batch_natural_revisits': float(
                self._last_batch_revisits
            ),
            'webshop/task_sampler/query_queue_epoch': float(self._query_epoch),
            'webshop/task_sampler/natural_queue_epoch': float(self._natural_epoch),
        }
