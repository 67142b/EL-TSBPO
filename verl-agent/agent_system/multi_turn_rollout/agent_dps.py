"""EL-TSBPO agent dps implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _canonical_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def state_key(task_scope: Any, anchor_obs: Any) -> str:

    payload = json.dumps(
        {
            "index": _canonical_text(task_scope),
            "anchor_obs": _canonical_text(anchor_obs),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "agent_dps:" + hashlib.sha256(payload).hexdigest()


def root_curriculum_key(task_ref: Any) -> str:

    return state_key(f"root_curriculum:{_canonical_text(task_ref)}", "__root__")


def action_key(action: Any) -> str:

    return " ".join(_canonical_text(action).strip().lower().split())


@dataclass
class StateStats:
    is_root: bool = False
    success_mass: float = 0.0
    failure_mass: float = 0.0
    last_step: int = 0
    visit_count: int = 0
    last_seen_step: int = 0


    outcome_buckets: dict[int, list[float]] = field(default_factory=dict)


    hmm_transition_mass: list[list[float]] = field(default_factory=lambda: [[0.0] * 3 for _ in range(3)])
    hmm_transition_buckets: dict[int, list[list[float]]] = field(default_factory=dict)
    hmm_belief: list[float] = field(default_factory=lambda: [1.0 / 3.0] * 3)
    hmm_belief_step: int = 0
    hmm_last_class: int = -1
    hmm_last_observation_step: int = -1
    hmm_pending_step: int = -1
    hmm_pending_success: float = 0.0
    hmm_pending_failure: float = 0.0


@dataclass
class EdgeStats:
    source_key: str
    action: str
    next_key: str
    visit_mass: float = 0.0
    success_mass: float = 0.0
    failure_mass: float = 0.0
    last_step: int = 0
    last_seen_step: int = 0

    evidence_buckets: dict[int, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminalContrastPosterior:

    alpha: float
    beta: float
    history_mass: float
    current_mass: float

    @property
    def probability(self) -> float:
        total = max(float(self.alpha) + float(self.beta), 1e-12)
        return min(1.0, max(0.0, float(self.alpha) / total))

    @property
    def variance(self) -> float:
        total = max(float(self.alpha) + float(self.beta), 1e-12)
        return max(
            0.0,
            float(self.alpha) * float(self.beta)
            / max(total * total * (total + 1.0), 1e-12),
        )

    def utility(self, count: int) -> float:
        return beta_event_utility(self.alpha, self.beta, count)


@dataclass(frozen=True)
class TransitionContrastPosterior(TerminalContrastPosterior):

    edge_count: int
    same_edge_opposite_probability: float
    factual_edge_available: bool


def branch_contrast_counts(
    factual_outcome: float | bool,
    outcomes: Mapping[str, float | bool],
    *,
    factual_edge: tuple[str, str] = ("", ""),
    branch_edges: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, int]:

    counts = {"terminal": 0, "transition": 0, "same_first_edge": 0, "edge_coverage": 0}
    factual_success = float(factual_outcome) >= 0.5
    edges = branch_edges or {}
    for trajectory, outcome in outcomes.items():
        opposite = (float(outcome) >= 0.5) != factual_success
        counts["terminal"] += int(opposite)
        edge = edges.get(str(trajectory))
        if edge is None:
            continue
        counts["edge_coverage"] += 1
        if opposite and factual_edge[0] and factual_edge[1]:
            counts["transition"] += int(tuple(edge) != factual_edge)
            counts["same_first_edge"] += int(tuple(edge) == factual_edge)
    return counts


class AgentDPSStateMemory:
    """Track decayed task and state evidence for allocation."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.90,
        root_decay: float | None = None,
        max_states: int = 100_000,
        max_age_steps: int = 32,
        hmm_prior_alpha: float = 1.0,
        hmm_transition_decay: float = 0.9,
        max_edges: int = 200_000,
    ) -> None:
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("Agent-DPS alpha must be finite and positive")
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("Agent-DPS beta must be finite and positive")
        if not math.isfinite(float(decay)) or not 0.0 < float(decay) <= 1.0:
            raise ValueError("Agent-DPS decay must be in (0, 1]")
        if not math.isfinite(float(hmm_prior_alpha)) or float(hmm_prior_alpha) <= 0.0:
            raise ValueError("Agent-DPS HMM prior alpha must be finite and positive")
        if not math.isfinite(float(hmm_transition_decay)) or not 0.0 < float(hmm_transition_decay) <= 1.0:
            raise ValueError("Agent-DPS HMM transition decay must be in (0, 1]")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.decay = float(decay)
        self.root_decay = float(decay if root_decay is None else root_decay)
        self.root_hmm_decay = float(hmm_transition_decay if root_decay is None else root_decay)
        if not math.isfinite(self.root_decay) or not 0.0 < self.root_decay <= 1.0:
            raise ValueError("Agent-DPS root decay must be in (0, 1]")
        self.max_states = max(1, int(max_states))
        self.max_age_steps = max(0, int(max_age_steps))
        self.hmm_prior_alpha = float(hmm_prior_alpha)
        self.hmm_transition_decay = float(hmm_transition_decay)
        self.max_edges = max(1, int(max_edges))
        self.states: dict[str, StateStats] = {}
        self.edges: dict[str, EdgeStats] = {}
        self._outgoing: dict[str, set[str]] = {}
        self.dirty = False
        self.pruned_total = 0
        self.last_pruned = 0
        self.pruned_edges_total = 0
        self.last_pruned_edges = 0
        self._last_prune_step: int | None = None
        self.task_sampler_state: dict[str, object] = {}


        self.root_state_keys: dict[str, str] = {}


        self.survival_memory: dict[str, Any] = {}
        self.probability_calibration: dict[str, Any] = {}

    def _refresh_stats(self, stats: StateStats, current: int) -> None:

        current = int(current)
        success_mass = 0.0
        failure_mass = 0.0
        for bucket_step, counts in stats.outcome_buckets.items():
            age = max(0, current - int(bucket_step))
            factor = (self.root_decay if stats.is_root else self.decay)**age
            success = max(0.0, float(counts[0] if counts else 0.0))
            failure = max(0.0, float(counts[1] if len(counts) > 1 else 0.0))
            success_mass += factor * success
            failure_mass += factor * failure
        stats.success_mass = success_mass
        stats.failure_mass = failure_mass
        stats.outcome_buckets = (
            {current: [success_mass, failure_mass]}
            if success_mass > 0.0 or failure_mass > 0.0
            else {}
        )
        stats.last_step = current

    def _refresh_hmm_transitions(self, stats: StateStats, current: int) -> None:

        current = int(current)
        mass = [[0.0] * 3 for _ in range(3)]
        for bucket_step, matrix in stats.hmm_transition_buckets.items():
            age = max(0, current - int(bucket_step))
            factor = (self.root_hmm_decay if stats.is_root else self.hmm_transition_decay)**age
            for row in range(3):
                values = matrix[row] if row < len(matrix) else ()
                for column in range(3):
                    value = values[column] if column < len(values) else 0.0
                    mass[row][column] += factor * max(0.0, float(value))
        stats.hmm_transition_mass = mass
        stats.hmm_transition_buckets = (
            {current: [row[:] for row in mass]}
            if any(value > 0.0 for row in mass for value in row)
            else {}
        )

    @staticmethod
    def _normalize_belief(values: Sequence[float]) -> list[float]:
        cleaned = [max(0.0, float(value)) for value in values[:3]]
        if len(cleaned) < 3:
            cleaned.extend([0.0] * (3 - len(cleaned)))
        total = sum(cleaned)
        if not math.isfinite(total) or total <= 1e-12:
            return [1.0 / 3.0] * 3
        return [value / total for value in cleaned]

    def _transition_matrix(self, stats: StateStats, step: int | None = None) -> list[list[float]]:
        if step is not None:
            self._refresh_hmm_transitions(stats, int(step))
        matrix = []
        for row in range(3):
            values = [self.hmm_prior_alpha + max(0.0, float(stats.hmm_transition_mass[row][column])) for column in range(3)]
            matrix.append(self._normalize_belief(values))
        return matrix

    @staticmethod
    def _advance_belief(belief: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
        return AgentDPSStateMemory._normalize_belief([sum(float(belief[source]) * float(matrix[source][target]) for source in range(3)) for target in range(3)])

    def _belief_at(self, stats: StateStats, step: int) -> list[float]:
        matrix = self._transition_matrix(stats, step)
        if int(stats.hmm_last_class) >= 0:
            belief = [0.0, 0.0, 0.0]
            belief[int(stats.hmm_last_class)] = 1.0
            delta = max(0, int(step) - int(stats.hmm_last_observation_step))
        else:
            belief = self._normalize_belief(stats.hmm_belief)
            delta = max(0, int(step) - int(stats.hmm_belief_step))
        for _ in range(delta):
            belief = self._advance_belief(belief, matrix)
        return belief

    def _finalize_pending_hmm(self, stats: StateStats, current: int) -> None:
        pending_step = int(stats.hmm_pending_step)
        if pending_step < 0 or pending_step >= int(current):
            return
        success = float(stats.hmm_pending_success)
        failure = float(stats.hmm_pending_failure)
        if success > 0.0 and failure > 0.0:
            observed_class = 1
        elif success > 0.0:
            observed_class = 2
        else:
            observed_class = 0


        source_belief = self._belief_at(stats, pending_step)
        transition = self._transition_matrix(stats, pending_step)


        source_weights = [
            float(source_belief[source])
            * float(transition[source][observed_class])
            for source in range(3)
        ]
        normalizer = sum(source_weights)
        if normalizer > 1e-12 and math.isfinite(normalizer):
            source_weights = [value / normalizer for value in source_weights]
        else:
            source_weights = list(source_belief)
        bucket = stats.hmm_transition_buckets.setdefault(
            pending_step, [[0.0] * 3 for _ in range(3)]
        )
        for source, probability in enumerate(source_weights):
            bucket[source][observed_class] += probability
        self._refresh_hmm_transitions(stats, pending_step)

        stats.hmm_last_class = observed_class
        stats.hmm_last_observation_step = pending_step
        posterior = [0.0, 0.0, 0.0]
        posterior[observed_class] = 1.0
        stats.hmm_belief = self._advance_belief(posterior, self._transition_matrix(stats, pending_step))
        stats.hmm_belief_step = pending_step + 1
        stats.hmm_pending_step = -1
        stats.hmm_pending_success = 0.0
        stats.hmm_pending_failure = 0.0
        self.dirty = True

    def dps_belief(self, key: str, step: int) -> tuple[float, float, float]:

        self.prune(step)
        stats = self.states.get(str(key))
        if stats is None:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self._finalize_pending_hmm(stats, int(step))
        return tuple(self._belief_at(stats, int(step)))

    @staticmethod
    def _edge_id(source_key: str, action: str, next_key: str) -> str:
        payload = json.dumps(
            [str(source_key), action_key(action), str(next_key)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "agent_dps_edge:" + hashlib.sha256(payload).hexdigest()

    def _refresh_edge(self, edge: EdgeStats, current: int) -> None:

        current = int(current)
        visit_mass = 0.0
        success_mass = 0.0
        failure_mass = 0.0
        for bucket_step, counts in edge.evidence_buckets.items():
            age = max(0, current - int(bucket_step))
            factor = self.decay**age
            visit_mass += factor * max(0.0, float(counts[0] if counts else 0.0))
            success_mass += factor * max(0.0, float(counts[1] if len(counts) > 1 else 0.0))
            failure_mass += factor * max(0.0, float(counts[2] if len(counts) > 2 else 0.0))
        edge.visit_mass = visit_mass
        edge.success_mass = success_mass
        edge.failure_mass = failure_mass
        edge.evidence_buckets = (
            {current: [visit_mass, success_mass, failure_mass]}
            if visit_mass > 0.0
            else {}
        )
        edge.last_step = current

    def _rebuild_outgoing(self) -> None:
        self._outgoing = {}
        for edge_id, edge in self.edges.items():
            self._outgoing.setdefault(str(edge.source_key), set()).add(edge_id)

    def prune(self, step: int) -> int:

        current = int(step)
        if self._last_prune_step == current:
            return 0
        self.last_pruned = 0
        self.last_pruned_edges = 0
        self._last_prune_step = current
        return 0

    def _get(self, key: str, step: int) -> StateStats:
        key = str(key)
        current = int(step)
        stats = self.states.get(key)
        if stats is None:
            stats = StateStats(
                is_root=key in self.root_state_keys.values(),
                last_step=current,
                last_seen_step=current,
                hmm_belief_step=current,
            )
            self.states[key] = stats
            return stats
        self._finalize_pending_hmm(stats, current)
        self._refresh_stats(stats, current)
        return stats

    def evidence_counts(self, key: str, step: int) -> tuple[float, float]:
        self.posterior(key, step)
        stats = self.states.get(str(key))
        return (stats.success_mass, stats.failure_mass) if stats else (0.0, 0.0)

    def posterior(self, key: str, step: int) -> tuple[float, float, float, int]:


        self.prune(step)
        stats = self.states.get(str(key))
        if stats is None:
            return (
                self.alpha / max(self.alpha + self.beta, 1e-12),
                self.alpha
                * self.beta
                / max(
                    (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0),
                    1e-12,
                ),
                0.0,
                0,
            )
        self._finalize_pending_hmm(stats, int(step))
        self._refresh_stats(stats, int(step))
        total = self.alpha + self.beta + stats.success_mass + stats.failure_mass
        probability = (self.alpha + stats.success_mass) / max(total, 1e-12)
        variance = (self.alpha + stats.success_mass) * (self.beta + stats.failure_mass) / max(total * total * (total + 1.0), 1e-12)
        return (
            min(1.0, max(0.0, probability)),
            max(0.0, variance),
            stats.success_mass + stats.failure_mass,
            int(stats.visit_count),
        )

    def posterior_parameters(self, key: str, step: int) -> tuple[float, float]:

        self.posterior(key, step)
        stats = self.states.get(str(key))
        if stats is None:
            return self.alpha, self.beta
        return (
            self.alpha + max(0.0, float(stats.success_mass)),
            self.beta + max(0.0, float(stats.failure_mass)),
        )

    def has_recent_evidence(
        self,
        key: str,
        step: int,
        *,
        max_age_steps: int | None = None,
    ) -> bool:

        current = int(step)
        self.prune(current)
        stats = self.states.get(str(key))
        if stats is None or int(stats.visit_count) <= 0:
            return False
        horizon = (
            self.max_age_steps
            if max_age_steps is None
            else max(0, int(max_age_steps))
        )
        return current - int(stats.last_seen_step) <= horizon

    def _observe_value(
        self, key: str, reward: float | bool, step: int, *, record_hmm: bool = True
    ) -> None:
        stats = self._get(key, step)
        value = 1.0 if bool(float(reward) >= 0.5) else 0.0
        bucket = stats.outcome_buckets.setdefault(int(step), [0.0, 0.0])
        if value:
            bucket[0] += 1.0
        else:
            bucket[1] += 1.0
        self._refresh_stats(stats, int(step))
        stats.visit_count += 1
        stats.last_seen_step = int(step)
        if record_hmm:
            if stats.hmm_pending_step not in {-1, int(step)}:
                self._finalize_pending_hmm(stats, int(step))
            stats.hmm_pending_step = int(step)
            if value:
                stats.hmm_pending_success += 1.0
            else:
                stats.hmm_pending_failure += 1.0
        self.dirty = True

    def observe(self, key: str, reward: float | bool, step: int) -> None:
        self.prune(step)
        self._observe_value(str(key), reward, step)
        self._evict()

    def observe_unique(
        self,
        observations: Iterable[tuple[str, float | bool, str]],
        step: int,
        *,
        record_hmm: bool = True,
    ) -> int:

        self.prune(step)
        seen: set[tuple[str, str]] = set()
        count = 0
        for key, reward, trajectory_id in observations:
            marker = (str(key), str(trajectory_id))
            if marker in seen:
                continue
            seen.add(marker)
            self._observe_value(str(key), reward, step, record_hmm=record_hmm)
            count += 1
        self._evict()
        return count

    def observe_transitions(
        self,
        observations: Iterable[tuple[str, str, str, float | bool, str]],
        step: int,
    ) -> int:

        self.prune(step)
        seen: set[tuple[str, str]] = set()
        count = 0
        for source, action, next_state, reward, trajectory_id in observations:
            normalized_action = action_key(action)
            if not source or not normalized_action or not next_state:
                continue
            edge_id = self._edge_id(source, normalized_action, next_state)
            marker = (edge_id, str(trajectory_id))
            if marker in seen:
                continue
            seen.add(marker)
            edge = self.edges.get(edge_id)
            if edge is None:
                edge = EdgeStats(
                    source_key=str(source),
                    action=normalized_action,
                    next_key=str(next_state),
                    last_step=int(step),
                    last_seen_step=int(step),
                )
                self.edges[edge_id] = edge
                self._outgoing.setdefault(str(source), set()).add(edge_id)
            else:
                self._refresh_edge(edge, int(step))
            value = 1.0 if bool(float(reward) >= 0.5) else 0.0
            bucket = edge.evidence_buckets.setdefault(int(step), [0.0, 0.0, 0.0])
            bucket[0] += 1.0
            if value:
                bucket[1] += 1.0
            else:
                bucket[2] += 1.0
            self._refresh_edge(edge, int(step))
            edge.last_seen_step = int(step)
            count += 1
        if count:
            self.dirty = True
            self._evict_edges()
        return count

    def terminal_contrast_posterior(
        self,
        key: str,
        step: int,
        *,
        factual_outcome: float | bool,
        current_outcomes: Sequence[tuple[float | bool, str]] = (),
    ) -> TerminalContrastPosterior:

        alpha, beta = self.posterior_parameters(str(key), int(step))
        history_mass = max(0.0, alpha - self.alpha) + max(0.0, beta - self.beta)
        seen: set[str] = set()
        for outcome, trajectory in current_outcomes:
            if str(trajectory) in seen:
                continue
            seen.add(str(trajectory))
            if float(outcome) >= 0.5:
                alpha += 1.0
            else:
                beta += 1.0
        if float(factual_outcome) >= 0.5:
            alpha, beta = beta, alpha
        return TerminalContrastPosterior(
            alpha=alpha, beta=beta,
            history_mass=history_mass, current_mass=float(len(seen)),
        )

    def transition_contrast_posterior(
        self,
        key: str,
        step: int,
        *,
        factual_action: str = "",
        factual_next_state: str = "",
        factual_outcome: float | bool = False,
        current_outcomes: Sequence[tuple[float | bool, str]] = (),
        current_transitions: Sequence[
            tuple[str, str, float | bool, str]
        ] = (),
    ) -> TransitionContrastPosterior:

        current = int(step)
        self.prune(current)
        historical_alpha, historical_beta = self.posterior_parameters(
            str(key), current
        )
        historical_success = max(0.0, historical_alpha - self.alpha)
        historical_failure = max(0.0, historical_beta - self.beta)

        seen_outcomes: set[str] = set()
        current_success = 0.0
        current_failure = 0.0
        for outcome, trajectory_id in current_outcomes:
            marker = str(trajectory_id)
            if marker in seen_outcomes:
                continue
            seen_outcomes.add(marker)
            if bool(float(outcome) >= 0.5):
                current_success += 1.0
            else:
                current_failure += 1.0

        posterior_alpha = historical_alpha + current_success
        posterior_beta = historical_beta + current_failure
        factual_success = bool(float(factual_outcome) >= 0.5)
        opposite_prior = self.beta if factual_success else self.alpha
        opposite_observed = (
            historical_failure + current_failure
            if factual_success
            else historical_success + current_success
        )

        normalized_factual_action = action_key(factual_action)
        factual_edge_id = ""
        if normalized_factual_action and factual_next_state:
            factual_edge_id = self._edge_id(
                str(key), normalized_factual_action, str(factual_next_state)
            )

        known_edge_ids = set(self._outgoing.get(str(key), set()))
        current_edge_rows: dict[str, tuple[float, float]] = {}
        seen_transitions: set[tuple[str, str]] = set()
        for action, next_state, outcome, trajectory_id in current_transitions:
            normalized_action = action_key(action)
            if not normalized_action or not next_state:
                continue
            edge_id = self._edge_id(str(key), normalized_action, str(next_state))
            marker = (edge_id, str(trajectory_id))
            if marker in seen_transitions:
                continue
            seen_transitions.add(marker)
            known_edge_ids.add(edge_id)
            success, failure = current_edge_rows.get(edge_id, (0.0, 0.0))
            if bool(float(outcome) >= 0.5):
                success += 1.0
            else:
                failure += 1.0
            current_edge_rows[edge_id] = (success, failure)

        same_edge_opposite = 0.0
        if factual_edge_id:
            known_edge_ids.add(factual_edge_id)
            edge = self.edges.get(factual_edge_id)
            if edge is not None:
                self._refresh_edge(edge, current)
                same_edge_opposite += (
                    edge.failure_mass if factual_success else edge.success_mass
                )
            current_edge = current_edge_rows.get(factual_edge_id, (0.0, 0.0))
            same_edge_opposite += (
                current_edge[1] if factual_success else current_edge[0]
            )


        edge_count = len(known_edge_ids)
        same_edge_prior = (
            opposite_prior / max(1, edge_count + 1)
            if factual_edge_id else 0.0
        )
        same_edge_opposite = min(
            max(0.0, same_edge_opposite), max(0.0, opposite_observed)
        )
        event_alpha = (
            opposite_prior - same_edge_prior
            + max(0.0, opposite_observed - same_edge_opposite)
        )
        total_mass = posterior_alpha + posterior_beta
        event_alpha = min(max(1e-12, event_alpha), max(1e-12, total_mass))
        event_beta = max(1e-12, total_mass - event_alpha)
        opposite_total = max(1e-12, opposite_prior + opposite_observed)
        same_edge_probability = (
            (same_edge_prior + same_edge_opposite) / opposite_total
            if factual_edge_id else 0.0
        )
        return TransitionContrastPosterior(
            alpha=event_alpha,
            beta=event_beta,
            history_mass=historical_success + historical_failure,
            current_mass=current_success + current_failure,
            edge_count=edge_count,
            same_edge_opposite_probability=min(
                1.0, max(0.0, same_edge_probability)
            ),
            factual_edge_available=bool(factual_edge_id),
        )

    def _state_graph_value(
        self,
        key: str,
        step: int,
        depth: int,
        min_edge_evidence: float,
        visited: set[str],
    ) -> float:
        probability, _, _, _ = self.posterior(key, step)
        if depth <= 0 or key in visited:
            return probability
        edge_ids = self._outgoing.get(str(key), set())
        eligible = []
        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if edge is None:
                continue
            self._refresh_edge(edge, int(step))
            if edge.visit_mass >= min_edge_evidence:
                eligible.append(edge)
        if not eligible:
            return probability
        total = sum(edge.visit_mass for edge in eligible)
        values = []
        for edge in eligible:
            edge_total = self.alpha + self.beta + edge.success_mass + edge.failure_mass
            direct = (self.alpha + edge.success_mass) / max(edge_total, 1e-12)
            successor = self._state_graph_value(
                edge.next_key,
                step,
                depth - 1,
                min_edge_evidence,
                visited | {str(key)},
            )
            confidence = edge.visit_mass / (edge.visit_mass + self.alpha + self.beta)
            values.append((edge.visit_mass, confidence * direct + (1.0 - confidence) * successor))
        return sum(weight * value for weight, value in values) / max(total, 1e-12)

    def graph_signal(
        self,
        key: str,
        step: int,
        *,
        current_transitions: Sequence[tuple[str, str]] = (),
        max_depth: int = 1,
        min_edge_evidence: float = 1.0,
        uncertainty_weight: float = 0.25,
    ) -> tuple[float, float, int, float]:

        self.prune(step)
        action_values: dict[str, list[tuple[float, float, float]]] = {}
        historical_mass = 0.0
        for edge_id in tuple(self._outgoing.get(str(key), set())):
            edge = self.edges.get(edge_id)
            if edge is None:
                continue
            self._refresh_edge(edge, int(step))
            if edge.visit_mass < float(min_edge_evidence):
                continue
            edge_total = self.alpha + self.beta + edge.success_mass + edge.failure_mass
            direct = (self.alpha + edge.success_mass) / max(edge_total, 1e-12)
            direct_variance = (self.alpha + edge.success_mass) * (self.beta + edge.failure_mass) / max(edge_total * edge_total * (edge_total + 1.0), 1e-12)
            successor = self._state_graph_value(
                edge.next_key,
                int(step),
                max(0, int(max_depth) - 1),
                float(min_edge_evidence),
                {str(key)},
            )
            confidence = edge.visit_mass / (edge.visit_mass + self.alpha + self.beta)
            value = confidence * direct + (1.0 - confidence) * successor
            action_values.setdefault(edge.action, []).append((edge.visit_mass, value, direct_variance))
            historical_mass += edge.visit_mass

        for action, next_key in current_transitions:
            normalized_action = action_key(action)
            if not normalized_action or not next_key:
                continue
            successor, variance, _, _ = self.posterior(str(next_key), int(step))
            action_values.setdefault(normalized_action, []).append((1.0, successor, variance))

        collapsed = []
        for action, values in action_values.items():
            total = sum(item[0] for item in values)
            if total <= 0.0:
                continue
            collapsed.append(
                (
                    action,
                    total,
                    sum(weight * value for weight, value, _ in values) / total,
                    sum(weight * variance for weight, _, variance in values) / total,
                )
            )
        if len(collapsed) < 2:
            return 0.0, 0.0, len(collapsed), historical_mass
        total = sum(item[1] for item in collapsed)
        mean = sum(weight * value for _, weight, value, _ in collapsed) / total
        dispersion = sum(weight * (value - mean) ** 2 for _, weight, value, _ in collapsed) / total
        uncertainty = sum(weight * variance for _, weight, _, variance in collapsed) / total
        signal = min(
            1.0,
            max(0.0, 4.0 * dispersion + float(uncertainty_weight) * 12.0 * uncertainty),
        )
        return signal, min(1.0, 4.0 * dispersion), len(collapsed), historical_mass

    def _evict(self) -> None:
        overflow = len(self.states) - self.max_states
        if overflow <= 0:
            return
        victims = sorted(
            self.states.items(),
            key=lambda item: (item[1].last_seen_step, item[1].visit_count, item[0]),
        )[:overflow]
        for key, _ in victims:
            self.states.pop(key, None)

    def _evict_edges(self) -> None:
        overflow = len(self.edges) - self.max_edges
        if overflow <= 0:
            return
        victims = sorted(
            self.edges.items(),
            key=lambda item: (item[1].last_seen_step, item[1].visit_mass, item[0]),
        )[:overflow]
        for edge_id, edge in victims:
            self.edges.pop(edge_id, None)
            outgoing = self._outgoing.get(str(edge.source_key))
            if outgoing is not None:
                outgoing.discard(edge_id)
                if not outgoing:
                    self._outgoing.pop(str(edge.source_key), None)

    def metrics(self, step: int) -> dict[str, float]:
        self.prune(step)
        pruned_last = self.last_pruned
        pruned_edges_last = self.last_pruned_edges
        if not self.states:
            return {
                "agent_dps/state_memory/n_states": 0.0,
                "agent_dps/state_memory/mean_p": 0.5,
                "agent_dps/state_memory/mean_visits": 0.0,
                "agent_dps/state_memory/pruned_last": float(pruned_last),
                "agent_dps/state_memory/pruned_total": float(self.pruned_total),
                "agent_dps/state_memory/n_edges": float(len(self.edges)),
                "agent_dps/state_memory/pruned_edges_last": float(pruned_edges_last),
                "agent_dps/state_memory/pruned_edges_total": float(self.pruned_edges_total),
                "agent_dps/state_memory/hmm_partial_mean": 1.0 / 3.0,
            }
        probabilities = []
        visits = []
        partial_probabilities = []
        for key in self.states:
            probability, _, _, visit_count = self.posterior(key, step)
            probabilities.append(probability)
            visits.append(float(visit_count))
            partial_probabilities.append(self.dps_belief(key, step)[1])


        self.last_pruned = pruned_last
        self.last_pruned_edges = pruned_edges_last
        return {
            "agent_dps/state_memory/n_states": float(len(self.states)),
            "agent_dps/state_memory/mean_p": float(sum(probabilities) / len(probabilities)),
            "agent_dps/state_memory/min_p": float(min(probabilities)),
            "agent_dps/state_memory/max_p": float(max(probabilities)),
            "agent_dps/state_memory/mean_visits": float(sum(visits) / len(visits)),
            "agent_dps/state_memory/pruned_last": float(pruned_last),
            "agent_dps/state_memory/pruned_total": float(self.pruned_total),
            "agent_dps/state_memory/n_edges": float(len(self.edges)),
            "agent_dps/state_memory/pruned_edges_last": float(pruned_edges_last),
            "agent_dps/state_memory/pruned_edges_total": float(self.pruned_edges_total),
            "agent_dps/state_memory/hmm_partial_mean": float(sum(partial_probabilities) / len(partial_probabilities)),
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "schema_version": 6,
            "alpha": self.alpha,
            "beta": self.beta,
            "decay": self.decay,
            "root_decay": self.root_decay,
            "max_states": self.max_states,
            "max_age_steps": self.max_age_steps,
            "hmm_prior_alpha": self.hmm_prior_alpha,
            "hmm_transition_decay": self.hmm_transition_decay,
            "max_edges": self.max_edges,
            "task_sampler_state": self.task_sampler_state,
            "root_state_keys": self.root_state_keys,
            "survival_memory": self.survival_memory,
            "probability_calibration": self.probability_calibration,
            "states": {
                key: {
                    "success_mass": value.success_mass,
                    "failure_mass": value.failure_mass,
                    "last_step": value.last_step,
                    "visit_count": value.visit_count,
                    "last_seen_step": value.last_seen_step,
                    "outcome_buckets": value.outcome_buckets,
                    "hmm_transition_mass": value.hmm_transition_mass,
                    "hmm_transition_buckets": value.hmm_transition_buckets,
                    "hmm_belief": value.hmm_belief,
                    "hmm_belief_step": value.hmm_belief_step,
                    "hmm_last_class": value.hmm_last_class,
                    "hmm_last_observation_step": value.hmm_last_observation_step,
                    "hmm_pending_step": value.hmm_pending_step,
                    "hmm_pending_success": value.hmm_pending_success,
                    "hmm_pending_failure": value.hmm_pending_failure,
                }
                for key, value in self.states.items()
            },
            "edges": {
                edge_id: {
                    "source_key": edge.source_key,
                    "action": edge.action,
                    "next_key": edge.next_key,
                    "visit_mass": edge.visit_mass,
                    "success_mass": edge.success_mass,
                    "failure_mass": edge.failure_mass,
                    "last_step": edge.last_step,
                    "last_seen_step": edge.last_seen_step,
                    "evidence_buckets": edge.evidence_buckets,
                }
                for edge_id, edge in self.edges.items()
            },
        }
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, sort_keys=True)
        os.replace(temporary, path)
        self.dirty = False

    @classmethod
    def load_or_create(
        cls,
        path: str,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.90,
        root_decay: float | None = None,
        max_states: int = 100_000,
        max_age_steps: int = 32,
        hmm_prior_alpha: float = 1.0,
        hmm_transition_decay: float = 0.9,
        max_edges: int = 200_000,
    ) -> AgentDPSStateMemory:
        memory = cls(
            alpha=alpha,
            beta=beta,
            decay=decay,
            root_decay=root_decay,
            max_states=max_states,
            max_age_steps=max_age_steps,
            hmm_prior_alpha=hmm_prior_alpha,
            hmm_transition_decay=hmm_transition_decay,
            max_edges=max_edges,
        )
        if not os.path.isfile(path):
            return memory
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            memory.alpha = float(data.get("alpha", memory.alpha))
            memory.beta = float(data.get("beta", memory.beta))
            memory.task_sampler_state = dict(
                data.get("task_sampler_state", {}) or {}
            )
            memory.survival_memory = dict(data.get("survival_memory", {}) or {})
            memory.probability_calibration = dict(data.get("probability_calibration", {}) or {})
            memory.root_state_keys = {
                str(task_ref): str(key)
                for task_ref, key in dict(
                    data.get("root_state_keys", {}) or {}
                ).items()
            }


            memory.max_states = max(1, int(max_states))
            root_keys = set(memory.root_state_keys.values())
            for key, raw in dict(data.get("states", {})).items():
                last_step = int(raw.get("last_step", 0))
                outcome_buckets = {
                    int(bucket_step): [
                        float(counts[0] if counts else 0.0),
                        float(counts[1] if len(counts) > 1 else 0.0),
                    ]
                    for bucket_step, counts in dict(raw.get("outcome_buckets", {})).items()
                }
                if not outcome_buckets:
                    legacy_success = float(raw.get("success_mass", 0.0))
                    legacy_failure = float(raw.get("failure_mass", 0.0))
                    if legacy_success > 0.0 or legacy_failure > 0.0:
                        outcome_buckets[last_step] = [
                            legacy_success,
                            legacy_failure,
                        ]
                hmm_transition_mass = [[float(value) for value in row[:3]] for row in raw.get("hmm_transition_mass", [[0.0] * 3 for _ in range(3)])[:3]]
                while len(hmm_transition_mass) < 3:
                    hmm_transition_mass.append([0.0] * 3)
                for row in hmm_transition_mass:
                    row.extend([0.0] * (3 - len(row)))
                hmm_transition_buckets = {int(bucket_step): [[float(value) for value in row[:3]] + [0.0] * max(0, 3 - len(row)) for row in list(matrix)[:3]] for bucket_step, matrix in dict(raw.get("hmm_transition_buckets", {})).items()}
                for matrix in hmm_transition_buckets.values():
                    while len(matrix) < 3:
                        matrix.append([0.0] * 3)
                if not hmm_transition_buckets and any(value > 0.0 for row in hmm_transition_mass for value in row):
                    transition_step = int(raw.get("hmm_last_observation_step", last_step))
                    if transition_step < 0:
                        transition_step = last_step
                    hmm_transition_buckets[transition_step] = [row[:] for row in hmm_transition_mass]
                memory.states[str(key)] = StateStats(
                    is_root=str(key) in root_keys,
                    success_mass=float(raw.get("success_mass", 0.0)),
                    failure_mass=float(raw.get("failure_mass", 0.0)),
                    last_step=last_step,
                    visit_count=int(raw.get("visit_count", 0)),
                    last_seen_step=int(raw.get("last_seen_step", 0)),
                    outcome_buckets=outcome_buckets,
                    hmm_transition_mass=hmm_transition_mass,
                    hmm_transition_buckets=hmm_transition_buckets,
                    hmm_belief=memory._normalize_belief(raw.get("hmm_belief", [1.0 / 3.0] * 3)),
                    hmm_belief_step=int(raw.get("hmm_belief_step", raw.get("last_step", 0))),
                    hmm_last_class=int(raw.get("hmm_last_class", -1)),
                    hmm_last_observation_step=int(raw.get("hmm_last_observation_step", -1)),
                    hmm_pending_step=int(raw.get("hmm_pending_step", -1)),
                    hmm_pending_success=float(raw.get("hmm_pending_success", 0.0)),
                    hmm_pending_failure=float(raw.get("hmm_pending_failure", 0.0)),
                )
            for edge_id, raw in dict(data.get("edges", {})).items():
                last_step = int(raw.get("last_step", 0))
                evidence_buckets = {
                    int(bucket_step): [
                        float(counts[0] if counts else 0.0),
                        float(counts[1] if len(counts) > 1 else 0.0),
                        float(counts[2] if len(counts) > 2 else 0.0),
                    ]
                    for bucket_step, counts in dict(raw.get("evidence_buckets", {})).items()
                }
                if not evidence_buckets:
                    legacy_visit = float(raw.get("visit_mass", 0.0))
                    legacy_success = float(raw.get("success_mass", 0.0))
                    legacy_failure = float(raw.get("failure_mass", 0.0))
                    if legacy_visit > 0.0:
                        evidence_buckets[last_step] = [
                            legacy_visit,
                            legacy_success,
                            legacy_failure,
                        ]
                memory.edges[str(edge_id)] = EdgeStats(
                    source_key=str(raw.get("source_key", "")),
                    action=action_key(raw.get("action", "")),
                    next_key=str(raw.get("next_key", "")),
                    visit_mass=float(raw.get("visit_mass", 0.0)),
                    success_mass=float(raw.get("success_mass", 0.0)),
                    failure_mass=float(raw.get("failure_mass", 0.0)),
                    last_step=last_step,
                    last_seen_step=int(raw.get("last_seen_step", 0)),
                    evidence_buckets=evidence_buckets,
                )
            memory._rebuild_outgoing()
            memory._evict()
            memory._evict_edges()
            memory.dirty = int(data.get("schema_version", 1)) < 4
        except Exception:


            memory = cls(
                alpha=alpha,
                beta=beta,
                decay=decay,
                max_states=max_states,
                max_age_steps=max_age_steps,
                hmm_prior_alpha=hmm_prior_alpha,
                hmm_transition_decay=hmm_transition_decay,
                max_edges=max_edges,
            )
        return memory


def mixed_utility(probability: float, count: int) -> float:

    p = min(1.0, max(0.0, float(probability)))
    k = int(count)
    if k < 2:
        return 0.0
    return max(0.0, 1.0 - p**k - (1.0 - p) ** k)


def beta_mixed_utility(alpha: float, beta: float, count: int) -> float:

    a = max(1e-12, float(alpha))
    b = max(1e-12, float(beta))
    k = int(count)
    if k < 2:
        return 0.0
    total = a + b
    all_success = math.exp(
        math.lgamma(a + k) - math.lgamma(a)
        + math.lgamma(total) - math.lgamma(total + k)
    )
    all_failure = math.exp(
        math.lgamma(b + k) - math.lgamma(b)
        + math.lgamma(total) - math.lgamma(total + k)
    )
    return min(1.0, max(0.0, 1.0 - all_success - all_failure))


def beta_event_utility(alpha: float, beta: float, count: int) -> float:

    a = max(1e-12, float(alpha))
    b = max(1e-12, float(beta))
    k = int(count)
    if k <= 0:
        return 0.0
    total = a + b
    no_event = math.exp(
        math.lgamma(b + k) - math.lgamma(b)
        + math.lgamma(total) - math.lgamma(total + k)
    )
    return min(1.0, max(0.0, 1.0 - no_event))


def expected_beta_mixed_gain(
    alpha: float,
    beta: float,
    count: int,
    existing_count: int = 0,
) -> float:

    k = int(count)
    existing = max(0, int(existing_count))
    if k <= 0:
        return 0.0
    return max(
        0.0,
        beta_mixed_utility(alpha, beta, existing + k)
        - beta_mixed_utility(alpha, beta, existing),
    )


def conditional_mixed_utility(
    probability: float,
    count: int,
    factual_outcomes: Sequence[float | bool],
) -> float:

    k = int(count)
    if k <= 0:
        return 0.0
    observed = [1 if bool(float(value) >= 0.5) else 0 for value in factual_outcomes]
    if not observed:
        return mixed_utility(probability, k)
    if any(value == 1 for value in observed) and any(value == 0 for value in observed):
        return 0.0
    p = min(1.0, max(0.0, float(probability)))
    if observed[0] == 1:
        return max(0.0, 1.0 - p**k)
    return max(0.0, 1.0 - (1.0 - p) ** k)


def expected_mixed_gain(
    probability: float,
    count: int,
    existing_count: int = 1,
) -> float:

    k = int(count)
    if k <= 0:
        return 0.0
    existing = max(0, int(existing_count))
    return max(
        0.0,
        mixed_utility(probability, existing + k) - mixed_utility(probability, existing),
    )


@dataclass
class AgentDPSNode:
    state_key: str
    task_uid: str
    anchor_id: str
    trajectory_id: str
    probability: float
    uncertainty: float
    posterior_alpha: float = 0.0
    posterior_beta: float = 0.0
    hmm_partial_probability: float = 1.0 / 3.0
    graph_score: float = 0.0
    graph_dispersion: float = 0.0
    graph_action_count: int = 0
    graph_history_mass: float = 0.0


    acquisition_score: float = 1.0
    contrast_alpha: float = 0.0
    contrast_beta: float = 0.0
    contrast_history_mass: float = 0.0
    contrast_current_mass: float = 0.0
    contrast_edge_count: int = 0
    contrast_same_edge_opposite_probability: float = 0.0
    factual_edge_available: bool = False
    factual_action: str = ""
    factual_next_state: str = ""
    factual_outcome: float = 0.0
    factual_outcomes: tuple[float, ...] = ()


    existing_factual_count: int = 0
    historical_evidence_count: float = 0.0

    tie_breaker: float = 0.0
    candidate_count: int = 1
    representative: Any = None
    current_k: int = 0
    max_k: int = 0
    opened: bool = False
    contrast_objective: str = "transition_contrast"

    opposite_particles: tuple[float, ...] = ()

    root_utility_curve: tuple[float, ...] = ()

    @property
    def homogeneous(self) -> bool:
        return bool(self.factual_outcomes) and len(set(self.factual_outcomes)) == 1

    @property
    def uses_transition_contrast(self) -> bool:
        return self.uses_branch_contrast and self.contrast_objective == "transition_contrast"

    @property
    def uses_terminal_contrast(self) -> bool:
        return self.uses_branch_contrast and self.contrast_objective == "terminal_contrast"

    @property
    def uses_branch_contrast(self) -> bool:
        return (
            self.contrast_alpha > 0.0 and self.contrast_beta > 0.0
            and self.contrast_objective in {"terminal_contrast", "transition_contrast"}
        )


@dataclass(frozen=True)
class AgentDPSAllocation:
    anchor_id: str
    trajectory_id: str
    continuation_count: int
    predicted_mean: float
    predicted_uncertainty: float
    external_cost: float
    score: float
    provenance: str = "agent_dps"
    selection_probability: float = 1.0
    cumulative_utility: float = 0.0
    marginal_utility: float = 0.0
    marginal_cost: float = 1.0
    marginal_efficiency: float = 0.0
    marginal_score: float = 0.0
    marginal_score_efficiency: float = 0.0
    state_group_key: str = ""


    factual_count: int = 0


    physical_factual_count: int = 0

    @property
    def predicted_cost(self) -> float:
        return self.external_cost


@dataclass(frozen=True)
class AgentDPSCurriculumCandidate:

    task_ref: str
    state_key: str = ""
    frontier_probability: float = 1.0 / 3.0
    evidence_mass: float = 0.0


def select_agent_dps_curriculum(
    candidates: Sequence[AgentDPSCurriculumCandidate],
    count: int,
    *,
    exploration_fraction: float = 0.30,
    temperature: float = 0.25,
    seed: int = 0,
) -> tuple[list[AgentDPSCurriculumCandidate], dict[str, float]]:
    """Choose root task candidates from calibrated evidence."""

    population = list(candidates)
    target = int(count)
    if target < 0 or target > len(population):
        raise ValueError("curriculum sample count must be within candidate pool")
    if target == 0:
        return [], {
            "agent_dps/root/curriculum_candidate_count": float(len(population)),
            "agent_dps/root/curriculum_selected_count": 0.0,
        }
    epsilon = min(1.0, max(0.0, float(exploration_fraction)))
    tau = max(1e-6, float(temperature))
    scores = [
        min(1.0, max(0.0, float(item.frontier_probability)))
        for item in population
    ]
    max_score = max(scores)
    logits = [math.exp((score - max_score) / tau) for score in scores]
    logits_total = max(sum(logits), 1e-12)
    curriculum = [value / logits_total for value in logits]
    weights = [
        epsilon / len(population) + (1.0 - epsilon) * value
        for value in curriculum
    ]

    rng = random.Random(int(seed))
    remaining = list(range(len(population)))
    selected_indices: list[int] = []
    for _ in range(target):
        total = sum(weights[index] for index in remaining)
        draw = rng.random() * total
        cumulative = 0.0
        chosen = remaining[-1]
        for index in remaining:
            cumulative += weights[index]
            if draw <= cumulative:
                chosen = index
                break
        selected_indices.append(chosen)
        remaining.remove(chosen)

    selected = [population[index] for index in selected_indices]
    selected_scores = [scores[index] for index in selected_indices]
    seen = [float(item.evidence_mass > 0.0) for item in population]
    selected_seen = [float(item.evidence_mass > 0.0) for item in selected]
    metrics = {
        "agent_dps/root/curriculum_candidate_count": float(len(population)),
        "agent_dps/root/curriculum_selected_count": float(len(selected)),
        "agent_dps/root/curriculum_exploration_fraction": float(epsilon),
        "agent_dps/root/curriculum_temperature": float(tau),
        "agent_dps/root/candidate_frontier_mean": float(
            sum(scores) / max(1, len(scores))
        ),
        "agent_dps/root/selected_frontier_mean": float(
            sum(selected_scores) / max(1, len(selected_scores))
        ),
        "agent_dps/root/frontier_selection_lift": float(
            sum(selected_scores) / max(1, len(selected_scores))
            - sum(scores) / max(1, len(scores))
        ),
        "agent_dps/root/candidate_history_seen_fraction": float(
            sum(seen) / max(1, len(seen))
        ),
        "agent_dps/root/selected_history_seen_fraction": float(
            sum(selected_seen) / max(1, len(selected_seen))
        ),
    }
    return selected, metrics


def allocate_agent_dps(
    nodes: Sequence[AgentDPSNode],
    budget: int,
    *,
    min_new_branch: int = 4,
    initial_branch_size: int = 4,
    max_k: int | None = None,
    reject_homogeneous: bool = False,
    root_mode: bool = False,
    root_baseline: int = 0,
    task_min_coverage: bool = False,
    task_max_budget_factor: float | None = None,
    task_diminishing_exponent: float = 0.0,
    candidate_top_m: int | None = None,
) -> tuple[list[AgentDPSAllocation], dict[str, float], list[dict[str, Any]]]:
    """Allocate factual root sampling under the global budget."""

    remaining = max(0, int(budget))
    if not nodes or remaining <= 0:
        return [], {"agent_dps/allocator/budget": float(remaining)}, []
    opening_size = max(1, int(initial_branch_size))
    min_new = max(1, int(min_new_branch))
    if root_mode:
        opening_size = max(2, opening_size)
        min_new = max(2, min_new)
    opening_size = max(opening_size, min_new)
    represented_tasks = sorted({str(node.task_uid) for node in nodes})
    target_task_budget = float(remaining) / max(1, len(represented_tasks))
    task_allocated = {task: 0 for task in represented_tasks}
    records = []
    for node in nodes:
        node.current_k = 0
        node.opened = False
        configured_limit = max(0, int(max_k if max_k is not None else remaining))


        if max_k is None:
            node.max_k = configured_limit
        elif int(node.max_k) > 0:
            node.max_k = min(int(node.max_k), configured_limit)
        else:
            node.max_k = configured_limit
        records.append(
            {
                "state_group_key": node.state_key,
                "candidate_count": int(node.candidate_count),
                "selected_count": 0,
                "allocated_k": 0,
                "selected_anchor_ids": [],
                "selected_k": [],
                "candidate_anchor_ids": [str(node.anchor_id)],
                "candidate_node_ids": [str(getattr(node.representative, "node_id", ""))],
                "candidate_trajectory_ids": [str(node.trajectory_id)],
                "candidate_steps": [int(getattr(node.representative, "step_idx", 0))],
            }
        )
    allocations: dict[str, AgentDPSAllocation] = {}

    def branch_utility(node: AgentDPSNode, count: int) -> float:
        if node.uses_branch_contrast:
            return beta_event_utility(
                node.contrast_alpha, node.contrast_beta, count
            )

        if node.factual_outcomes:
            return conditional_mixed_utility(node.probability, count, node.factual_outcomes)
        if node.posterior_alpha > 0.0 and node.posterior_beta > 0.0:
            return expected_beta_mixed_gain(
                node.posterior_alpha,
                node.posterior_beta,
                count,
                existing_count=node.existing_factual_count,
            )
        return expected_mixed_gain(
            node.probability,
            count,
            existing_count=node.existing_factual_count,
        )

    def scaled_utility(node: AgentDPSNode, count: int) -> float:


        if root_mode and node.root_utility_curve:
            return float(node.root_utility_curve[count])
        if root_mode and node.posterior_alpha > 0.0 and node.posterior_beta > 0.0:
            return beta_mixed_utility(
                node.posterior_alpha, node.posterior_beta, count
            )
        return mixed_utility(node.probability, count) if root_mode else branch_utility(node, count)

    contrast_objective = (
        not root_mode and bool(nodes)
        and all(node.uses_branch_contrast for node in nodes)
    )
    allocation_nodes = list(nodes)
    requested_top_m = len(nodes)
    if not contrast_objective and not root_mode and candidate_top_m is not None:
        requested_top_m = max(1, min(len(nodes), int(candidate_top_m)))
        allocation_nodes = sorted(
            nodes,
            key=lambda node: (
                float(node.acquisition_score),
                float(node.tie_breaker),
                str(node.state_key),
            ),
            reverse=True,
        )[:requested_top_m]


    baseline = max(0, int(root_baseline)) if root_mode else 0
    if baseline:
        baseline = max(opening_size, baseline)
        eligible = [
            node
            for node in allocation_nodes
            if not (reject_homogeneous and node.homogeneous)
            and int(node.max_k) >= baseline
        ]
        if eligible and baseline * len(eligible) <= remaining:
            for node in eligible:
                node.current_k = baseline
                node.opened = True
                remaining -= baseline
                task_allocated[str(node.task_uid)] = (
                    task_allocated.get(str(node.task_uid), 0) + baseline
                )
                utility = scaled_utility(node, baseline)
                allocations[node.state_key] = AgentDPSAllocation(
                    anchor_id=str(node.anchor_id),
                    trajectory_id=str(node.trajectory_id),
                    continuation_count=baseline,
                    predicted_mean=float(node.probability),
                    predicted_uncertainty=float(math.sqrt(max(0.0, node.uncertainty))),
                    external_cost=float(baseline),
                    score=float(utility),
                    cumulative_utility=float(utility),
                    marginal_utility=float(utility),
                    marginal_cost=float(baseline),
                    marginal_efficiency=float(utility / baseline),
                    marginal_score=float(utility),
                    marginal_score_efficiency=float(utility / baseline),
                    state_group_key=str(node.state_key),
                    factual_count=len(node.factual_outcomes),
                    physical_factual_count=0,
                )


    if contrast_objective and remaining > 0:
        ordered_nodes = sorted(
            [
                node for node in allocation_nodes
                if not (reject_homogeneous and node.homogeneous)
            ],
            key=lambda node: (
                -float(node.tie_breaker),
                str(node.state_key),
            ),
        )
        budget_limit = remaining
        negative_infinity = float("-inf")
        dp = [negative_infinity] * (budget_limit + 1)
        dp[0] = 0.0
        parent_budgets: list[list[int]] = []
        parent_choices: list[list[int]] = []
        for node in ordered_nodes:
            next_dp = list(dp)
            previous_for_budget = list(range(budget_limit + 1))
            choice_for_budget = [0] * (budget_limit + 1)
            local_limit = min(int(node.max_k), budget_limit)
            options = [
                (count, branch_utility(node, count))
                for count in range(opening_size, local_limit + 1)
            ]
            for used, previous_score in enumerate(dp):
                if not math.isfinite(previous_score):
                    continue
                for count, utility in options:
                    total = used + count
                    if total > budget_limit:
                        break
                    score = previous_score + utility
                    if score > next_dp[total] + 1e-12:
                        next_dp[total] = score
                        previous_for_budget[total] = used
                        choice_for_budget[total] = count
            dp = next_dp
            parent_budgets.append(previous_for_budget)
            parent_choices.append(choice_for_budget)

        best_used = max(
            range(budget_limit + 1),
            key=lambda used: (dp[used], used),
        )
        selected_counts: dict[str, int] = {}
        cursor = best_used
        for index in range(len(ordered_nodes) - 1, -1, -1):
            count = parent_choices[index][cursor]
            previous = parent_budgets[index][cursor]
            if count > 0:
                selected_counts[ordered_nodes[index].state_key] = count
            cursor = previous

        for node in ordered_nodes:
            count = int(selected_counts.get(node.state_key, 0))
            if count <= 0:
                continue
            node.current_k = count
            node.opened = True
            utility = branch_utility(node, count)
            previous_utility = branch_utility(node, max(0, count - 1))
            marginal = max(0.0, utility - previous_utility)
            task_allocated[str(node.task_uid)] = (
                task_allocated.get(str(node.task_uid), 0) + count
            )
            allocations[node.state_key] = AgentDPSAllocation(
                anchor_id=str(node.anchor_id),
                trajectory_id=str(node.trajectory_id),
                continuation_count=count,
                predicted_mean=float(node.probability),
                predicted_uncertainty=float(
                    math.sqrt(max(0.0, node.uncertainty))
                ),
                external_cost=float(count),
                score=float(utility),
                cumulative_utility=float(utility),
                marginal_utility=float(marginal),
                marginal_cost=1.0,
                marginal_efficiency=float(marginal),
                marginal_score=float(marginal),
                marginal_score_efficiency=float(marginal),
                state_group_key=str(node.state_key),
                factual_count=max(
                    len(node.factual_outcomes), int(node.existing_factual_count)
                ),
                physical_factual_count=max(1, int(node.existing_factual_count)),
            )
        remaining -= best_used

    while remaining > 0 and not contrast_objective:
        choices: list[tuple[float, int, float, str, AgentDPSNode, float, float]] = []
        for node in allocation_nodes:
            if reject_homogeneous and node.homogeneous:
                continue
            current = int(node.current_k)
            if current == 0:
                cost = opening_size
                if cost > remaining or cost > node.max_k:
                    continue
                utility = scaled_utility(node, cost)
                gain = utility
            else:
                cost = 1
                if current >= node.max_k:
                    continue
                previous = scaled_utility(node, current)
                utility = scaled_utility(node, current + 1)
                gain = utility - previous
            if not math.isfinite(gain) or gain <= 1e-12:
                continue
            choices.append(
                (
                    gain / cost,
                    cost,
                    float(node.tie_breaker),
                    node.state_key,
                    node,
                    utility,
                    gain,
                )
            )
        if not choices:
            break
        _, cost, _, _, node, utility, gain = max(choices, key=lambda item: (item[0], item[6], item[2], item[3]))
        if cost > remaining:
            break
        node.current_k += cost
        node.opened = True
        remaining -= cost
        task_allocated[str(node.task_uid)] = task_allocated.get(str(node.task_uid), 0) + cost
        allocations[node.state_key] = AgentDPSAllocation(
            anchor_id=str(node.anchor_id),
            trajectory_id=str(node.trajectory_id),
            continuation_count=int(node.current_k),
            predicted_mean=float(node.probability),
            predicted_uncertainty=float(math.sqrt(max(0.0, node.uncertainty))),
            external_cost=float(node.current_k),
            score=float(utility),
            cumulative_utility=float(utility),
            marginal_utility=float(gain),
            marginal_cost=float(cost),
            marginal_efficiency=float(gain / cost),
            marginal_score=float(gain),
            marginal_score_efficiency=float(gain / cost),
            state_group_key=str(node.state_key),
            factual_count=max(len(node.factual_outcomes), int(node.existing_factual_count)),
            physical_factual_count=(0 if root_mode else 1),
        )
    allocation_list = sorted(allocations.values(), key=lambda item: item.anchor_id)
    for record in records:
        allocation = allocations.get(str(record["state_group_key"]))
        if allocation is None:
            continue
        record["selected_count"] = 1
        record["allocated_k"] = int(allocation.continuation_count)
        record["selected_anchor_ids"] = [allocation.anchor_id]
        record["selected_k"] = [int(allocation.continuation_count)]
    metrics = {
        "agent_dps/allocator/budget": float(budget),
        "agent_dps/allocator/used_budget": float(sum(item.continuation_count for item in allocation_list)),
        "agent_dps/allocator/budget_utilization": float(sum(item.continuation_count for item in allocation_list) / max(1, int(budget))),
        "agent_dps/allocator/n_states": float(len(nodes)),
        "agent_dps/allocator/candidate_top_m": float(requested_top_m or len(nodes)),
        "agent_dps/allocator/prefiltered_states": float(max(0, len(nodes) - len(allocation_nodes))),
        "agent_dps/allocator/prefilter_keep_fraction": float(len(allocation_nodes) / max(1, len(nodes))),
        "agent_dps/allocator/n_selected_states": float(len(allocation_list)),
        "agent_dps/allocator/mean_k": float(sum(item.continuation_count for item in allocation_list) / max(1, len(allocation_list))),
        "agent_dps/allocator/max_k": float(max((item.continuation_count for item in allocation_list), default=0)),
        "agent_dps/allocator/min_k": float(min((item.continuation_count for item in allocation_list), default=0)),
        "agent_dps/allocator/atomic_opening_size": float(opening_size),
        "agent_dps/allocator/min_new_branch": float(min_new),
        "agent_dps/allocator/root_baseline": float(baseline),
        "agent_dps/allocator/root_floor_tasks": float(
            sum(1 for item in allocation_list if item.continuation_count >= baseline)
            if root_mode and baseline > 0 else 0
        ),
        "agent_dps/allocator/underdense_selected_states": float(sum(0 < item.continuation_count < min_new for item in allocation_list)),
        "agent_dps/allocator/unspent_budget": float(remaining),
        "agent_dps/allocator/mean_marginal_gain": float(sum(item.marginal_utility for item in allocation_list) / max(1, len(allocation_list))),
        "agent_dps/allocator/rejected_homogeneous": float(sum(1 for node in nodes if reject_homogeneous and node.homogeneous)),
        "agent_dps/allocator/task_min_coverage_enabled": 0.0,
        "agent_dps/allocator/task_coverage_fallback": 0.0,
        "agent_dps/allocator/task_target_budget": float(target_task_budget),
        "agent_dps/allocator/task_budget_cap": 0.0,
        "agent_dps/allocator/task_diminishing_exponent": 0.0,
        "agent_dps/allocator/task_covered_count": float(sum(value > 0 for value in task_allocated.values())),
        "agent_dps/allocator/task_max_allocated": float(max(task_allocated.values(), default=0)),
        "agent_dps/allocator/mixed_objective_enabled": 1.0,
        "agent_dps/allocator/bayesian_transition_objective_enabled": float(
            contrast_objective and all(node.uses_transition_contrast for node in nodes)
        ),
        "agent_dps/allocator/bayesian_terminal_objective_enabled": float(
            contrast_objective and all(node.uses_terminal_contrast for node in nodes)
        ),
        "agent_dps/allocator/posterior_optimal_knapsack": float(
            contrast_objective
        ),
        "agent_dps/allocator/predicted_created_contrast_sum": float(
            sum(item.cumulative_utility for item in allocation_list)
            if contrast_objective else 0.0
        ),
        "agent_dps/allocator/predicted_created_contrast_rate": float(
            sum(item.cumulative_utility for item in allocation_list)
            / max(1, len(allocation_list))
            if contrast_objective else 0.0
        ),
        "agent_dps/allocator/legacy_objective_terms_ignored": float(
            bool(task_min_coverage)
            or task_max_budget_factor is not None
            or float(task_diminishing_exponent) != 0.0
        ),
        "agent_dps/allocator/acquisition_prefilter_enabled": float(
            not contrast_objective
            and not root_mode and requested_top_m < len(nodes)
        ),
    }
    for allocation in allocation_list:
        metrics[f"agent_dps/allocator/k_{allocation.continuation_count}_states"] = metrics.get(f"agent_dps/allocator/k_{allocation.continuation_count}_states", 0.0) + 1.0
    return allocation_list, metrics, records


__all__ = [
    "AgentDPSAllocation",
    "AgentDPSCurriculumCandidate",
    "AgentDPSNode",
    "AgentDPSStateMemory",
    "StateStats",
    "TransitionContrastPosterior",
    "TerminalContrastPosterior",
    "action_key",
    "allocate_agent_dps",
    "beta_event_utility",
    "beta_mixed_utility",
    "branch_contrast_counts",
    "conditional_mixed_utility",
    "expected_beta_mixed_gain",
    "expected_mixed_gain",
    "mixed_utility",
    "root_curriculum_key",
    "select_agent_dps_curriculum",
    "state_key",
]
