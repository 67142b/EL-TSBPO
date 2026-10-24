"""Collect successful WebShop trajectories with an OpenAI teacher model.

The exported JSON has the same schema as ``data.json``: each environment step
is one ``{"messages": [user_message, assistant_message]}`` item. Only steps
from successful trajectories are exported.

Successful trajectories are first flushed to an append-only JSONL journal.
The flat JSON output is then updated incrementally and can always be rebuilt
from that journal after an interruption. Failed-attempt reflections are private
control context and are never written to either file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
current_pythonpath = os.environ.get("PYTHONPATH", "")
if str(PROJECT_DIR) not in current_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_DIR), current_pythonpath) if part
    )

from openai import (  # noqa: E402
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
FULL_RESPONSE_RE = re.compile(
    r"^\s*<think>\s*(.+?)\s*</think>\s*"
    r"<action>\s*(.+?)\s*</action>\s*$",
    re.IGNORECASE | re.DOTALL,
)
SEARCH_RE = re.compile(r"^search\[(.+)]$", re.IGNORECASE | re.DOTALL)
CLICK_RE = re.compile(r"^click\[(.+)]$", re.IGNORECASE | re.DOTALL)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
PRIVATE_MARKERS = (
    "private_retry_context",
    "private_correction_context",
    "private correction context",
    "private context",
    "prior answer",
    "earlier answer",
    "previous response",
    "rejected because",
    "previous attempt",
    "previous failed attempt",
    "failed attempt",
    "retry",
    "reflection",
    "reflection advice",
    "feedback",
    "correction",
    "on this retry",
)


class FatalAPIError(RuntimeError):
    """An API error that will not recover by waiting and retrying."""


class TransientAPIError(RuntimeError):
    """A malformed empty response that may succeed on retry."""


@dataclass
class SessionJob:
    session_id: int
    query: str
    attempt: int = 1
    reflection: str = ""


@dataclass
class AttemptOutcome:
    job: SessionJob
    success: bool
    steps: list[dict[str, Any]]
    reason: str
    task_score: float = 0.0


class AtomicTrajectorySaver:
    """Append successful trajectories incrementally and materialize flat JSON."""

    def __init__(self, output_path: Path):
        self.output_path = output_path.resolve()
        self.journal_path = Path(f"{self.output_path}.journal.jsonl")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._records = self._load_journal()
        self._session_ids = {
            int(record["session_id"])
            for record in self._records
        }

        if self.output_path.exists() and not self.journal_path.exists():
            try:
                existing = json.loads(self.output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read existing output: {self.output_path}") from exc
            if existing:
                raise ValueError(
                    "Existing output has no matching journal and cannot be resumed safely. "
                    "Choose a new --output path or restore its .journal.jsonl file."
                )

        if self._records:
            self._materialize_sync()
        elif not self.output_path.exists():
            self._write_json_atomic([])

    @property
    def success_count(self) -> int:
        return len(self._records)

    @property
    def step_count(self) -> int:
        return sum(len(record["steps"]) for record in self._records)

    @property
    def successful_session_ids(self) -> set[int]:
        return set(self._session_ids)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def _load_journal(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        records = []
        with self.journal_path.open("r", encoding="utf-8") as journal:
            for line_number, line in enumerate(journal, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._validate_record(record)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                    logging.warning(
                        "Ignoring incomplete journal line %s in %s: %s",
                        line_number,
                        self.journal_path,
                        exc,
                    )
                    continue
                records.append(record)
        return records

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> None:
        if not isinstance(record.get("session_id"), int):
            raise ValueError("session_id must be an integer")
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("successful trajectory must contain steps")
        for step in steps:
            messages = step.get("messages") if isinstance(step, dict) else None
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError("each step must contain one user/assistant message pair")
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError("message roles must be user then assistant")

    def _append_journal_sync(self, record: dict[str, Any]) -> None:
        needs_separator = False
        if self.journal_path.exists() and self.journal_path.stat().st_size:
            with self.journal_path.open("rb") as journal:
                journal.seek(-1, os.SEEK_END)
                needs_separator = journal.read(1) != b"\n"
        with self.journal_path.open("a", encoding="utf-8") as journal:
            if needs_separator:
                journal.write("\n")
            journal.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            journal.write("\n")
            journal.flush()

    def _write_json_atomic(self, data: list[dict[str, Any]]) -> None:
        temporary = self.output_path.with_name(
            f".{self.output_path.name}.tmp.{os.getpid()}"
        )
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
            os.replace(temporary, self.output_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _materialize_sync(self) -> None:
        flat_steps = [
            step
            for record in self._records
            for step in record["steps"]
        ]
        self._write_json_atomic(flat_steps)

    def _append_steps_sync(
        self,
        steps: list[dict[str, Any]],
        *,
        has_existing_steps: bool,
    ) -> None:
        """Append JSON-array items in place; the journal remains canonical."""
        serialized = json.dumps(steps, ensure_ascii=False, indent=2)
        inner = serialized[1:-1].strip()
        indented = "\n".join(f"  {line}" for line in inner.splitlines())
        separator = ",\n" if has_existing_steps else "\n"
        payload = f"{separator}{indented}\n]\n".encode("utf-8")

        with self.output_path.open("r+b") as output:
            output.seek(0, os.SEEK_END)
            position = output.tell()
            byte = b""
            while position > 0:
                position -= 1
                output.seek(position)
                byte = output.read(1)
                if not byte.isspace():
                    break
            if byte != b"]":
                raise ValueError("flat output is incomplete and must be rebuilt")
            output.seek(position)
            output.write(payload)
            output.truncate()
            output.flush()

    async def save_success(
        self,
        job: SessionJob,
        steps: list[dict[str, Any]],
        task_score: float,
    ) -> None:
        """Persist one success without content or session-level deduplication."""
        async with self._lock:
            record = {
                "trajectory_id": (
                    f"webshop-session-{job.session_id}-sample-{uuid.uuid4().hex}"
                ),
                "session_id": job.session_id,
                "query": job.query,
                "attempt": job.attempt,
                "task_score": float(task_score),
                "steps": steps,
            }
            self._validate_record(record)
            has_existing_steps = self.step_count > 0
            self._append_journal_sync(record)
            self._records.append(record)
            self._session_ids.add(job.session_id)
            try:
                self._append_steps_sync(
                    steps,
                    has_existing_steps=has_existing_steps,
                )
            except (OSError, ValueError):
                self._materialize_sync()


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        status = int(getattr(exc, "status_code", 0) or 0)
        return status in {408, 409, 429} or status >= 500
    return isinstance(exc, TransientAPIError)


class OpenAITeacher:
    def __init__(self, args: argparse.Namespace):
        self.model = args.model
        self.temperature = args.temperature
        self.reasoning_effort = args.reasoning_effort
        self.max_tokens = args.max_tokens
        self.reflection_max_tokens = args.reflection_max_tokens
        self.api_max_retries = args.api_max_retries
        self.retry_initial_delay = args.retry_initial_delay
        self.retry_max_delay = args.retry_max_delay
        self.semaphore = asyncio.Semaphore(args.api_concurrency)
        self.rng = random.Random(args.seed + 991)
        self.client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.api_base,
            timeout=args.request_timeout,
            max_retries=0,
        )

    async def close(self) -> None:
        await self.client.close()

    async def _request(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        failure_count = 0
        while True:
            try:
                async with self.semaphore:
                    request: dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "max_completion_tokens": max_tokens,
                    }
                    if temperature is not None:
                        request["temperature"] = temperature
                    if self.reasoning_effort:
                        request["reasoning_effort"] = self.reasoning_effort
                    response = await self.client.chat.completions.create(
                        **request,
                    )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise TransientAPIError("API returned empty content")
                return content.strip()
            except Exception as exc:
                if not _is_retryable_api_error(exc):
                    raise FatalAPIError(f"Non-retryable OpenAI API error: {exc}") from exc
                failure_count += 1
                if self.api_max_retries > 0 and failure_count > self.api_max_retries:
                    raise RuntimeError(
                        f"OpenAI API remained unavailable after {failure_count} failures"
                    ) from exc
                delay = min(
                    self.retry_initial_delay * (2 ** min(failure_count - 1, 10)),
                    self.retry_max_delay,
                )
                delay *= self.rng.uniform(0.8, 1.2)
                logging.warning(
                    "Transient API failure %s; retrying in %.1fs: %s",
                    failure_count,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    async def action_response(
        self,
        prompt: str,
        *,
        reflection: str = "",
        correction: str = "",
    ) -> str:
        messages: list[dict[str, str]] = []
        private_parts = []
        if reflection:
            private_parts.append(
                "<private_retry_context>\n"
                f"{reflection}\n"
                "</private_retry_context>"
            )
        if correction:
            private_parts.append(
                "<private_correction_context>\n"
                f"{correction}\n"
                "</private_correction_context>"
            )
        if private_parts:
            messages.append({
                "role": "system",
                "content": (
                    "Use the private context silently to improve the next WebShop action. "
                    "Never mention a retry, failed attempt, reflection, correction, or any "
                    "private context. Answer the user's original prompt directly and return "
                    "exactly one <think>...</think> block followed by exactly one "
                    "<action>...</action> block.\n\n" + "\n\n".join(private_parts)
                ),
            })
        messages.append({"role": "user", "content": prompt})
        return await self._request(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    async def reflect(self, job: SessionJob, outcome: AttemptOutcome) -> str:
        transcript = compact_failed_trajectory(outcome.steps)
        messages = [
            {
                "role": "system",
                "content": (
                    "Diagnose a failed WebShop agent attempt. Produce concise private advice "
                    "for a fresh retry of the same task. Identify search/query, product, option, "
                    "navigation, or purchase mistakes. Do not emit action tags."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Session: {job.session_id}\nFailure: {outcome.reason}\n"
                    f"Trajectory:\n{transcript}"
                ),
            },
        ]
        return await self._request(
            messages,
            max_tokens=self.reflection_max_tokens,
            temperature=self.temperature,
        )


class MockTeacher:
    """Offline valid-action generator used only for pipeline smoke tests."""

    async def close(self) -> None:
        return None

    async def action_response(
        self,
        prompt: str,
        *,
        reflection: str = "",
        correction: str = "",
    ) -> str:
        actions = re.findall(r"'(search\[[^']+]|click\[[^']+])'", prompt)
        if not actions:
            action = "search[test product]"
        elif actions[0].lower().startswith("search["):
            task_match = re.search(r"Your task is to:\s*(.*?)\.", prompt, re.DOTALL)
            query = task_match.group(1).strip() if task_match else "test product"
            action = f"search[{query[:120]}]"
        else:
            action = next(
                (candidate for candidate in actions if candidate.lower() == "click[buy now]"),
                actions[0],
            )
        return f"<think>Select a currently admissible action.</think>\n<action>{action}</action>"

    async def reflect(self, job: SessionJob, outcome: AttemptOutcome) -> str:
        return "Try a different admissible product-navigation path."


def extract_action(response: str) -> str | None:
    matches = ACTION_RE.findall(response)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def validate_action_response(
    response: str,
    available_actions: dict[str, Any],
) -> tuple[bool, str, str | None]:
    lowered = response.lower()
    if any(marker in lowered for marker in PRIVATE_MARKERS):
        return False, "response leaked private retry context", None
    if CHINESE_RE.search(response):
        return False, "response contains Chinese text", None
    if not FULL_RESPONSE_RE.fullmatch(response):
        return False, "response must contain only one think block and one action block", None
    action = extract_action(response)
    if not action:
        return False, "response must contain exactly one action block", None

    search_match = SEARCH_RE.fullmatch(action)
    if search_match:
        query = search_match.group(1).strip()
        if available_actions.get("has_search_bar") and query and query != "<your query>":
            return True, "", action
        return False, "search action is unavailable or has an empty query", action

    click_match = CLICK_RE.fullmatch(action)
    if click_match:
        target = click_match.group(1).strip().lower()
        clickables = {
            str(value).strip().lower()
            for value in available_actions.get("clickables", [])
        }
        if target in clickables:
            return True, "", action
        return False, f"click target is not currently admissible: {target}", action
    return False, "action must be search[...] or click[...]", action


async def generate_valid_response(
    teacher: OpenAITeacher | MockTeacher,
    prompt: str,
    available_actions: dict[str, Any],
    reflection: str,
    max_repairs: int,
) -> tuple[str, str]:
    correction = ""
    last_reason = ""
    for repair_index in range(max_repairs + 1):
        response = await teacher.action_response(
            prompt,
            reflection=reflection,
            correction=correction,
        )
        valid, reason, action = validate_action_response(response, available_actions)
        if valid and action is not None:
            return response, action
        last_reason = reason
        logging.info("Rejecting malformed teacher response: %s", reason)
        correction = (
            f"The prior answer was rejected because: {reason}. Select a legal action from "
            "the user's current admissible-action list and obey the requested tag format."
        )
    raise RuntimeError(
        f"Teacher failed to produce a valid action after {max_repairs + 1} responses: "
        f"{last_reason}"
    )


def compact_failed_trajectory(steps: Sequence[dict[str, Any]]) -> str:
    lines = []
    for step_number, step in enumerate(steps[-12:], start=max(1, len(steps) - 11)):
        user_text = step["messages"][0]["content"]
        assistant_text = step["messages"][1]["content"]
        observation = user_text
        marker = "current observation is:"
        if marker in user_text:
            observation = user_text.split(marker, 1)[1]
        observation = observation.split("Your admissible actions", 1)[0].strip()
        action = extract_action(assistant_text) or "<invalid>"
        lines.append(
            f"Step {step_number}: observation={observation[:800]!r}; action={action!r}"
        )
    return "\n".join(lines) if lines else "No valid environment steps were executed."


def build_webshop_manager(args: argparse.Namespace):
    import ray
    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
        )

    webshop_root = (
        PROJECT_DIR
        / "agent_system/environments/env_package/webshop/webshop"
    )
    suffix = "_1000" if args.use_small else ""
    env_kwargs = {
        "observation_mode": "text",
        "num_products": 1000 if args.use_small else None,
        "human_goals": args.human_goals,
        # A session ID must identify the same goal after moving to another slot.
        "_shared_goal_order": True,
        "file_path": str(webshop_root / f"data/items_shuffle{suffix}.json"),
        "attr_path": str(webshop_root / f"data/items_ins_v2{suffix}.json"),
    }
    envs = build_webshop_envs(
        seed=args.seed,
        env_num=args.concurrency,
        group_n=1,
        is_train=args.split == "train",
        env_kwargs=env_kwargs,
        resources_per_worker={"num_cpus": args.cpus_per_env, "num_gpus": 0},
    )
    config = OmegaConf.create({
        "env": {
            "env_name": "webshop/WebAgentTextEnv",
            "history_length": args.history_length,
        }
    })
    return WebshopEnvironmentManager(envs, webshop_projection, config)


async def run_wave(
    manager,
    jobs: list[SessionJob],
    teacher: OpenAITeacher | MockTeacher,
    saver: AtomicTrajectorySaver,
    coverage: QueryCoverage,
    args: argparse.Namespace,
) -> list[AttemptOutcome]:
    observations, infos = manager.reset({
        "session_ids": [job.session_id for job in jobs],
    })
    trajectories: list[list[dict[str, Any]]] = [[] for _ in jobs]
    active = set(range(len(jobs)))
    outcomes: dict[int, AttemptOutcome] = {}

    for step_number in range(1, args.max_steps + 1):
        if not active:
            break
        active_indices = sorted(active)
        requests = [
            generate_valid_response(
                teacher,
                observations["text"][index],
                infos[index]["available_actions"],
                jobs[index].reflection,
                args.max_response_repairs,
            )
            for index in active_indices
        ]
        results = await asyncio.gather(*requests, return_exceptions=True)

        step_indices = []
        step_responses = []
        for index, result in zip(active_indices, results):
            if isinstance(result, FatalAPIError):
                raise result
            if isinstance(result, Exception):
                outcomes[index] = AttemptOutcome(
                    job=jobs[index],
                    success=False,
                    steps=trajectories[index],
                    reason=f"teacher response failure: {result}",
                )
                active.remove(index)
                continue
            response, _action = result
            trajectories[index].append({
                "messages": [
                    {"role": "user", "content": observations["text"][index]},
                    {"role": "assistant", "content": response},
                ]
            })
            step_indices.append(index)
            step_responses.append(response)

        if not step_indices:
            continue

        observations, _rewards, dones, infos = manager.step_active(
            step_indices,
            step_responses,
        )
        for index in step_indices:
            if not bool(dones[index]):
                continue
            info = infos[index]
            won = bool(info.get("won", False))
            score = float(info.get("task_score", 0.0))
            reason = "success" if won else f"terminal purchase score={score:.4f}"
            outcome = AttemptOutcome(
                job=jobs[index],
                success=won,
                steps=trajectories[index],
                reason=reason,
                task_score=score,
            )
            outcomes[index] = outcome
            active.remove(index)
            if won:
                await saver.save_success(jobs[index], trajectories[index], score)
                coverage.record_success(jobs[index].query)
                logging.info(
                    "Saved success %s: session=%s query=%r attempt=%s "
                    "trajectory_steps=%s total_pairs=%s/%s",
                    saver.success_count,
                    jobs[index].session_id,
                    jobs[index].query,
                    jobs[index].attempt,
                    len(trajectories[index]),
                    saver.step_count,
                    args.target_pairs,
                )

        logging.info(
            "Wave step %s/%s: active=%s pairs=%s/%s query_coverage=%s/%s",
            step_number,
            args.max_steps,
            len(active),
            saver.step_count,
            args.target_pairs,
            coverage.satisfied_count,
            len(coverage.required_queries),
        )
        if collection_complete(saver, coverage, args.target_pairs):
            break

    for index in active:
        outcomes[index] = AttemptOutcome(
            job=jobs[index],
            success=False,
            steps=trajectories[index],
            reason=(
                "collection target reached"
                if collection_complete(saver, coverage, args.target_pairs)
                else f"step limit reached ({args.max_steps})"
            ),
        )
    return [outcomes[index] for index in range(len(jobs))]


@dataclass
class QueryCoverage:
    required_queries: set[str]
    min_successes: int
    successes: Counter[str]

    @property
    def missing_queries(self) -> set[str]:
        return {
            query
            for query in self.required_queries
            if self.successes[query] < self.min_successes
        }

    @property
    def satisfied_count(self) -> int:
        return len(self.required_queries) - len(self.missing_queries)

    def record_success(self, query: str) -> None:
        self.successes[query] += 1


class QuerySessionScheduler:
    """Prefer under-covered queries, then sample the remaining sessions randomly."""

    def __init__(
        self,
        session_ids: Iterable[int],
        query_by_session: dict[int, str],
        max_samples_per_session: int,
        seed: int,
    ) -> None:
        candidates = [int(session_id) for session_id in session_ids]
        if not candidates:
            raise ValueError("QuerySessionScheduler requires at least one session")
        if max_samples_per_session <= 0:
            raise ValueError("max_samples_per_session must be positive")
        self._rng = random.Random(seed)
        self._candidates = candidates
        self._max_samples_per_session = max_samples_per_session
        self._sample_counts: Counter[int] = Counter()

        sessions_by_query: dict[str, list[int]] = defaultdict(list)
        for session_id in candidates:
            sessions_by_query[query_by_session[session_id]].append(session_id)
        self._sessions_by_query = dict(sessions_by_query)
        self._by_query: dict[str, deque[int]] = {
            query: deque() for query in self._sessions_by_query
        }
        self._general: deque[int] = deque()
        query_order = sorted(self._sessions_by_query)
        self._rng.shuffle(query_order)
        self._query_order = deque(query_order)
        self._query_by_session = query_by_session

    def _pop_for_query(self, query: str) -> int | None:
        sessions = self._by_query[query]
        while True:
            while sessions:
                session_id = sessions.popleft()
                if self._sample_counts[session_id] < self._max_samples_per_session:
                    self._sample_counts[session_id] += 1
                    return session_id
            refill = list(self._sessions_by_query[query])
            self._rng.shuffle(refill)
            sessions.extend(refill)
            if not any(
                self._sample_counts[session_id] < self._max_samples_per_session
                for session_id in refill
            ):
                return None

    def _pop_general(self) -> int | None:
        while True:
            while self._general:
                session_id = self._general.popleft()
                if self._sample_counts[session_id] < self._max_samples_per_session:
                    self._sample_counts[session_id] += 1
                    return session_id
            refill = list(self._candidates)
            self._rng.shuffle(refill)
            self._general.extend(refill)
            if not any(
                self._sample_counts[session_id] < self._max_samples_per_session
                for session_id in refill
            ):
                return None

    def next_job(self, coverage: QueryCoverage) -> SessionJob | None:
        for _ in range(len(self._query_order)):
            query = self._query_order.popleft()
            self._query_order.append(query)
            if query not in coverage.missing_queries:
                continue
            session_id = self._pop_for_query(query)
            if session_id is not None:
                return SessionJob(session_id=session_id, query=query)

        session_id = self._pop_general()
        if session_id is None:
            return None
        return SessionJob(
            session_id=session_id,
            query=self._query_by_session[session_id],
        )


def collection_complete(
    saver: AtomicTrajectorySaver,
    coverage: QueryCoverage,
    target_pairs: int,
) -> bool:
    return saver.step_count >= target_pairs and not coverage.missing_queries


async def collect(args: argparse.Namespace) -> None:
    saver = AtomicTrajectorySaver(Path(args.output))
    logging.info(
        "Resume state: successful trajectories=%s, exported state-action pairs=%s",
        saver.success_count,
        saver.step_count,
    )

    if args.mock:
        teacher: OpenAITeacher | MockTeacher = MockTeacher()
    else:
        teacher = OpenAITeacher(args)

    manager = None
    try:
        manager = build_webshop_manager(args)
        goal_queries = list(manager.envs.goal_queries)
        val_session_count = min(500, len(goal_queries))
        val_queries = {
            query for query in goal_queries[:val_session_count] if query
        }
        eligible_session_ids = [
            int(session_id)
            for session_id in manager.envs.goal_idxs
            if goal_queries[int(session_id)]
        ]
        random.Random(args.seed).shuffle(eligible_session_ids)
        if args.max_unique_tasks > 0:
            eligible_session_ids = eligible_session_ids[:args.max_unique_tasks]
        eligible_session_set = set(eligible_session_ids)
        query_by_session = {
            session_id: goal_queries[session_id]
            for session_id in eligible_session_ids
        }
        invalid_saved_sessions = saver.successful_session_ids - eligible_session_set
        if invalid_saved_sessions:
            raise ValueError(
                "The existing journal contains validation sessions or sessions outside "
                "the configured train task pool: "
                f"{sorted(invalid_saved_sessions)[:10]}. Use a new --output path."
            )

        required_queries = {
            query for query in query_by_session.values() if query not in val_queries
        }
        if not required_queries and args.min_successes_per_query > 0:
            raise ValueError("No train-only WebShop queries remain after validation exclusion")
        success_counts: Counter[str] = Counter()
        for record in saver.records:
            session_id = int(record["session_id"])
            current_query = query_by_session[session_id]
            stored_query = record.get("query")
            if stored_query is not None:
                normalized_stored_query = " ".join(
                    str(stored_query).lower().split()
                )
                if normalized_stored_query != current_query:
                    raise ValueError(
                        "The existing journal was created with a different WebShop "
                        f"query mapping for session {session_id}: "
                        f"stored={stored_query!r}, current={current_query!r}. "
                        "Use the original dataset/configuration or choose a new output."
                    )
            success_counts[current_query] += 1
        coverage = QueryCoverage(
            required_queries=required_queries,
            min_successes=args.min_successes_per_query,
            successes=success_counts,
        )
        logging.info(
            "Task pool: val_sessions=%s val_queries=%s train_sessions=%s "
            "coverage_queries_excluding_val=%s minimum_successes_per_query=%s",
            val_session_count,
            len(val_queries),
            len(eligible_session_ids),
            len(required_queries),
            args.min_successes_per_query,
        )
        if collection_complete(saver, coverage, args.target_pairs):
            logging.info("Pair target and query coverage are already satisfied.")
            return

        scheduler = QuerySessionScheduler(
            eligible_session_ids,
            query_by_session,
            args.max_attempts_per_task,
            args.seed,
        )
        retry_queue: deque[SessionJob] = deque()
        attempted_unique: set[int] = set()

        while not collection_complete(saver, coverage, args.target_pairs):
            jobs: list[SessionJob] = []
            while retry_queue and len(jobs) < args.concurrency:
                jobs.append(retry_queue.popleft())
            while len(jobs) < args.concurrency:
                job = scheduler.next_job(coverage)
                if job is None:
                    break
                attempted_unique.add(job.session_id)
                jobs.append(job)
            if len(jobs) < args.concurrency:
                missing_preview = sorted(coverage.missing_queries)[:10]
                raise RuntimeError(
                    "Collection exhausted session sampling attempts before satisfying "
                    f"the target: pairs={saver.step_count}/{args.target_pairs}, "
                    f"missing_queries={missing_preview}"
                )

            logging.info(
                "Starting wave: sessions=%s attempts=%s",
                [job.session_id for job in jobs],
                [job.attempt for job in jobs],
            )
            outcomes = await run_wave(manager, jobs, teacher, saver, coverage, args)
            if collection_complete(saver, coverage, args.target_pairs):
                break

            retryable = [
                outcome
                for outcome in outcomes
                if not outcome.success
                and outcome.reason != "collection target reached"
                and outcome.job.attempt < args.max_attempts_per_task
            ]
            if args.reflection and retryable:
                reflections = await asyncio.gather(
                    *(teacher.reflect(outcome.job, outcome) for outcome in retryable),
                    return_exceptions=True,
                )
            else:
                reflections = [""] * len(retryable)

            for outcome, reflection in zip(retryable, reflections):
                if isinstance(reflection, FatalAPIError):
                    raise reflection
                if isinstance(reflection, Exception):
                    logging.warning(
                        "Reflection failed for session %s: %s",
                        outcome.job.session_id,
                        reflection,
                    )
                    reflection = ""
                retry_queue.append(SessionJob(
                    session_id=outcome.job.session_id,
                    query=outcome.job.query,
                    attempt=outcome.job.attempt + 1,
                    reflection=str(reflection),
                ))

            logging.info(
                "Progress: trajectories=%s pairs=%s/%s query_coverage=%s/%s "
                "attempted_unique=%s pending_retries=%s",
                saver.success_count,
                saver.step_count,
                args.target_pairs,
                coverage.satisfied_count,
                len(coverage.required_queries),
                len(attempted_unique),
                len(retry_queue),
            )
    finally:
        if manager is not None:
            manager.close()
        await teacher.close()
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass

    logging.info(
        "Collection finished: trajectories=%s, state-action pairs=%s, "
        "query_coverage=%s/%s, output=%s, journal=%s",
        saver.success_count,
        saver.step_count,
        coverage.satisfied_count,
        len(coverage.required_queries),
        saver.output_path,
        saver.journal_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect clean successful WebShop trajectories with an OpenAI teacher",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--api_base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR.parent / "llamafactory/data/webshop_success_data.json"),
    )
    parser.add_argument(
        "--target_pairs",
        type=int,
        default=10_000,
        help="Minimum exported state-action pairs; the final wave may exceed it",
    )
    parser.add_argument(
        "--min_successes_per_query",
        type=int,
        default=1,
        help="Minimum successful trajectories for every train-only query",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--api_concurrency", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_attempts_per_task", type=int, default=3)
    parser.add_argument("--max_response_repairs", type=int, default=2)
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; omit for compatibility with reasoning models",
    )
    parser.add_argument(
        "--reasoning_effort",
        default="high",
        help="Optional OpenAI reasoning effort, for example low, medium, or high",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--reflection_max_tokens", type=int, default=512)
    parser.add_argument("--reflection", type=int, choices=[0, 1], default=1)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument(
        "--api_max_retries",
        type=int,
        default=0,
        help="Transient request retries; 0 means retry indefinitely until network recovery",
    )
    parser.add_argument("--retry_initial_delay", type=float, default=1.0)
    parser.add_argument("--retry_max_delay", type=float, default=60.0)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--use_small", type=int, choices=[0, 1], default=1)
    parser.add_argument("--human_goals", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cpus_per_env", type=float, default=1.0)
    parser.add_argument("--max_unique_tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use an offline valid-action stub to smoke-test the pipeline",
    )
    args = parser.parse_args()

    for name in (
        "target_pairs",
        "concurrency",
        "api_concurrency",
        "max_steps",
        "max_attempts_per_task",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.min_successes_per_query < 0:
        parser.error("--min_successes_per_query must be non-negative")
    if not args.mock and not args.api_key:
        parser.error("set OPENAI_API_KEY or pass --api_key")
    if not args.mock and not args.model:
        parser.error("set OPENAI_MODEL or pass --model")
    args.use_small = bool(args.use_small)
    args.human_goals = bool(args.human_goals)
    args.reflection = bool(args.reflection)
    return args


def configure_logging() -> Path:
    log_dir = PROJECT_DIR / "logs/webshop"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"openai_collect_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    logging.info(
        "Configuration: model=%s target_pairs=%s min_successes_per_query=%s "
        "concurrency=%s output=%s mock=%s",
        args.model or "<placeholder>",
        args.target_pairs,
        args.min_successes_per_query,
        args.concurrency,
        args.output,
        args.mock,
    )
    logging.info("Log file: %s", log_path)
    try:
        asyncio.run(collect(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted; all previously journaled successes remain recoverable.")


if __name__ == "__main__":
    main()
