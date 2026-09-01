"""Qualified exact-action prefixes for legal live reverse curricula."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autodancer.curriculum import EpisodeResetSpec
from autodancer.training.demonstrations import validate_demonstration_bank

TRACE_PREFIX_RECURRENT_MODES = ("fresh", "warm")


def parse_trace_tail_window(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, unique set of positive learner-tail lengths."""

    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("trace-prefix tail window must contain integers") from error
    if not values:
        raise ValueError("trace-prefix tail window must not be empty")
    if any(item <= 0 for item in values):
        raise ValueError("trace-prefix tail window values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("trace-prefix tail window must not contain duplicates")
    return tuple(sorted(values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QualifiedActionTrace:
    trace_id: str
    seed: int
    reset_spec: EpisodeResetSpec
    actions: tuple[int, ...]
    turn_digests: tuple[str, ...]
    source_policy_version: int
    source_global_step: int


@dataclass(frozen=True, slots=True)
class QualifiedTracePrefixBank:
    """One deterministically selected, freshly replay-qualified trace per seed."""

    bank_path: Path
    qualification_path: Path
    bank_sha256: str
    qualification_sha256: str
    action_contract: str
    tail_actions: int
    tail_action_window: tuple[int, ...]
    recurrent_state_mode: str
    traces: tuple[QualifiedActionTrace, ...]

    @classmethod
    def load(
        cls,
        bank_path: str | Path,
        qualification_path: str | Path,
        *,
        tail_actions: int,
        tail_action_window: tuple[int, ...] = (),
        recurrent_state_mode: str = "warm",
    ) -> QualifiedTracePrefixBank:
        bank_source = Path(bank_path).resolve()
        qualification_source = Path(qualification_path).resolve()
        if tail_actions <= 0:
            raise ValueError("trace-prefix tail_actions must be positive")
        selected_window = tuple(int(value) for value in tail_action_window) or (
            int(tail_actions),
        )
        if any(value <= 0 for value in selected_window):
            raise ValueError("trace-prefix tail_action_window values must be positive")
        if len(set(selected_window)) != len(selected_window):
            raise ValueError("trace-prefix tail_action_window must not contain duplicates")
        selected_window = tuple(sorted(selected_window))
        if int(tail_actions) not in selected_window:
            raise ValueError("trace-prefix tail_actions must belong to tail_action_window")
        if recurrent_state_mode not in TRACE_PREFIX_RECURRENT_MODES:
            raise ValueError(
                "trace-prefix recurrent_state_mode must be one of "
                + ", ".join(TRACE_PREFIX_RECURRENT_MODES)
            )
        bank = json.loads(bank_source.read_text(encoding="utf-8"))
        validate_demonstration_bank(bank)
        report = json.loads(qualification_source.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise ValueError("trace-prefix qualification report must use schema 1")
        if report.get("kind") != "qualified-live-action-traces-report-v1":
            raise ValueError("trace-prefix qualification report has the wrong kind")
        if report.get("valid") is not True or int(report.get("worker_restarts", -1)) != 0:
            raise ValueError("trace-prefix qualification report is not valid and restart-free")
        if report.get("bank_sha256") != bank.get("bank_sha256"):
            raise ValueError("trace-prefix bank and qualification report do not match")
        source_contracts = {
            str(source.get("action_contract"))
            for source in bank.get("sources", [])
            if isinstance(source, Mapping) and source.get("successful_trace_count", 0)
        }
        if len(source_contracts) != 1 or "None" in source_contracts:
            raise ValueError("trace-prefix sources must bind exactly one action contract")
        action_contract = source_contracts.pop()
        raw_results = report.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("trace-prefix qualification results must be a list")
        results = {
            str(result.get("trace_id")): result
            for result in raw_results
            if isinstance(result, Mapping) and result.get("valid") is True
        }
        candidates: dict[int, list[QualifiedActionTrace]] = {}
        for raw_trace in bank["traces"]:
            trace_id = str(raw_trace["trace_id"])
            result = results.get(trace_id)
            if result is None:
                raise ValueError(f"trace {trace_id} has no valid live replay qualification")
            source_actions = tuple(int(action) for action in raw_trace["action_sequence"])
            raw_qualified_actions = result.get("actual", {}).get(
                "qualified_action_sequence"
            )
            if not isinstance(raw_qualified_actions, list):
                raise ValueError(f"trace {trace_id} has no qualified action prefix")
            actions = tuple(int(action) for action in raw_qualified_actions)
            if not actions or len(actions) > len(source_actions):
                raise ValueError(f"trace {trace_id} has an invalid qualified action prefix")
            if source_actions[: len(actions)] != actions:
                raise ValueError(f"trace {trace_id} qualification is not a source prefix")
            digests = tuple(str(value) for value in result.get("turn_digests", ()))
            if len(digests) != len(actions) + 1 or any(len(value) != 64 for value in digests):
                raise ValueError(f"trace {trace_id} has invalid per-turn observation digests")
            invalid_tails = [value for value in selected_window if value >= len(actions)]
            if invalid_tails:
                if len(invalid_tails) == 1:
                    raise ValueError(
                        f"trace-prefix tail {invalid_tails[0]} leaves no qualified prefix "
                        f"for {trace_id}"
                    )
                raise ValueError(
                    "trace-prefix tails "
                    + ", ".join(str(value) for value in invalid_tails)
                    + f" leave no qualified prefix for {trace_id}"
                )
            reset = EpisodeResetSpec.from_mapping(raw_trace["curriculum_reset"])
            trace = QualifiedActionTrace(
                trace_id=trace_id,
                seed=int(raw_trace["seed"]),
                reset_spec=reset,
                actions=actions,
                turn_digests=digests,
                source_policy_version=int(raw_trace["source_policy_version"]),
                source_global_step=int(raw_trace["source_global_step"]),
            )
            candidates.setdefault(trace.seed, []).append(trace)
        selected = tuple(
            min(values, key=lambda trace: (len(trace.actions), trace.trace_id))
            for _, values in sorted(candidates.items())
        )
        if not selected:
            raise ValueError("trace-prefix bank has no qualified traces")
        return cls(
            bank_path=bank_source,
            qualification_path=qualification_source,
            bank_sha256=str(bank["bank_sha256"]),
            qualification_sha256=_sha256_file(qualification_source),
            action_contract=action_contract,
            tail_actions=int(tail_actions),
            tail_action_window=selected_window,
            recurrent_state_mode=recurrent_state_mode,
            traces=selected,
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(trace.seed for trace in self.traces)

    def trace_for_seed(self, seed: int) -> QualifiedActionTrace:
        for trace in self.traces:
            if trace.seed == int(seed):
                return trace
        raise ValueError(f"game seed {seed} has no qualified trace prefix")

    def tail_for_episode(self, slot: int, episode_index: int) -> int:
        """Select a balanced handoff window independently of actor timing."""

        if slot < 0 or episode_index < 0:
            raise ValueError("trace-prefix slot and episode_index must be non-negative")
        return self.tail_action_window[
            (int(slot) + int(episode_index)) % len(self.tail_action_window)
        ]

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": 2 if len(self.tail_action_window) > 1 else 1,
            "kind": (
                "qualified-live-trace-window-v2"
                if len(self.tail_action_window) > 1
                else "qualified-live-trace-prefix-v1"
            ),
            "bank": str(self.bank_path),
            "bank_sha256": self.bank_sha256,
            "qualification": str(self.qualification_path),
            "qualification_sha256": self.qualification_sha256,
            "action_contract": self.action_contract,
            "tail_actions": self.tail_actions,
            "tail_action_window": list(self.tail_action_window),
            "prefix_actions": {
                str(trace.seed): len(trace.actions) - self.tail_actions
                for trace in self.traces
            },
            "prefix_actions_by_tail": {
                str(trace.seed): {
                    str(value): len(trace.actions) - value
                    for value in self.tail_action_window
                }
                for trace in self.traces
            },
            "recurrent_state_mode": self.recurrent_state_mode,
            "selected_traces": [
                {
                    "trace_id": trace.trace_id,
                    "seed": trace.seed,
                    "turns": len(trace.actions),
                    "source_policy_version": trace.source_policy_version,
                    "source_global_step": trace.source_global_step,
                }
                for trace in self.traces
            ],
            "guide_transitions_in_ppo": False,
            "state_semantics": "fresh-live-reset-qualified-action-prefix",
        }


class TracePrefixError(RuntimeError):
    """Raised when a qualified trace does not reproduce at training handoff."""

    def __init__(self, worker_id: str, trace_id: str, turn: int, reason: str) -> None:
        self.worker_id = worker_id
        self.trace_id = trace_id
        self.turn = turn
        self.reason = reason
        super().__init__(f"{worker_id} trace {trace_id} diverged at prefix turn {turn}: {reason}")
