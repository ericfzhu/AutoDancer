from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from autodancer.constants import (
    GRID_CHANNELS,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    PLAYER_FEATURES,
    Action,
    BossType,
    GridChannel,
    InventoryFeature,
    PlayerFeature,
    Terrain,
)
from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.bridge import CURRICULUM_PROFILES, BridgeCommand
from autodancer.live.protocol import (
    LOG_MARKER,
    SCHEMA_VERSION,
    CommandLifecycleError,
    JsonlTurnSource,
    NativePipeTurnSource,
    ProtocolError,
    QueueTurnSource,
    validate_record,
)


class FakeReceiver:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads

    def receive(self, timeout: float = 10.0, *, max_bytes: int = 262144) -> bytes:
        del timeout
        payload = self.payloads.pop(0)
        assert len(payload) <= max_bytes
        return payload


class FakeBridge:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.restarts = 0
        self.gotos: list[int] = []
        self.goto_profiles: list[str | None] = []

    def send_action(self, action: Action) -> BridgeCommand:
        self.actions.append(action)
        return BridgeCommand("test-session", len(self.actions), action, "ACTION", "worker-0000")

    def reset(self, seed: int) -> BridgeCommand:
        self.restarts += 1
        return BridgeCommand("test-session", self.restarts, None, "RESET", "worker-0000", seed)

    def goto_level(self, level: int, profile: str | None = None) -> BridgeCommand:
        self.gotos.append(level)
        self.goto_profiles.append(profile)
        return BridgeCommand(
            "test-session",
            self.restarts + len(self.gotos),
            None,
            "GOTO",
            "worker-0000",
            target_level=level,
            curriculum_profile=profile,
        )

    def restart(self) -> BridgeCommand:
        return self.reset(7)


def record(
    sequence: int,
    kind: str,
    *,
    status: str = "running",
    run_id: str = "run-7",
    events: list[dict] | None = None,
    requested_action: Action | None = None,
    command_id: int | None = None,
) -> dict:
    terminal = status in {"won", "dead"}
    truncated = status == "aborted"
    mask = np.zeros(11, dtype=int)
    if status == "running":
        mask[:5] = 1
    player = np.zeros(PLAYER_FEATURES, dtype=int)
    player[PlayerFeature.HEALTH] = 6
    player[PlayerFeature.MAX_HEALTH] = 6
    player[PlayerFeature.ZONE] = 1
    player[PlayerFeature.FLOOR] = 1
    player[PlayerFeature.WON] = int(status == "won")
    player[PlayerFeature.DEAD] = int(status == "dead")
    return {
        "message_type": "transition",
        "schema_version": SCHEMA_VERSION,
        "instance_id": "worker-0000",
        "role": "worker",
        "session_id": "test-session",
        "launch_id": "test-launch",
        "run_id": run_id,
        "sequence": sequence,
        "kind": kind,
        "game": {"version": "v4.2.1-b5713", "steam_build": "22938426"},
        "character": "Bard",
        "seed": 7,
        "zone": 1,
        "floor": 1,
        "observation": {
            "grid": np.zeros((21, 21, GRID_CHANNELS), dtype=int).tolist(),
            "player": player.tolist(),
            "inventory": np.zeros((INVENTORY_SLOTS, INVENTORY_FEATURES), dtype=int).tolist(),
            "action_mask": mask.tolist(),
            "map_bounds": {"x": -16, "y": -16, "width": 32, "height": 32},
        },
        "events": events or [],
        "episode_status": status,
        "terminated": terminal,
        "truncated": truncated,
        "metrics": {"turns": sequence},
        "bridge": {
            "kind": "RESET",
            "session_id": "test-session",
            "command_id": 1,
            "seed": 7,
        }
        if kind == "reset"
        else None
        if requested_action is None
        else {
            "kind": "ACTION",
            "session_id": "test-session",
            "command_id": sequence if command_id is None else command_id,
            "requested_action": int(requested_action),
            "engine_action": 1,
            "observed_action": 1,
        },
    }


def goto_record(
    sequence: int,
    level: int,
    zone: int,
    floor: int,
    profile: str | None = None,
) -> dict:
    payload = record(sequence, "turn")
    payload["zone"] = zone
    payload["floor"] = floor
    payload["observation"]["player"][PlayerFeature.ZONE] = zone
    payload["observation"]["player"][PlayerFeature.FLOOR] = floor
    payload["bridge"] = {
        "kind": "GOTO",
        "session_id": "test-session",
        "command_id": level,
        "target_level": level,
    }
    if profile is not None:
        payload["bridge"]["curriculum_profile"] = profile
        expected_health = int(profile.rsplit("player", 1)[1])
        objectives = [
            {"role": "boss", "type": 101, "health": 9, "max_health": 9}
        ]
        payload["bridge"]["curriculum_application"] = {
            "player_health": expected_health,
            "player_max_health": expected_health,
            "objective_state_unchanged": True,
            "objectives_before": [dict(value) for value in objectives],
            "objectives_after": [dict(value) for value in objectives],
        }
        payload["observation"]["player"][PlayerFeature.HEALTH] = expected_health
        payload["observation"]["player"][PlayerFeature.MAX_HEALTH] = expected_health
    return payload


def test_native_pipe_source_accepts_large_schema10_records_without_log_markers() -> None:
    reset = record(0, "reset")
    turn = record(1, "turn", requested_action=Action.UP)
    encoded = [json.dumps(item).encode("utf-8") for item in (reset, turn)]
    assert all(len(item) > 4096 for item in encoded)
    source = NativePipeTurnSource(FakeReceiver(encoded))
    assert source.read(1)["kind"] == "reset"
    assert source.read(1)["sequence"] == 1


def test_live_victory_terminates_and_reports_completion() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(0, "reset"),
            record(
                1,
                "terminal",
                status="won",
                events=[{"kind": "success", "amount": 1, "data": {"task_complete": True}}],
                requested_action=Action.RIGHT,
            ),
        ]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=sender,
        max_turns=10,
    )
    environment.reset(seed=7)
    observation, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    assert terminated and not truncated
    assert info["episode_status"] == "won"
    assert info["completed"] == 1
    assert observation["player"][PlayerFeature.WON] == 1
    assert reward == pytest.approx(50.0)
    assert info["extrinsic_reward"] == pytest.approx(50.0)
    assert info["shaping_reward"] == pytest.approx(0.0)
    assert info["reward_components"]["victory"] == 50.0
    with pytest.raises(RuntimeError, match="episode ended"):
        environment.step(Action.RIGHT)


def test_live_adapter_rederives_deployment_features() -> None:
    payload = record(0, "reset")
    payload["observation"]["player"][PlayerFeature.TASK] = BossType.CORAL_RIFF
    grid = np.asarray(payload["observation"]["grid"], dtype=int)
    grid[10, 10, GridChannel.TERRAIN_CLASS] = Terrain.STAIRS
    grid[10, 11, GridChannel.ACTOR_CLASS] = 2
    grid[10, 11, GridChannel.VISIBILITY] = 2
    payload["observation"]["grid"] = grid.tolist()
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([payload]),
        bridge=FakeBridge(),
        attach_existing=True,
    )
    observation, info = environment.reset()
    assert observation["player"][PlayerFeature.VISIBLE_ENEMIES] == 1
    assert observation["player"][PlayerFeature.ON_STAIRS] == 1
    assert observation["player"][PlayerFeature.TASK] == BossType.CORAL_RIFF
    assert info["boss_type"] == BossType.CORAL_RIFF


def test_live_environment_applies_client_turn_limit() -> None:
    source = QueueTurnSource([record(0, "reset"), record(1, "turn", requested_action=Action.UP)])
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=FakeBridge(),
        max_turns=1,
    )
    environment.reset(seed=7)
    _, reward, terminated, truncated, info = environment.step(Action.UP)
    assert not terminated and truncated
    assert info["episode_status"] == "time_limit"
    assert info["client_turn_limit"] == 1
    assert reward == pytest.approx(0.0)
    assert "aborted" not in info["reward_components"]
    assert not any(event.get("kind") == "failure" for event in info["raw_events"])


def test_natural_prefix_handoff_rebases_only_client_accounting() -> None:
    source = QueueTurnSource(
        [
            record(0, "reset"),
            record(1, "turn", requested_action=Action.UP),
            record(2, "turn", requested_action=Action.RIGHT),
            record(3, "turn", requested_action=Action.DOWN),
        ]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=FakeBridge(),
        max_turns=2,
    )
    observation, _ = environment.reset(seed=7)
    first_observation, _, _, first_truncated, first_info = environment.step(Action.UP)
    assert not first_truncated
    handed_observation, handed_info = environment.begin_learning_segment(
        first_info,
        metadata={"kind": "death-metal-natural-prefix-v1", "guide_turns": 1},
    )
    assert handed_observation is first_observation
    assert handed_observation is observation or np.array_equal(
        handed_observation["player"], observation["player"]
    )
    assert handed_info["run_id"] == "run-7"
    assert handed_info["seed"] == 7
    assert handed_info["turns"] == 0

    _, _, _, truncated, info = environment.step(Action.RIGHT)
    assert not truncated
    assert info["turns"] == 1
    assert info["learning_segment"]["guide_turns"] == 1
    _, _, _, truncated, info = environment.step(Action.DOWN)
    assert truncated
    assert info["turns"] == 2


def test_curriculum_reset_jumps_sequentially_without_reward_and_terminates_at_target() -> None:
    sender = FakeBridge()
    target = record(4, "turn", requested_action=Action.UP, command_id=1)
    target["zone"] = 2
    target["floor"] = 1
    target["observation"]["player"][PlayerFeature.ZONE] = 2
    target["observation"]["player"][PlayerFeature.FLOOR] = 1
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(
            [
                record(0, "reset"),
                goto_record(1, 2, 1, 2),
                goto_record(2, 3, 1, 3),
                goto_record(3, 4, 1, 4),
                target,
            ]
        ),
        bridge=sender,
        curriculum_start_level=4,
        curriculum_target_level=5,
    )

    observation, reset_info = environment.reset(seed=7)
    assert sender.gotos == [2, 3, 4]
    assert sender.goto_profiles == [None, None, None]
    assert tuple(observation["player"][[PlayerFeature.ZONE, PlayerFeature.FLOOR]]) == (1, 4)
    assert reset_info["curriculum_start_level"] == 4
    assert reset_info["curriculum_target_level"] == 5

    _, _, terminated, truncated, info = environment.step(Action.UP)
    assert terminated and not truncated
    assert info["episode_status"] == "curriculum_complete"
    assert info["curriculum_completed"] is True
    assert info["completed"] == 0
    assert info["deaths"] == 0
    assert "zone_complete" in info["reward_components"]
    assert info["reward_components"]["floor_complete"] == 5.0
    assert info["reward_components"]["zone_complete"] == 10.0
    assert "victory" not in info["reward_components"]
    assert info["extrinsic_reward"] == 15.0


def test_curriculum_profile_is_routed_only_to_the_start_floor() -> None:
    sender = FakeBridge()
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(
            [
                record(0, "reset"),
                goto_record(1, 2, 1, 2),
                goto_record(2, 3, 1, 3),
                goto_record(3, 4, 1, 4, "player20"),
            ]
        ),
        bridge=sender,
        curriculum_start_level=4,
        curriculum_target_level=5,
        curriculum_profile="player20",
    )
    _, info = environment.reset(seed=7)
    assert sender.gotos == [2, 3, 4]
    assert sender.goto_profiles == [None, None, "player20"]
    assert info["curriculum_profile"] == "player20"
    assert info["curriculum_observed_player_health"] == 20
    assert info["curriculum_observed_player_max_health"] == 20


def test_curriculum_profile_rejects_unapplied_player_health() -> None:
    sender = FakeBridge()
    assisted = goto_record(3, 4, 1, 4, "player8")
    assisted["observation"]["player"][PlayerFeature.HEALTH] = 6
    assisted["observation"]["player"][PlayerFeature.MAX_HEALTH] = 6
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(
            [
                record(0, "reset"),
                goto_record(1, 2, 1, 2),
                goto_record(2, 3, 1, 3),
                assisted,
            ]
        ),
        bridge=sender,
        curriculum_start_level=4,
        curriculum_target_level=5,
        curriculum_profile="player8",
    )

    with pytest.raises(ProtocolError, match="application mismatch"):
        environment.reset(seed=7)


def test_curriculum_profile_rejects_mutated_boss_objective_evidence() -> None:
    sender = FakeBridge()
    assisted = goto_record(3, 4, 1, 4, "player8")
    assisted["bridge"]["curriculum_application"]["objectives_after"][0]["health"] = 1
    assisted["bridge"]["curriculum_application"]["objective_state_unchanged"] = False
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(
            [
                record(0, "reset"),
                goto_record(1, 2, 1, 2),
                goto_record(2, 3, 1, 3),
                assisted,
            ]
        ),
        bridge=sender,
        curriculum_start_level=4,
        curriculum_target_level=5,
        curriculum_profile="player8",
    )

    with pytest.raises(ProtocolError, match="preserve boss objective state"):
        environment.reset(seed=7)


def test_per_episode_reset_options_override_fixed_defaults_and_are_reported() -> None:
    sender = FakeBridge()
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(
            [
                record(0, "reset"),
                goto_record(1, 2, 1, 2),
                goto_record(2, 3, 1, 3),
                goto_record(3, 4, 1, 4, "player10"),
            ]
        ),
        bridge=sender,
    )
    _, info = environment.reset(
        seed=7,
        options={
            "curriculum": {
                "id": "reduced",
                "start_level": 4,
                "target_level": 5,
                "profile": "player10",
            }
        },
    )
    assert sender.gotos == [2, 3, 4]
    assert sender.goto_profiles == [None, None, "player10"]
    assert info["curriculum_reset_id"] == "reduced"
    assert info["curriculum_reset"] == {
        "id": "reduced",
        "start_level": 4,
        "target_level": 5,
        "profile": "player10",
    }


def test_live_reset_rejects_unknown_options_before_sending_a_command() -> None:
    sender = FakeBridge()
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([]),
        bridge=sender,
    )
    with pytest.raises(ValueError, match="unknown live reset options"):
        environment.reset(seed=7, options={"surprise": True})
    assert sender.restarts == 0


def test_live_environment_scores_actual_health_loss_not_raw_attack_damage() -> None:
    reset_record = record(0, "reset")
    turn_record = record(
        1,
        "turn",
        requested_action=Action.UP,
        events=[
            {"kind": "player_damage", "amount": 999, "entity_id": 10},
            {"kind": "player_damage", "amount": 2, "entity_id": 11},
        ],
    )
    turn_record["observation"]["player"][PlayerFeature.HEALTH] = 2
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([reset_record, turn_record]),
        bridge=FakeBridge(),
    )
    environment.reset(seed=7)
    _, reward, terminated, truncated, info = environment.step(Action.UP)
    assert not terminated and not truncated
    damage_events = [event for event in info["raw_events"] if event.get("kind") == "player_damage"]
    assert len(damage_events) == 1
    assert damage_events[0]["amount"] == 4
    assert damage_events[0]["data"] == {"raw_damage": 1001, "event_count": 2}
    assert info["reward_components"]["player_damage"] == pytest.approx(-0.6)
    assert reward == pytest.approx(-0.6)


def test_live_environment_derives_item_pickups_from_inventory_delta() -> None:
    reset_record = record(0, "reset")
    reset_record["observation"]["inventory"][0][InventoryFeature.ITEM_TYPE] = 10
    turn_record = record(1, "turn", requested_action=Action.UP)
    turn_record["observation"]["inventory"][0][InventoryFeature.ITEM_TYPE] = 10
    turn_record["observation"]["inventory"][1][InventoryFeature.ITEM_TYPE] = 20
    environment = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([reset_record, turn_record]),
        bridge=FakeBridge(),
    )
    environment.reset(seed=7)
    _, _, _, _, info = environment.step(Action.UP)
    assert [event for event in info["raw_events"] if event.get("kind") == "item_collected"] == [
        {
            "kind": "item_collected",
            "amount": 1,
            "entity_id": 0,
            "data": {"item_type": 20, "source": "inventory_delta"},
        }
    ]


def test_terminal_record_may_mask_every_action() -> None:
    validate_record(record(3, "terminal", status="won"))
    payload = record(0, "reset")
    payload["observation"]["action_mask"] = [0] * 11
    with pytest.raises(ProtocolError, match="directions and WAIT"):
        validate_record(payload)


def test_protocol_detects_lost_turn_and_run_change() -> None:
    source = QueueTurnSource([record(0, "reset"), record(2, "turn")])
    source.read()
    with pytest.raises(ProtocolError, match="expected 1"):
        source.read()

    source = QueueTurnSource([record(0, "reset"), record(1, "turn", run_id="another-run")])
    source.read()
    with pytest.raises(ProtocolError, match="Run identity changed"):
        source.read()


def test_protocol_rejects_invalid_observation_and_event() -> None:
    payload = record(0, "reset")
    payload["observation"]["action_mask"][0] = 2
    with pytest.raises(ProtocolError, match="only 0 or 1"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["inventory"][0][1] = 4096
    with pytest.raises(ProtocolError, match="exact type identifier"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["grid"][10][10][GridChannel.FACING] = 5
    with pytest.raises(ProtocolError, match="facing"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["grid"][10][10][GridChannel.INTERACTION_FLAGS] = 32
    with pytest.raises(ProtocolError, match="interaction_flags"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["inventory"][0][6] = 2
    with pytest.raises(ProtocolError, match="ready/active"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["player"][PlayerFeature.SONG_END_REACHED] = 2
    with pytest.raises(ProtocolError, match="song-end"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["events"] = [{"kind": "enemy_damage", "amount": -1}]
    with pytest.raises(ProtocolError, match="at least 0"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["revealed_map"] = np.zeros((65, 64), dtype=int).tolist()
    with pytest.raises(ProtocolError, match="revealed_map.*shape"):
        validate_record(payload)

    payload = record(0, "reset")
    payload["observation"]["map_bounds"]["width"] = 0
    with pytest.raises(ProtocolError, match="width and height"):
        validate_record(payload)


def test_full_revealed_map_fits_pipe_frame_and_validates() -> None:
    payload = record(0, "reset")
    payload["observation"]["revealed_map"] = np.full((65, 65), 2, dtype=int).tolist()
    payload["observation"]["revealed_map_origin"] = {"x": -32, "y": -32}
    validate_record(payload)
    assert len(json.dumps(payload).encode("utf-8")) < 65_536


def test_revealed_map_origin_requires_integer_coordinates() -> None:
    payload = record(0, "reset")
    payload["observation"]["revealed_map"] = np.zeros((65, 65), dtype=int).tolist()
    payload["observation"]["revealed_map_origin"] = {"x": "bad", "y": 0}
    with pytest.raises(ProtocolError, match="revealed_map_origin.x"):
        validate_record(payload)


def test_log_source_establishes_boundary_before_fast_restart(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text("old line\n", encoding="utf-8")
    source = JsonlTurnSource(path)
    source.reset_sequence()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(LOG_MARKER + json.dumps(record(0, "reset")) + "\n")
    assert source.read(timeout=0.2)["sequence"] == 0


def test_live_environment_uses_shared_schema_and_reward_mapping() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(0, "reset"),
            record(
                1,
                "turn",
                events=[{"kind": "enemy_damage", "amount": 1}],
                requested_action=Action.RIGHT,
            ),
        ]
    )
    environment = AutoDancerLiveEnv(turn_source=source, bridge=sender)
    observation, info = environment.reset(seed=7)
    assert environment.observation_space.contains(observation)
    assert sender.restarts == 1
    assert info["protocol_schema_version"] == SCHEMA_VERSION
    _, reward, terminated, truncated, info = environment.step(Action.RIGHT)
    assert sender.actions == [Action.RIGHT]
    assert reward == pytest.approx(0.01)
    assert info["reward_components"]["enemy_damage"] == pytest.approx(0.01)
    assert not terminated and not truncated
    assert info["episode_status"] == "running"


def test_live_reset_ignores_post_death_turn_before_reset() -> None:
    source = QueueTurnSource(
        [
            record(83, "turn", run_id="dead-run"),
            record(0, "reset", run_id="restarted-run"),
        ]
    )
    environment = AutoDancerLiveEnv(turn_source=source, bridge=FakeBridge())
    _, info = environment.reset(seed=7)
    assert info["run_id"] == "restarted-run"


def test_queue_source_accepts_a_fresh_reset_after_attach() -> None:
    source = QueueTurnSource([record(7, "turn"), record(0, "reset", run_id="new-run")])
    source.read_latest()
    assert source.read()["sequence"] == 0


def test_log_source_reads_marker_and_logger_suffix(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        "unrelated log\n[Debug] '" + LOG_MARKER + json.dumps(record(0, "reset")) + "'\n",
        encoding="utf-8",
    )
    source = JsonlTurnSource(path, start_at_end=False)
    assert source.read(timeout=0.2)["run_id"] == "run-7"


def test_log_source_waits_for_a_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    payload = LOG_MARKER + json.dumps(record(0, "reset"))
    path.write_text(payload[:-10], encoding="utf-8")
    source = JsonlTurnSource(path, start_at_end=False)
    with pytest.raises(TimeoutError):
        source.read(timeout=0.02)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload[-10:] + "\n")
    assert source.read(timeout=0.2)["sequence"] == 0


def test_log_source_can_attach_to_latest_record(tmp_path: Path) -> None:
    path = tmp_path / "NecroDancer.log"
    path.write_text(
        "\n".join(
            [
                LOG_MARKER + json.dumps(record(3, "turn")),
                LOG_MARKER + json.dumps(record(4, "turn")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = JsonlTurnSource(path)
    assert source.read_latest(timeout=0.2)["sequence"] == 4
    with path.open("a", encoding="utf-8") as handle:
        handle.write(LOG_MARKER + json.dumps(record(5, "turn")) + "\n")
    assert source.read(timeout=0.2)["sequence"] == 5


def test_live_environment_can_attach_without_restart() -> None:
    sender = FakeBridge()
    source = QueueTurnSource(
        [
            record(7, "turn"),
            record(8, "turn", requested_action=Action.LEFT, command_id=1),
        ]
    )
    environment = AutoDancerLiveEnv(
        turn_source=source,
        bridge=sender,
        attach_existing=True,
    )
    _, info = environment.reset()
    assert info["sequence"] == 7
    assert sender.restarts == 0
    environment.step(Action.LEFT)
    assert sender.actions == [Action.LEFT]


def test_placeholder_build_is_rejected() -> None:
    payload = record(0, "reset")
    payload["game"]["version"] = "SET_GAME_VERSION"
    with pytest.raises(ProtocolError, match="Unsupported game build"):
        validate_record(payload)


def test_protocol_rejects_cross_worker_and_mismatched_seed() -> None:
    payload = record(0, "reset")
    payload["instance_id"] = "worker-0001"
    environment = AutoDancerLiveEnv(turn_source=QueueTurnSource([payload]), bridge=FakeBridge())
    with pytest.raises(ProtocolError, match="Expected worker 'worker-0000'"):
        environment.reset(seed=7)

    payload = record(0, "reset")
    payload["seed"] = 8
    environment = AutoDancerLiveEnv(turn_source=QueueTurnSource([payload]), bridge=FakeBridge())
    with pytest.raises(ProtocolError, match="acknowledgement mismatch"):
        environment.reset(seed=7)


def command_status(*, phase: str = "received", launch_id: str = "test-launch") -> dict:
    return {
        "message_type": "command_status",
        "schema_version": SCHEMA_VERSION,
        "instance_id": "worker-0000",
        "role": "worker",
        "session_id": "test-session",
        "launch_id": launch_id,
        "command_kind": "ACTION",
        "command_id": 2,
        "command_session_id": "test-session",
        "phase": phase,
        "reason": "accepted_action_no_turn" if phase == "command_error" else "",
        "requested_action": 0,
        "engine_state": {"tick": 10, "loading": False},
    }


def test_pipe_source_filters_lifecycle_messages_without_advancing_sequence() -> None:
    payloads = [
        json.dumps(command_status()).encode(),
        json.dumps(command_status(phase="accepted")).encode(),
        json.dumps(command_status(phase="input_observed")).encode(),
        json.dumps(command_status(phase="turn_completed")).encode(),
        json.dumps(record(0, "reset")).encode(),
    ]
    statuses: list[dict] = []
    source = NativePipeTurnSource(
        FakeReceiver(payloads),
        instance_id="worker-0000",
        session_id="test-session",
        launch_id="test-launch",
        status_callback=statuses.append,
    )
    assert source.read(1)["kind"] == "reset"
    assert [item["phase"] for item in statuses] == [
        "received",
        "accepted",
        "input_observed",
        "turn_completed",
    ]
    assert source.max_frame_bytes > 0


def test_pipe_source_rejects_stale_launch_and_structured_command_error() -> None:
    stale = NativePipeTurnSource(
        FakeReceiver([json.dumps(command_status(launch_id="old-launch")).encode()]),
        instance_id="worker-0000",
        session_id="test-session",
        launch_id="new-launch",
    )
    with pytest.raises(ProtocolError, match="Expected launch"):
        stale.read(1)

    failed = NativePipeTurnSource(
        FakeReceiver(
            [
                json.dumps(command_status()).encode(),
                json.dumps(command_status(phase="command_error")).encode(),
            ]
        ),
        instance_id="worker-0000",
        session_id="test-session",
        launch_id="test-launch",
    )
    with pytest.raises(CommandLifecycleError, match="accepted_action_no_turn"):
        failed.read(1)


def test_pipe_source_rejects_out_of_order_command_lifecycle() -> None:
    source = NativePipeTurnSource(
        FakeReceiver([json.dumps(command_status(phase="turn_completed")).encode()]),
        instance_id="worker-0000",
        session_id="test-session",
        launch_id="test-launch",
    )
    with pytest.raises(ProtocolError, match="unexpected command"):
        source.read(1)


def test_transition_rejects_floor_metadata_that_disagrees_with_observation() -> None:
    payload = record(1, "turn")
    payload["floor"] = 2
    with pytest.raises(ProtocolError, match="zone/floor"):
        validate_record(payload)


def test_lua_inventory_uses_the_gameplay_cooldown_component() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts" / "AutoDancer.lua"
    ).read_text(encoding="utf-8")
    assert "spellCooldownTime.remainingTurns" in source
    assert "itemHUDCooldown.remainingTurns" not in source
    assert "priceTagCostHealth.cost or 0" in source
    assert "priceTagCostHealth.costMultiplier" not in source


def test_lua_uses_the_trusted_bridge_to_pin_variant_content() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts" / "AutoDancer.lua"
    ).read_text(encoding="utf-8")
    for mod_name in ("Amplified", "DynChar", "Synchrony"):
        assert f'Native.blacklistMod("{mod_name}")' in source

    patcher = (Path(__file__).parents[1] / "tools" / "patch_wsp.py").read_text(
        encoding="utf-8"
    )
    assert 'native.blacklistMod = mods.blacklist' in patcher


def test_lua_telemetry_reuses_the_built_observation_and_bounds_collection() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts" / "AutoDancer.lua"
    ).read_text(encoding="utf-8")
    status_start = source.index("local function observationForStatus")
    status_end = source.index("\nend", status_start)
    status_function = source[status_start:status_end]
    assert "clone(" not in status_function
    assert "local result = observation or emptyObservation()" in status_function
    assert "local TELEMETRY_COLLECTION_INTERVAL = 1000" in source
    emit_start = source.index("local function emitRecord")
    emit_end = source.index("\nlocal function emitTurn", emit_start)
    emit_function = source[emit_start:emit_end]
    assert emit_function.index("Native.collect()") < emit_function.index("Native.send(")


def test_lua_player_health_profiles_preserve_boss_and_boss_add_state() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts" / "AutoDancer.lua"
    ).read_text(encoding="utf-8")
    start = source.index("local function applyCurriculumProfile")
    end = source.index("\nlocal function itemKind", start)
    application = source[start:end]
    assert 'string.match(profile, "^player(%d+)$")' in application
    assert "entity.health.health = 1" not in application
    assert "objectiveHealthSnapshot" in application
    assert "objective_state_unchanged" in application
    assert 'hasComponent(entity, "bossAdd")' in source


def test_lua_bridge_and_python_use_the_same_curriculum_profile_whitelist() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts" / "Bridge.lua"
    ).read_text(encoding="utf-8")
    declared = set(
        re.findall(r'^\s*\["([^"]+)"\]\s*=\s*true,?$', source, re.MULTILINE)
    )

    assert declared == set(CURRICULUM_PROFILES)
    assert not any(profile.startswith("boss1hp-") for profile in declared)


def test_lua_reset_acknowledgement_waits_for_the_new_run() -> None:
    root = Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts"
    telemetry = (root / "AutoDancer.lua").read_text(encoding="utf-8")
    bridge = (root / "Bridge.lua").read_text(encoding="utf-8")
    emit_turn = telemetry.index("local function emitTurn")
    begin = telemetry.index("local newRun = beginRunIfNeeded()", emit_turn)
    consume = telemetry.index("Bridge.consumeCompletedCommand(newRun)", begin)
    assert begin < consume
    assert 'completed.kind == "RESET" and not allowReset' in bridge


def test_lua_action_acknowledgement_waits_for_pending_level_transition() -> None:
    root = Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts"
    telemetry = (root / "AutoDancer.lua").read_text(encoding="utf-8")
    bridge = (root / "Bridge.lua").read_text(encoding="utf-8")

    assert 'require "necro.client.LevelTransition"' in bridge
    assert 'require "necro.game.tile.LevelExit"' in bridge
    assert (
        'actionCompletionBlockReason(completed)' in bridge
    )
    assert "local outstanding = pending or completed or queuedCommand" in bridge
    assert 'return "level_transition_pending"' in bridge
    assert 'return "unlocked_exit_pending"' in bridge
    assert 'return "exit_activation_pending"' in bridge
    assert "local EXIT_ACTIVATION_GRACE_TICKS = 6" in bridge
    assert 'return "command_completion_pending"' in bridge
    assert '"action_completion_timeout"' in bridge

    tick_observer = telemetry.index('event.tick.add("emitAutoDancerInitialObservation"')
    identity_gate = telemetry.index("levelIdentity ~= lastLevelIdentity", tick_observer)
    completed_gate = telemetry.index("Bridge.hasCompletedCommand()", identity_gate)
    emit = telemetry.index("emitTurn()", identity_gate)
    assert identity_gate < completed_gate < emit


def test_live_action_can_acknowledge_on_the_materialized_next_floor() -> None:
    reset = record(0, "reset")
    next_floor = record(1, "turn", requested_action=Action.UP, command_id=1)
    next_floor["zone"] = 2
    next_floor["floor"] = 1
    next_floor["observation"]["player"][PlayerFeature.ZONE] = 2
    next_floor["observation"]["player"][PlayerFeature.FLOOR] = 1

    statuses = []
    for phase in ("received", "accepted", "input_observed", "turn_completed"):
        status = command_status(phase=phase)
        status["command_id"] = 1
        statuses.append(status)
    heartbeat = command_status(phase="heartbeat")
    heartbeat["command_id"] = 1
    heartbeat["engine_state"] = {"tick": 11, "loading": True}
    statuses.append(heartbeat)
    telemetry_sent = command_status(phase="telemetry_sent")
    telemetry_sent["command_id"] = 1
    statuses.append(telemetry_sent)

    payloads = [json.dumps(reset).encode()]
    payloads.extend(json.dumps(status).encode() for status in statuses)
    payloads.append(json.dumps(next_floor).encode())
    source = NativePipeTurnSource(
        FakeReceiver(payloads),
        instance_id="worker-0000",
        session_id="test-session",
        launch_id="test-launch",
    )
    environment = AutoDancerLiveEnv(turn_source=source, bridge=FakeBridge())

    environment.reset(seed=7)
    observation, _, terminated, truncated, info = environment.step(Action.UP)

    assert not terminated
    assert not truncated
    assert info["zone"] == 2
    assert info["floor"] == 1
    assert observation["player"][PlayerFeature.ZONE] == 2
    assert source.last_status is not None
    assert source.last_status["phase"] == "telemetry_sent"


def test_lua_waits_for_a_materialized_visible_world_before_reset_or_action() -> None:
    root = Path(__file__).parents[1] / "mods" / "AutoDancer" / "scripts"
    telemetry = (root / "AutoDancer.lua").read_text(encoding="utf-8")
    bridge = (root / "Bridge.lua").read_text(encoding="utf-8")

    emit_turn = telemetry.index("local function emitTurn")
    begin_run = telemetry.index("local newRun = beginRunIfNeeded()", emit_turn)
    readiness_gate = telemetry.index("not observationReady()", emit_turn)
    assert readiness_gate < begin_run
    assert "Tile.exists(player.position.x, player.position.y)" in telemetry
    assert "Vision.isVisible(player.position.x, player.position.y)" in telemetry
    assert 'return "world_not_ready"' in bridge
    assert r'\"world_ready\"' in bridge
