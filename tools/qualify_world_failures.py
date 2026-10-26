"""Exercise collection failure boundaries with frozen weights and injected faults."""

import copy
from pathlib import Path

import torch

from autodancer.constants import Action
from autodancer.envs.vector import AutoDancerVectorEnv, VectorInfrastructureError
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.model import ModelConfig, WorldActorCritic


class ExistingWorker:
    """Use the qualifier's existing connection; never spawn or recover a worker."""

    num_envs = 1
    _failure = AutoDancerVectorEnv._failure

    def __init__(self, env, supervisor):
        self.supervisor = supervisor
        self.worker_ids = supervisor.worker_ids
        self.environments = {self.worker_ids[0]: env}
        self.infrastructure_events = []
        self._invalid = False

    def reset(self, seeds, options=None):
        obs, info = self.environments[self.worker_ids[0]].reset(seed=seeds[0], options=options[0])
        return {key: value[None] for key, value in obs.items()}, [info]

    def recover(self, *args, **kwargs):
        raise AssertionError("Qualification must not recover failed workers")


def qualify_failures(env, supervisor, result):
    torch.set_num_threads(1)
    torch.manual_seed(25)
    model = WorldActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
            map_size=8,
        ),
        equipment_capacity=env.equipment_capacity,
        equipment_profile=env.equipment_profile,
        world_capacity=env.world_capacity,
        initialize=False,
    ).eval()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    step = model.step

    def waiting_step(*args, **kwargs):
        logits, value, hidden = step(*args, **kwargs)
        logits = torch.full_like(logits, -1e6)
        logits[:, int(Action.WAIT)] = 0
        return logits, value, hidden

    model.step = waiting_step
    evidence = []
    from autodancer.harness_contract import information_contract, task_contract
    from autodancer.rewards import reward_semantics

    for target in (None, 5):
        observation, info = env.reset(
            seed=25,
            options={"curriculum": dict(id="audit", start_level=1, target_level=target)},
        )
        expected = [1, 0] if target is None else [2, target]
        assert observation["task_context"].tolist() == expected
        assert info["information_contract"] == information_contract(
            env.equipment_profile, world=True
        )
        tensors = {k: torch.from_numpy(v).unsqueeze(0) for k, v in observation.items()}
        with torch.inference_mode():
            baseline = step(tensors, model.initial_state(1))
            tensors["previous_reward"] = torch.tensor([12345.0])
            changed = step(tensors, model.initial_state(1))
        assert all(torch.equal(a, b) for a, b in zip(baseline, changed, strict=True))
        evidence.append(dict(case="task_and_feedback", target=target, passed=True))
    result["information_contract"] = information_contract(env.equipment_profile, world=True)
    result["task_contract"] = task_contract()
    result["reward_semantics"] = reward_semantics()
    accept = env._accept_record
    capacity = env.world_capacity
    for fault in (None, "identity", "capacity", "timeout", "abort"):
        wrapper = ExistingWorker(env, supervisor)
        collector = VersionedAsyncRolloutCollector(
            wrapper, model, device=torch.device("cpu"), seed=25
        )
        seen = 0

        def inject(record, fault=fault):
            nonlocal seen
            if record["kind"] == "turn":
                seen += 1
                if seen == 2:
                    if fault == "timeout":
                        raise TimeoutError("Injected qualification timeout")
                    if fault == "abort":
                        record["episode_status"] = "aborted"
                    if fault == "identity":
                        record["observation"]["world_snapshot"]["entities"][0]["name"] = (
                            "UnregisteredWorldFixture"
                        )
                    if fault == "capacity":
                        env.world_capacity = 1
                        rows = record["observation"]["world_snapshot"]["entities"]
                        extra = copy.deepcopy(rows[0])
                        extra["id"] = max(r["id"] for r in rows) + 100000
                        rows.append(extra)
            return accept(record)

        env._accept_record = inject
        try:
            if fault is None:
                batch = collector.collect(2)
                assert batch.observations["world_entities"].dtype == torch.int32
                assert batch.observations["world_current"].all()
                assert batch.actions.shape == (2, 1)
                assert batch.observations["task_context"][..., 0].eq(1).all()
                evidence.append(dict(case="frozen_collection", steps=2, passed=True))
            else:
                try:
                    collector.collect(3)
                except VectorInfrastructureError as error:
                    diagnostic = wrapper.infrastructure_events[-1]
                    assert diagnostic["fragment_valid"] is False
                    assert len(diagnostic["partial_fragment"]["rewards"]) == 1
                    bundle = supervisor.workers[supervisor.worker_ids[0]].failure_history[-1]
                    assert Path(bundle["bundle_path"]).is_file()
                    evidence.append(
                        dict(
                            case=fault,
                            passed=True,
                            error=str(error),
                            partial_steps=1,
                            bundle=bundle["bundle_path"],
                        )
                    )
                else:
                    raise AssertionError("Injected failure did not stop collection")
                try:
                    collector.collect(1)
                except RuntimeError as error:
                    assert "invalid" in str(error)
                else:
                    raise AssertionError("Invalid collector was reusable")
        finally:
            env._accept_record = accept
            env.world_capacity = capacity
            collector.close()
        assert supervisor.workers[supervisor.worker_ids[0]].restart_count == 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())
    result["failure_checks"] = evidence
    result["model_weights_unchanged"] = True
    result["optimizer_updates"] = 0
