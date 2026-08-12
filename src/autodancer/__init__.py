"""AutoDancer Gymnasium environments."""

from gymnasium.envs.registration import register, registry

__version__ = "0.1.0"


def _register(environment_id: str, entry_point: str, **kwargs: object) -> None:
    if environment_id not in registry:
        register(id=environment_id, entry_point=entry_point, **kwargs)


_register("AutoDancer-Sim-v0", "autodancer.envs:AutoDancerSimEnv")
_register("AutoDancer-Live-v0", "autodancer.envs:AutoDancerLiveEnv")

for _task_name in (
    "navigation",
    "single_enemy",
    "mixed_room",
    "floor",
    "zone",
    "all_zones",
):
    _register(
        f"AutoDancer-{_task_name.replace('_', '-').title()}-v0",
        "autodancer.envs:AutoDancerSimEnv",
        kwargs={"task": _task_name},
    )

