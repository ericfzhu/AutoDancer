"""AutoDancer Gymnasium environments."""

from gymnasium.envs.registration import register, registry

__version__ = "0.1.0"


def _register(environment_id: str, entry_point: str, **kwargs: object) -> None:
    if environment_id not in registry:
        register(id=environment_id, entry_point=entry_point, **kwargs)


_register("AutoDancer-Live-v0", "autodancer.envs:AutoDancerLiveEnv")
