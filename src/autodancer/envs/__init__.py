from typing import TYPE_CHECKING, Any

from autodancer.envs.live import AutoDancerLiveEnv

if TYPE_CHECKING:
    from autodancer.envs.vector import AutoDancerVectorEnv

__all__ = ["AutoDancerLiveEnv", "AutoDancerVectorEnv"]


def __getattr__(name: str) -> Any:
    if name == "AutoDancerVectorEnv":
        from autodancer.envs.vector import AutoDancerVectorEnv

        return AutoDancerVectorEnv
    raise AttributeError(name)
