from .manipulator.manipulator_env import ManipulatorSimEnv
from .manipulator.manipulator_eef_wrapper import ManipulatorEEFWrapper
from .splat.splat_env_wrapper import SplatEnvWrapper

__all__ = [
    "ManipulatorSimEnv",
    "ManipulatorEEFWrapper",
    "SplatEnvWrapper",
]

# from gymnasium.envs.registration import register

# register(
#     id="ManipulatorSimEnv-v0",
#     entry_point="sim_a_splat.env.manipulator.manipulator_env:ManipulatorSimEnv",
# )
