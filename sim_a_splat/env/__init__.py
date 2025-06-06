"""Robot Environment Package for Simulation and Control

This package provides a collection of robot environment implementations including:
- SPLAT environments for object manipulation
- XArm environments for robot control
- Push environments for contact-rich manipulation

Each environment follows a common interface for consistent usage across different robots
and tasks.
"""

# from .xarm.xarm_env import XarmSimEnv
# from .splat.splat_env import SplatEnv


# __all__ = ["XarmSimEnv", "SplatEnv"]

# from gymnasium.envs.registration import register

# register(
#     id="ManipulatorSimEnv-v0",
#     entry_point="sim_a_splat.env.manipulator.manipulator_env:ManipulatorSimEnv",
# )

from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
