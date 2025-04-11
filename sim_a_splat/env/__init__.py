"""Robot Environment Package for Simulation and Control

This package provides a collection of robot environment implementations including:
- SPLAT environments for object manipulation
- XArm environments for robot control
- Push environments for contact-rich manipulation

Each environment follows a common interface for consistent usage across different robots
and tasks.
"""

from .xarm.xarm_env import XarmSimEnv
from .scara.scara_env import ScaraSimEnv
from .splat.splat_scara_env import SplatEnv

__all__ = ["XarmSimEnv", "ScaraSimEnv", "SplatEnv"]
