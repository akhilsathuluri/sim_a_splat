from . import env
from . import common
from . import splat
from . import messaging
from . import ellipsoids

from gymnasium.envs.registration import register

__all__ = [
    # Submodules
    'env',
    'common', 
    'splat',
    'messaging',
    'ellipsoids',
]

register(
    id='sim_a_splat/Scara-v0',
    entry_point='sim_a_splat.env.splat.splat_scara_env:SplatEnv',
    max_episode_steps=300,
)