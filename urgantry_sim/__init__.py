"""Gantry-mounted bimanual UR7e MuJoCo tabletop sim, packaged as a Gymnasium
environment.

Same env interface as urtable_sim, over the gantry scene: two UR7e arms hanging
off a central column above a 112 x 58.5 cm exposed tabletop (build_urgantry.py).

Installing this package registers the env id below as an import side effect,
so any Gymnasium-based RL/eval code can do:

    import gymnasium as gym
    import urgantry_sim  # noqa: F401 -- registers "SimGantryUR7e-v0"

    env = gym.make("SimGantryUR7e-v0")                     # bare scene
    pick = gym.make("SimGantryUR7e-v0", spawn_props=True)  # with cube + tray

See env.py:SimGantryUR7eEnv for the observation/action spec and task.
"""

import gymnasium as gym

from .env import SimGantryUR7eEnv

ENV_ID = "SimGantryUR7e-v0"

if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="urgantry_sim.env:SimGantryUR7eEnv",
        # No max_episode_steps here: the env truncates itself
        # (max_episode_steps kwarg on SimGantryUR7eEnv), so wrapping it in
        # gym's own TimeLimit too would double up truncation.
    )

__all__ = ["SimGantryUR7eEnv", "ENV_ID"]
