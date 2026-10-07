"""Smoke-test the eval env by stepping it with RANDOM actions while showing the
live MuJoCo viewer. Nothing learned here -- it just confirms the env builds,
steps, renders, and that the viewer/reset/reward plumbing works end to end.

    uv run python test_env.py                 # 3 episodes, live window
    uv run python test_env.py --episodes 1 --max-steps 200
    uv run python test_env.py --no-view       # headless (CI / no display)
    uv run python test_env.py --hand sharpa
"""

from __future__ import annotations

import argparse

from build_urgantry import DEFAULT_HAND, HANDS
from env import SimGantryUR7eEnv


def run(episodes: int, max_steps: int, view: bool, seed: int, hand: str) -> None:
    env = SimGantryUR7eEnv(
        hand=hand,
        normalized_actions=True,   # Box(-1,1) so action_space.sample() is sane
        max_episode_steps=max_steps,
        show_viewer=view,
        spawn_props=True,
    )
    env.action_space.seed(seed)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()        # random control
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            # If the viewer window was closed, stop early.
            if view and env._viewer is not None and not env._viewer.is_running():
                done = True

        print(f"episode {ep}: steps={steps} reward={ep_reward:.3f} "
              f"success={info.get('success')} block_h={info.get('block_height'):.3f}")

    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hand", choices=HANDS, default=DEFAULT_HAND)
    p.add_argument("--no-view", dest="view", action="store_false",
                   help="run headless (no on-screen viewer)")
    p.set_defaults(view=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
