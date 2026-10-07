"""Gymnasium environment for the gantry-mounted bimanual UR7e tabletop, used as
an EVALUATION interface for learned policies (VLAs / behavior cloning) — not for
training.

Same interface as urtable_sim's env; only the scene differs (two arms hanging off
a central column over the half-depth tabletop, see build_urgantry.py).

The observation is camera images + joint positions. Two consumers share this env:

  * OGPO  — imports this env directly and runs it in-process. It expects a
            Gymnasium env whose observation is a dict with 'state' (proprio) and
            'image' (HWC), an action space of Box(-1, 1), and info['success'].
            Use normalized_actions=True for that.

  * OpenPi — the policy runs as a remote websocket server; the sim is a client
             that drives this env and uses `get_openpi_observation()` to build
             the server's obs dict.

The simulator is the world model standing in for hardware; the policy lives
elsewhere (in-process for OGPO, over the network for OpenPi).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import mujoco
import mujoco.viewer

try:
    # Installed-package form: `import urgantry_sim` -> this file loads as
    # `urgantry_sim.env`, which has a parent package, so the relative import resolves.
    from .build_urgantry import (
        build_model, set_initial_pose, block_height, pick_success, BLOCK_REST_Z,
        apply_initial_view, DEFAULT_HAND,
    )
except ImportError:
    # Dev/script form: run directly from inside urgantry_sim/ (e.g. `python
    # test_env.py`), where this file has no parent package and urgantry_sim/ is
    # on sys.path instead.
    from build_urgantry import (
        build_model, set_initial_pose, block_height, pick_success, BLOCK_REST_Z,
        apply_initial_view, DEFAULT_HAND,
    )


class SimGantryUR7eEnv(gym.Env):
    """Two 6-DOF UR7e arms hanging off a gantry column, each with a 5-finger
    hand, position-controlled. hand='wuji' (default): 52 actuators, 12 arm joints
    + 2 x 20 hand joints. hand='sharpa': 56 actuators, 12 arm + 2 x 22 hand.

    Task: lift the block off the table (see build_urgantry.pick_success).
    spawn_props=False (default) builds the bare scene (no block, no cardboard tray); reward
    is then always 0 and the episode only ends by truncation.

    Action: nu actuator targets (left arm, left hand, right arm, right hand).
    With normalized_actions=True the action space is Box(-1, 1) (OGPO style)
    mapped onto each actuator's ctrlrange; otherwise raw ctrlrange (all joints in
    radians; all-zero hand ctrl is the flat open hand).

    Observation: {'state': (nu,) actuator joint positions,
                  'image': (H, W, 3) uint8 — the top1 camera}.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        image_size: int = 224,
        control_hz: float = 20.0,
        max_episode_steps: int = 400,
        normalized_actions: bool = False,
        prompt: str = "",
        show_viewer: bool = False,
        spawn_props: bool = False,
        hand: str = DEFAULT_HAND,
    ):
        super().__init__()
        self.spawn_props = spawn_props
        self.hand = hand
        self.model = build_model(spawn_props, hand)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.normalized_actions = normalized_actions
        self.prompt = prompt
        self.max_episode_steps = max_episode_steps
        self._block_rest_z = BLOCK_REST_Z

        # Optional live, on-screen viewer (separate from the offscreen camera
        # renderer that feeds the policy). Handy for watching an eval run.
        self.show_viewer = show_viewer
        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False)
            apply_initial_view(self._viewer)
            self._viewer.sync()

        # Control runs slower than physics: step() advances n_substeps physics
        # ticks so one env step == one control tick at control_hz.
        self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self._step_count = 0

        self._renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)

        # Camera roles -> MuJoCo camera names. 'top1' feeds the policy obs 'image'.
        self.cameras = {
            "top1": "top1",
        }

        # --- action space ---
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        nu = self.model.nu
        if normalized_actions:
            self.action_space = Box(low=-1.0, high=1.0, shape=(nu,), dtype=np.float32)
        else:
            self.action_space = Box(
                low=self._ctrl_low.astype(np.float32),
                high=self._ctrl_high.astype(np.float32),
                shape=(nu,), dtype=np.float32,
            )

        # --- observation space ---
        self.observation_space = Dict({
            "state": Box(low=-np.inf, high=np.inf, shape=(nu,), dtype=np.float32),
            "image": Box(low=0, high=255,
                         shape=(image_size, image_size, 3),  # top1 RGB
                         dtype=np.uint8),
        })

    # ------------------------------------------------------------------ helpers
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalized [-1, 1] action onto the actuator ctrlrange."""
        action = np.clip(action, -1.0, 1.0)
        return self._ctrl_low + (action + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    def render_cameras(self, roles=None) -> dict[str, np.ndarray]:
        """Render the requested camera roles (default: all of self.cameras).
        Returns {role: (H, W, 3) uint8 RGB}."""
        roles = roles if roles is not None else self.cameras.keys()
        frames = {}
        for role in roles:
            self._renderer.update_scene(self.data, camera=self.cameras[role])
            frames[role] = self._renderer.render()  # uint8 RGB
        return frames

    def _proprio(self) -> np.ndarray:
        """Per-actuator joint position (nu,): 12 arm joints + both hands' joints.
        Read via each actuator's driven joint so it's robust to qpos layout."""
        m, d = self.model, self.data
        out = np.empty(m.nu, dtype=np.float32)
        for i in range(m.nu):
            out[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        return out

    def _get_obs(self) -> dict:
        frames = self.render_cameras(["top1"])  # OGPO obs is top1 only
        return {"state": self._proprio(), "image": frames["top1"]}

    def get_openpi_observation(self, prompt: str | None = None) -> dict:
        """Build an OpenPi server observation dict from the current sim state.

        Maps the top camera to observation/image; state is the raw joint positions;
        prompt is the task instruction. Adjust the key mapping to match your
        checkpoint.
        """
        frames = self.render_cameras()
        return {
            "observation/image": frames["top1"],
            "observation/state": self._proprio(),
            "prompt": prompt if prompt is not None else self.prompt,
        }

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        set_initial_pose(self.model, self.data)   # qpos + ctrl to the hard-coded pose
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._sync_viewer()
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.model.nu)
        ctrl = self._denormalize_action(action) if self.normalized_actions else \
            np.clip(action, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        self._sync_viewer()
        obs = self._get_obs()

        truncated = self._step_count >= self.max_episode_steps
        if not self.spawn_props:
            return obs, 0.0, False, truncated, {"success": 0}

        # Pick task: lift the block. Sparse success reward + a small shaping term
        # on block height so the signal isn't completely flat for eval logging.
        success = pick_success(self.model, self.data)
        height_gain = block_height(self.model, self.data) - self._block_rest_z
        reward = 1.0 if success else max(0.0, height_gain)
        terminated = bool(success)
        info = {"success": int(success), "block_height": block_height(self.model, self.data)}
        return obs, reward, terminated, truncated, info

    def _sync_viewer(self) -> None:
        """Push current physics state to the on-screen viewer, if one is open."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
