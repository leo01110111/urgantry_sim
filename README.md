# urgantry_sim

MuJoCo sim of the gantry-mounted bimanual setup: two UR7e arms hang upside down
off a central column over a 112 x 58.5 cm tabletop, each with a 5-finger hand:
**Wuji** (default) or **Sharpa Wave**, switchable with one setting. It can be used
two ways:

1. **As a Gymnasium env** (`SimGantryUR7e-v0`) to run or evaluate a policy.
2. **As a raw MuJoCo model** (`build_urgantry.build_scene`) for scripting, IK,
   data generation, or anything that needs direct `MjModel` / `MjData` access.
3. **As plain MJCF files** (`scene_wuji.xml`, `scene_sharpa.xml` in the repo
   root) for tools that only take an XML file, with no Python package needed.

The arm MJCF lives at `universal_robots_ur5e/ur5e.xml` for historical reasons;
the robot is a UR7e.

## Install

```bash
git clone https://github.com/leo01110111/urgantry_sim.git
cd urgantry_sim
uv sync --extra viz          # or: pip install -e ".[viz]"
```

The `viz` extra (OpenCV) is only needed for `test_viz.py`.

Headless machines need an offscreen GL backend for camera rendering:

```bash
export MUJOCO_GL=egl         # or osmesa
```

Quick checks:

```bash
cd urgantry_sim
uv run python build_urgantry.py               # interactive viewer, no stepping logic
uv run python build_urgantry.py --export scene.xml   # write the scene as one MJCF
uv run python test_env.py --no-view           # random actions through the gym env
uv run python test_viz.py                     # viewer + top1 camera window
```

All three take `--hand wuji|sharpa`; `build_urgantry.py` and `test_viz.py` also
take `--props` to spawn the block and tray.

Developed against MuJoCo 3.9. Newer versions print "Attach conflict" warnings
while building the model; they are harmless (the root model's solver options win).

## Scene

- World frame: origin on the floor at the center of the exposed table; +x right
  along the mount edge, +y away from the mount, +z up. Meters.
- Table board top: z = 0.775 (`BOARD_TOP`). Exposed area: x in [-0.56, 0.56],
  y in [-0.2925, 0.2925].
- Physics timestep: 2 ms. Every actuator is a position servo (ctrl = joint
  target in radians).
- One camera, `top1`: wide-angle (80 deg fovy) on the front of the gantry head,
  looking down the workspace past the arms.
- Optional props (`spawn_props=True`): a 5 cm green cube (`block`) at about
  (0.32, -0.07) and an open cardboard tray (`cardboard_box`) mirrored at x = -0.32.
  **The default is the bare scene without them.**

### Hands

`hand="wuji"` (default) or `hand="sharpa"` on the gym env and on every builder.
Both bolt onto the same 63 mm flange adapter cylinder with the same orientation:
palms toward the table at home, fingers along +y, thumbs on the inner side.

| | Wuji | Sharpa Wave |
|---|---|---|
| joints per hand | 20 | 22 |
| total actuators (`nu`) | 52 | 56 |
| actuator names | `{side}_hand_finger{1..5}_joint{1..4}` | `{side}_hand_{thumb,index,middle,ring,pinky}_{...}` |
| palm body | `{side}_hand_palm_link` | `{side}_hand_hand_C_MC` |

### Actuators

Order: left arm (6), left hand (H), right arm (6), right hand (H), with H = 20
(Wuji) or 22 (Sharpa).

| block | names | ctrlrange (rad) |
|---|---|---|
| left arm | `left_shoulder_pan`, `left_shoulder_lift`, `left_elbow`, `left_wrist_1`, `left_wrist_2`, `left_wrist_3` | +-2pi (elbow +-2.79) |
| left hand, Wuji | `left_hand_finger{1..5}_joint{1..4}`, finger-major | +-1.57 |
| left hand, Sharpa | `left_hand_thumb_{CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP}`, `left_hand_{index,middle,ring}_{MCP_FE, MCP_AA, PIP, DIP}`, `left_hand_pinky_{CMC, MCP_FE, MCP_AA, PIP, DIP}` | per joint, the joint's range |
| right arm / hand | same with `right_` | same |

All-zero hand ctrl is the flat open hand for both. Wuji: on fingers 2-5 `joint1`
flexes at the knuckle, `joint2` spreads, `joint3..4` curl. Sharpa: `_FE` / `PIP` /
`DIP` / `IP` flex (negative on the right hand, positive on the left), `_AA`
spreads. Use `set_hand` (below) for an open/fist command that works for both.

---

## Use case 1: the Gymnasium env

```python
import gymnasium as gym
import urgantry_sim  # registers SimGantryUR7e-v0

env = gym.make("SimGantryUR7e-v0")                     # bare scene
env = gym.make("SimGantryUR7e-v0", spawn_props=True)   # block-lift task
env = gym.make("SimGantryUR7e-v0", hand="sharpa")      # Sharpa Wave hands

obs, info = env.reset()
for _ in range(400):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### Constructor kwargs

| kwarg | default | meaning |
|---|---|---|
| `hand` | `"wuji"` | `"wuji"` or `"sharpa"` |
| `spawn_props` | `False` | add the block and tray (enables the pick task) |
| `normalized_actions` | `False` | `True`: actions in [-1, 1] mapped onto each ctrlrange |
| `control_hz` | `20.0` | policy rate; each `step()` runs `round(1 / (control_hz * 0.002))` physics steps |
| `max_episode_steps` | `400` | env truncates itself; no gym `TimeLimit` wrapper is added |
| `image_size` | `224` | square size of the `top1` render |
| `prompt` | `""` | instruction string returned by `get_openpi_observation()` |
| `show_viewer` | `False` | open a live MuJoCo viewer window alongside |

### Spaces

- **Action** `Box((nu,), float32)`, nu = 52 (Wuji) or 56 (Sharpa): actuator
  targets in the order above.
  Raw radians clipped to ctrlrange, or [-1, 1] with `normalized_actions=True`.
- **Observation** `Dict`:
  - `state`: `(nu,)` float32 joint positions, same order as the action.
  - `image`: `(image_size, image_size, 3)` uint8 RGB from `top1`.

### Reward and termination

- With `spawn_props=True`: `reward = 1.0` once the block center is 5 cm above its
  rest height (also sets `terminated=True`), otherwise `max(0, block lift in m)`.
  `info = {"success": 0|1, "block_height": z}`.
- With the bare scene: reward is always 0, `info = {"success": 0}`, and episodes
  only end by truncation.

### Reset

Deterministic: both arms go to the home pose (hands open, palms down over the
table, fingers pointing +y) with actuators commanded to hold it; props go back to
fixed rest poses. No randomization.

### Remote policies (OpenPi)

`env.unwrapped.get_openpi_observation(prompt=None)` returns
`{"observation/image", "observation/state", "prompt"}` for sending to an OpenPi
websocket server; feed the returned action chunk back through `env.step()`.

---

## Use case 2: the sim directly

`build_urgantry` exposes the model builder and helpers without any gym wrapper.

```python
import mujoco
import numpy as np
from urgantry_sim.build_urgantry import build_scene, set_hand, HAND_CLOSED

model, data = build_scene()                   # or build_scene(spawn_props=True, hand="sharpa")
# data is at the home pose, ctrl holds it, mj_forward has been run.

# Command joints by actuator name (position targets, radians).
data.ctrl[model.actuator("right_elbow").id] -= 0.2
set_hand(model, data, "left", HAND_CLOSED)    # left hand to a fist, either hand model

for _ in range(500):                          # 1 s at the 2 ms timestep
    mujoco.mj_step(model, data)

# Read state.
q = data.qpos[model.joint("right_elbow_joint").qposadr[0]]
palm = data.body("right_flange_adapter").xpos

# Render the top camera.
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="top1")
rgb = renderer.render()                       # (480, 640, 3) uint8
```

### Builders

| function | returns |
|---|---|
| `build_spec(spawn_props=False, hand="wuji")` | editable `mujoco.MjSpec`; add bodies, cameras, sensors before compiling |
| `build_model(spawn_props=False, hand="wuji")` | compiled `MjModel` |
| `build_scene(spawn_props=False, hand="wuji")` | `(model, data)` at the home pose, ready to step |
| `set_initial_pose(model, data)` | resets arms and hands to home and props to rest (call `mj_forward` after) |

### Helpers and constants

- `set_hand(model, data, side, closure)`: command one hand from open
  (`HAND_OPEN` = 0) to a fist (`HAND_CLOSED` = 1); works for either hand.
  `hand_pose(model, side, closure)` returns the targets without writing them.
- `hand_actuators(model, side)`: that hand's actuator names in order.
  `hand_type(model)`: `"wuji"` or `"sharpa"`.
- `HANDS`, `DEFAULT_HAND`.
- `LEFT_HOME_POSE`, `RIGHT_HOME_POSE`: home joint angles, in `ARM_JOINTS` order.
- `block_height(model, data)`, `pick_success(model, data)`, `has_props(model)`.
- Geometry: `BOARD_TOP`, `HALF_LEN`, `Y0`, `Y1`, `BLOCK_INIT_POS`, `BOX_INIT_POS`.

### Names worth knowing

- Joints: `{left,right}_{shoulder_pan,shoulder_lift,elbow,wrist_1,wrist_2,wrist_3}_joint`,
  hand joints named like their actuators; with props also the free joints
  `block_joint` and `cardboard_box_joint` (qpos `[x, y, z, qw, qx, qy, qz]`).
- Bodies: `{side}_flange_adapter`, the palm (see Hands), `block`, `cardboard_box`.
- Sites: `{side}_attachment_site` (UR tool flange), `{side}_ft_site`.
- Sensors: `{side}_ft_force`, `{side}_ft_torque`. These read the wrench between
  the hand and wrist_3 in the flange frame, like a UR wrist F/T sensor. They are
  nonzero at rest (tool weight), so tare against a no-contact reading.
  Access with `data.sensor("right_ft_force").data`.

### Exporting one MJCF file

`export_mjcf` writes any hand/props variant as a single MJCF file (see use case
3 for what the file contains and how to load it):

```bash
uv run python urgantry_sim/build_urgantry.py --hand sharpa --props --export scene.xml
```

```python
from urgantry_sim.build_urgantry import export_mjcf
export_mjcf("scene.xml", spawn_props=True, hand="sharpa")
```

Mesh paths are written relative to the exported file, so it loads from any
working directory but must stay where it was written (or be re-exported).

### Viewer

```python
import mujoco.viewer
from urgantry_sim.build_urgantry import apply_initial_view

with mujoco.viewer.launch_passive(model, data) as viewer:
    apply_initial_view(viewer)
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

---

## Use case 3: the MJCF scene files

The scene is assembled in Python (the arm and hand MJCFs are attached with the
`mjSpec` API), so these files are generated, not hand-written. They hold the
complete scene in one XML each and load with plain `mujoco`, with no need to
install or import `urgantry_sim`.

| file | hands | actuators | props |
|---|---|---|---|
| `scene_wuji.xml` | Wuji | 52 | none (bare scene) |
| `scene_sharpa.xml` | Sharpa Wave | 56 | none (bare scene) |

View one:

```bash
python -m mujoco.viewer --mjcf=scene_sharpa.xml
```

Load and start from the home pose:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("scene_sharpa.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
```

What is in them:

- Everything in the Scene section above: room, table, gantry, both arms with the
  flange adapter and hands, the `top1` camera, the wrist F/T sensors. Names and
  actuator order are the same as in the Python-built model.
- Keyframe `home`: the same qpos and ctrl that `set_initial_pose` sets (arms at
  home, hands open). Reset to it before stepping, otherwise every joint starts at
  zero. Ignore `left_home` / `right_home`: they come from the UR menagerie model
  and only pose one arm.
- Mesh and texture paths are relative to the repo root (`urgantry_sim/...`), so
  the files must stay in the repo root next to the `urgantry_sim/` folder.

What is not in them:

- The task logic. Reward, success and episode handling live in the Gymnasium
  env, not in the XML.
- The block and tray. Export a props variant with
  `--props` (see "Exporting one MJCF file") if you need them.

They are snapshots. After changing `build_urgantry.py`, regenerate them from the
repo root:

```bash
uv run python urgantry_sim/build_urgantry.py --hand wuji --export scene_wuji.xml
uv run python urgantry_sim/build_urgantry.py --hand sharpa --export scene_sharpa.xml
```

How closely they match the Python-built model (checked for both hands, with and
without props): identical sizes, names, solver options and parameters. The
largest difference is 1.6e-9, a rounded inertia on the fixed UR base. Sharpa
trajectories match to 1e-10 over 2 s of random commands. Wuji matches until a
~5e-8 solver-level difference appears and its finger contacts amplify it (the
same model perturbed by 1e-9 diverges similarly), reaching 1e-3 rad after 2 s.

The export writes two things explicitly that MuJoCo's `MjSpec.to_xml()` gets
wrong for this scene: every joint axis (otherwise `axis="0 0 1"` is dropped as a
default and the Wuji finger joints inherit the UR arm's `0 1 0`), and body
quaternions at full precision (otherwise rounded to ~6 digits).

---

## Third-party assets

- `urgantry_sim/universal_robots_ur5e/`: derived from the MuJoCo Menagerie UR5e
  model (BSD-3-Clause, see its `LICENSE`), re-tuned toward UR7e specs.
- `urgantry_sim/wuji_hand/`: MJCF and meshes from
  [wuji-technology/wuji-hand-description](https://github.com/wuji-technology/wuji-hand-description)
  (MIT, see its `LICENSE`).
- `urgantry_sim/sharpa_wave/`: Sharpa Wave MJCF and meshes from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/sharpa_wave)
  (Apache-2.0, see its `LICENSE`).
