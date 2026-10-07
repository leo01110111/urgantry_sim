"""Build the gantry UR7e tabletop scene with the MuJoCo spec (mjSpec) API.

One Vention column stands at the middle of the near edge of the table; two UR7e
arms hang upside down off its head, splayed 45 deg outward, each with a 5-finger
hand: Wuji (20 position-controlled joints) or Sharpa Wave (22), chosen with the
`hand` argument. The exposed tabletop in front of the
mount is the
"half" rectangle benchmarked in tabletop_tests/build_tabletop_half.py:
112 cm along the mount edge x 58.5 cm deep. Defines the reward functions used
in env.py.

NOTE FOR FUTURE AGENTS: the arm is a **UR7e**. The MJCF still lives at
`universal_robots_ur5e/ur5e.xml` (and the menagerie dir keeps its UR5e name)
because it started as the menagerie UR5e and is being re-tuned in place to UR7e
physics. Treat the robot as a UR7e everywhere; only the on-disk file paths keep
the legacy "ur5e" name.

Pure-XML <attach> of an <include>d robot doesn't work (include merges the robot's
bodies into the worldbody; attach needs the robot kept as a separate child model),
so we load the arm MJCF once per arm and attach a prefixed copy.

Coordinate frame: origin on the floor at the center of the exposed tabletop.
  +x = right along the mount edge, +y = away from the mount, +z = up. Meters.
  Exposed tabletop = x in [-HALF_LEN, HALF_LEN], y in [Y0, Y1] at z = BOARD_TOP.
  The column stands on its own strip of table behind Y0, so it never eats into
  the exposed rectangle.

Run directly to open the interactive viewer:
  uv run python build_urgantry.py [--hand wuji|sharpa] [--props]
"""

from pathlib import Path

import numpy as np
import mujoco

# Resolved relative to this file (not the process cwd) so the package works
# whether it's run in-place (`cd urgantry_sim && python test_viz.py`) or installed
# and imported from anywhere (`import urgantry_sim`).
ASSET_DIR = Path(__file__).resolve().parent

# UR7e arm MJCF. Path keeps the legacy "ur5e" name (see module docstring); the
# physics in that file is being changed to a UR7e.
UR7E_PATH = str(ASSET_DIR / "universal_robots_ur5e" / "ur5e.xml")

# Hand MJCF + meshes are verbatim copies: `wuji_hand/` from
# wuji-technology/wuji_hand_description, `sharpa_wave/` from mujoco_menagerie.
# Both share one palm frame convention, so they mount with the same HAND_ROLL:
# fingers grow along palm +z, the grasping side faces palm +x (measured: curling
# moves the fingertips toward +x), thumb on palm -y for the left hand, +y right.
HANDS = ("wuji", "sharpa")
DEFAULT_HAND = "wuji"
HAND_PATHS = {
    "wuji": {side: str(ASSET_DIR / "wuji_hand" / "mjcf" / f"{side}.xml")
             for side in ("left", "right")},
    "sharpa": {side: str(ASSET_DIR / "sharpa_wave" / f"{side}_hand.xml")
               for side in ("left", "right")},
}
# Palm body name after the Sharpa side prefix is stripped (see _strip_side_prefix).
HAND_PALM = {"wuji": "palm_link", "sharpa": "hand_C_MC"}

# ---------------------------------------------------------------- measurements
TABLE_H = 0.76          # aluminum frame top height
BOARD_T = 0.015         # black board thickness (on top of the frame)
PLATE_T = 0.006         # blue Vention mounting plate thickness (column footplate)
BOARD_TOP = TABLE_H + BOARD_T          # 0.775

# --------------------------------------------------------------- exposed table
EXPOSED_LEN = 1.12      # along x, mount centered on this edge
EXPOSED_DEPTH = 0.585   # along y, in front of the mount

HALF_LEN = EXPOSED_LEN / 2             # 0.56
Y0 = -EXPOSED_DEPTH / 2                # near (mount) edge of the exposed area
Y1 = EXPOSED_DEPTH / 2

# --------------------------------------------------------------- gantry column
COL_HALF = 0.045                       # 9x9 cm extrusion
COL_FOOT_HALF = 0.09                   # blue footplate under the column
COL_STRIP = 2 * COL_FOOT_HALF          # table depth taken up behind the exposed area
COL_TOP_Z = BOARD_TOP + 0.78           # top of the column above the board
COL_Y = Y0 - COL_FOOT_HALF

TABLE_Y0 = Y0 - COL_STRIP              # near edge of the *whole* table
TABLE_CY = (TABLE_Y0 + Y1) / 2
HALF_W = (EXPOSED_DEPTH + COL_STRIP) / 2

HEAD_HALF_X = 0.24
HEAD_HALF_Y = 0.10
HEAD_T = 0.03
HEAD_Z = COL_TOP_Z + HEAD_T / 2

# --------------------------------------------------------------- arm mounting
# Both arms hang upside down (base flange up, arm pointing down) and are rolled
# 45 deg outward about +y, so the left arm leans toward -x and the right toward +x.
ARM_TILT = np.deg2rad(45.0)
ARM_MOUNT_X = 0.33756 / 2             # mount center offset from the column axis; 337.56 mm between mounts on the Vention head
ARM_Z = COL_TOP_Z - 0.06               # base flange height

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
# Home: hands over the table, palms facing the board and backs of the hands
# toward the overhead camera, fingers pointing away from the mount -- the way
# hands appear in egocentric video. Solved by IK for the flange at
# (+-0.372, 0.092, 1.032) with the grasp side (palm +x) -> world -z and
# palm +z (fingers) -> world +y.
LEFT_HOME_POSE = [ 1.7002, -1.5626,  1.6061, -0.0435,  1.7002,  0.7908]
RIGHT_HOME_POSE = [-1.7002, -1.5789, -1.6061, -3.0981, -1.7002, -0.7908]

# Hand closure for set_hand: 0 = open, 1 = fist. All-zero ctrl is the flat open
# hand for both models.
# Wuji: 5 fingers x 4 joints, ctrl in radians. On fingers 2-5 joint1 flexes at
# the knuckle and joint2 spreads (measured); joint3..4 curl. A fist puts
# WUJI_CURL_CLOSED on every joint.
# Sharpa: 22 joints named by anatomy (thumb CMC/MCP/IP, finger MCP_FE/MCP_AA/
# PIP/DIP, pinky CMC). Flexion is negative on the right hand and positive on the
# left, so a fist drives each flexion joint SHARPA_CLOSED_FRAC of the way to its
# flexion limit and leaves abduction (AA) and pinky CMC at zero.
HAND_OPEN = 0.0
HAND_CLOSED = 1.0
WUJI_CURL_CLOSED = 1.2
SHARPA_CLOSED_FRAC = 0.8

# A cylindrical adapter sits on the tool flange and the palm sits on the adapter,
# rolled about the flange axis by HAND_ROLL. The right hand is rolled 180 deg,
# putting its grasping side (palm +x) on the opposite side from the left's.
# Each palm mesh reaches below its origin (palm -z): Wuji's two-hole wrist lip by
# 9.1 mm, Sharpa's wrist base by 4.9 mm (measured from the palm mesh vertices), so
# the palm origin sits that far past the adapter's top face. The UR mesh's
# flange face is 1.06 mm short of attachment_site (measured from wrist_3 mesh
# vertices), so the adapter starts at the mesh face, not at the site.
FLANGE_FACE_Z = -0.00106
ADAPTER_T = 0.0217
ADAPTER_R = 0.0315                     # ISO 9409-1-50 flange, 63 mm
ADAPTER_MASS = 0.09
PALM_LIP_DEPTH = {"wuji": 0.0091, "sharpa": 0.0049}
ADAPTER_RGBA = [0.75, 0.76, 0.78, 1]
HAND_ROLL = {
    "left": [1.0, 0.0, 0.0, 0.0],
    "right": [0.0, 0.0, 0.0, 1.0],
}

# --------------------------------------------------------------- overhead camera
# Mounted on the front face of the gantry head, centered between the two arms:
# an egocentric view down the workspace, past the arms.
CAM_POS = (0.0, COL_Y + HEAD_HALF_Y + 0.02, HEAD_Z - HEAD_T / 2 - 0.03)
CAM_TARGET = (0.0, Y0 + 0.5 * EXPOSED_DEPTH, BOARD_TOP)
# Wide-angle module, not the D435 that stood off the table: at this range a 42
# deg FOV sees only the middle ~60 cm of the 112 cm exposed edge.
CAM_FOVY = 80.0

# --------------------------------------------------------------- pick task
# A graspable block sits on the board; the task is to lift it. Success = block
# lifted LIFT_SUCCESS_H above its rest height.
BLOCK_HALF = 0.025                                   # 5 cm cube
BLOCK_REST_Z = BOARD_TOP + BLOCK_HALF
BLOCK_INIT_POS = (min(0.32, HALF_LEN - 0.12), Y0 + 0.22, BLOCK_REST_Z)
BLOCK_RGBA = [0.15, 0.75, 0.20, 1]
LIFT_SUCCESS_H = 0.05                                # meters above rest to count as a pick

# --------------------------------------------------------------- cardboard tray
# Open-top box (corrugated cardboard tray) sitting on the board, mirrored across
# the column from the block.
BOX_OUTER = 0.16                                     # outer footprint, square
BOX_WALL_T = 0.006
BOX_WALL_H = 0.035
BOX_FLOOR_T = 0.004
BOX_INIT_POS = (-min(0.32, HALF_LEN - 0.12), Y0 + 0.22, BOARD_TOP + BOX_FLOOR_T / 2)
BOX_RGBA = [0.72, 0.53, 0.34, 1]

# colors
COL_ALU = [0.62, 0.64, 0.66, 1]
COL_PLATE = [0.20, 0.42, 0.78, 1]
COL_LEG = [0.12, 0.13, 0.15, 1]


def _upside_down_quat(tilt: float) -> list[float]:
    """Quat [w,x,y,z] for Ry(tilt) * Rx(180deg): flip the base over (its +z, which
    the arm grows along, now points down) then lean it by `tilt` about +y, so the
    arm hangs and splays toward -x for tilt>0 / +x for tilt<0."""
    flip = np.array([np.cos(np.pi / 2), np.sin(np.pi / 2), 0.0, 0.0])   # Rx(pi)
    lean = np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0])     # Ry(tilt)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, lean, flip)
    return quat.tolist()


LEFT_BASE_QUAT = _upside_down_quat(ARM_TILT)
RIGHT_BASE_QUAT = _upside_down_quat(-ARM_TILT)


def _lookat_quat(cam_pos, target):
    """Quaternion orienting a MuJoCo camera (looks down -z) from cam_pos at target."""
    cam_pos, target = np.asarray(cam_pos, float), np.asarray(target, float)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # camera axes as columns: x=right, y=up, z=-forward  (row-major 3x3)
    mat = np.array([[right[0], up[0], -forward[0]],
                    [right[1], up[1], -forward[1]],
                    [right[2], up[2], -forward[2]]]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _strip_side_prefix(spec: mujoco.MjSpec, side: str) -> None:
    """Drop the Sharpa MJCF's own 'left_'/'right_' name prefix (and the
    references to those names), so the attach prefixes give
    'left_hand_index_MCP_FE' rather than 'left_hand_left_index_MCP_FE'."""
    prefix = f"{side}_"
    strip = lambda name: name[len(prefix):] if name.startswith(prefix) else name
    for elements in (spec.bodies, spec.joints, spec.geoms, spec.sites, spec.actuators):
        for el in elements:
            el.name = strip(el.name)
    for act in spec.actuators:
        act.target = strip(act.target)
    for ex in spec.excludes:
        ex.bodyname1 = strip(ex.bodyname1)
        ex.bodyname2 = strip(ex.bodyname2)


def _arm_with_hand(side: str, hand: str) -> mujoco.MjSpec:
    """Load a UR7e, add the flange adapter body (frame = flange frame, i.e. the
    attachment site), and bolt the matching (left/right) hand onto the adapter.
    The hand's position actuators and its contact exclusions come along with the
    attach, prefixed 'hand_' (final actuator names e.g. 'left_hand_finger1_joint1'
    for Wuji, 'left_hand_index_MCP_FE' for Sharpa)."""
    arm = mujoco.MjSpec.from_file(UR7E_PATH)
    flange = arm.site("attachment_site")
    adapter = arm.body("wrist_3_link").add_body(name="flange_adapter",
                                                pos=flange.pos, quat=flange.quat)
    adapter.add_geom(name="flange_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     size=[ADAPTER_R, ADAPTER_T / 2, 0],
                     pos=[0, 0, FLANGE_FACE_Z + ADAPTER_T / 2],
                     mass=ADAPTER_MASS, rgba=ADAPTER_RGBA)
    hand_spec = mujoco.MjSpec.from_file(HAND_PATHS[hand][side])
    if hand == "sharpa":
        _strip_side_prefix(hand_spec, side)
    palm = hand_spec.body(HAND_PALM[hand])
    palm.quat = HAND_ROLL[side]
    mount_z = FLANGE_FACE_Z + ADAPTER_T + PALM_LIP_DEPTH[hand]
    adapter.add_site(name="hand_mount", pos=[0, 0, mount_z]).attach_body(palm, "hand_", "")
    return arm


def _materials(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.8, 0.8, 0.78], rgb2=[0.8, 0.8, 0.78], width=512, height=3072)

    spec.add_texture(name="carpet", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(ASSET_DIR / "universal_robots_ur5e" / "assets" / "carpet.png"))
    grid_mat = spec.add_material(name="grid", rgba=[1, 1, 1, 1], reflectance=0.0,
                                 texrepeat=[10, 10], texuniform=True)
    grid_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "carpet"

    spec.add_material(name="wallpaper", rgba=[0.96, 0.96, 0.94, 1], reflectance=0.05)

    spec.add_texture(name="board_plastic", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     rgb1=[0, 0, 0], rgb2=[0, 0, 0], markrgb=[0.03, 0.03, 0.03], random=0.02,
                     width=512, height=512)
    board_mat = spec.add_material(name="board_plastic", texrepeat=[4, 4], texuniform=True,
                                  specular=0.9, shininess=0.9, reflectance=0.2)
    board_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "board_plastic"

    spec.add_texture(name="cardboard", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     rgb1=BOX_RGBA[:3], rgb2=BOX_RGBA[:3], markrgb=[0.55, 0.38, 0.22],
                     random=0.08, width=256, height=256)
    cardboard_mat = spec.add_material(name="cardboard", texrepeat=[2, 2], texuniform=True,
                                      specular=0.05, shininess=0.05, reflectance=0.0)
    cardboard_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cardboard"


def _add_props(wb: mujoco.MjsBody) -> None:
    # --- graspable block (free joint) for the pick task ---------------------
    block = wb.add_body(name="block", pos=list(BLOCK_INIT_POS))
    block.add_freejoint(name="block_joint")
    block.add_geom(name="block", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[BLOCK_HALF] * 3, rgba=BLOCK_RGBA,
                   mass=0.05, friction=[1.0, 0.01, 0.001])

    # --- open-top cardboard tray (free joint) -------------------------------
    box_half = BOX_OUTER / 2
    box_in_half = box_half - BOX_WALL_T
    box = wb.add_body(name="cardboard_box", pos=list(BOX_INIT_POS))
    box.add_freejoint(name="cardboard_box_joint")
    box.add_geom(name="cardboard_box_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[box_half, box_half, BOX_FLOOR_T / 2], material="cardboard", mass=0.03)
    wall_z = BOX_FLOOR_T / 2 + BOX_WALL_H / 2
    for name, size, pos in (
        ("neg_x", [BOX_WALL_T / 2, box_half, BOX_WALL_H / 2], [-box_in_half, 0, wall_z]),
        ("pos_x", [BOX_WALL_T / 2, box_half, BOX_WALL_H / 2], [box_in_half, 0, wall_z]),
        ("neg_y", [box_in_half, BOX_WALL_T / 2, BOX_WALL_H / 2], [0, -box_in_half, wall_z]),
        ("pos_y", [box_in_half, BOX_WALL_T / 2, BOX_WALL_H / 2], [0, box_in_half, wall_z]),
    ):
        box.add_geom(name=f"cardboard_box_wall_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=size, pos=pos, material="cardboard", mass=0.01)



def build_spec(spawn_props: bool = False, hand: str = DEFAULT_HAND) -> mujoco.MjSpec:
    """Floor, room, table, gantry column, two hanging UR7e arms with `hand`
    ('wuji' or 'sharpa') hands, the overhead camera, and (when spawn_props) the
    graspable block and cardboard tray."""
    if hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
    spec = mujoco.MjSpec()
    spec.compiler.autolimits = True
    # Offscreen buffer sized for the OV9782 wrist cameras (1280x800); covers the
    # smaller top-camera / policy renders too.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    # Pin the extent to the working area, not the room: znear is znear_ratio *
    # extent, and a 6 m room pushes it past 12 cm, clipping the wrist cameras.
    spec.stat.extent = 1.5

    _materials(spec)

    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.35, 0.35, 0.35])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="grid")

    # --- room walls: plain white wallpaper enclosing the floor --------------
    ROOM_HALF = 3.0     # matches floor plane half-extent
    WALL_H = 2.4         # typical room ceiling height
    WALL_T = 0.05
    for name, pos, size in (
        ("wall_neg_x", [-ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_pos_x", [ROOM_HALF, 0, WALL_H / 2], [WALL_T / 2, ROOM_HALF, WALL_H / 2]),
        ("wall_neg_y", [0, -ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
        ("wall_pos_y", [0, ROOM_HALF, WALL_H / 2], [ROOM_HALF, WALL_T / 2, WALL_H / 2]),
    ):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
                    material="wallpaper")

    # --- table: aluminum frame top + 4 legs ---------------------------------
    wb.add_geom(name="alu_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[HALF_LEN, HALF_W, 0.02], pos=[0, TABLE_CY, TABLE_H - 0.02], rgba=COL_ALU)
    leg_h = TABLE_H - 0.04
    leg_inset = 0.05
    for sx in (-1, 1):
        for sy in (-1, 1):
            wb.add_geom(name=f"leg_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.02, 0.02, leg_h / 2],
                        pos=[sx * (HALF_LEN - leg_inset), TABLE_CY + sy * (HALF_W - leg_inset),
                             leg_h / 2],
                        rgba=COL_LEG)

    # --- black board on top, cut around the column footplate ----------------
    board_z = TABLE_H + BOARD_T / 2
    nx0, nx1 = -COL_FOOT_HALF, COL_FOOT_HALF
    ny0, ny1 = COL_Y - COL_FOOT_HALF, COL_Y + COL_FOOT_HALF
    for name, (x0, x1), (y0, y1) in (
        ("neg_x", (-HALF_LEN, nx0), (TABLE_Y0, Y1)),
        ("pos_x", (nx1, HALF_LEN), (TABLE_Y0, Y1)),
        ("mid_front", (nx0, nx1), (TABLE_Y0, ny0)),
        ("mid_back", (nx0, nx1), (ny1, Y1)),
    ):
        if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
            continue
        wb.add_geom(name=f"board_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[(x1 - x0) / 2, (y1 - y0) / 2, BOARD_T / 2],
                    pos=[(x0 + x1) / 2, (y0 + y1) / 2, board_z], material="board_plastic")

    # --- gantry: blue footplate + vertical extrusion + head plate -----------
    wb.add_geom(name="column_foot", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COL_FOOT_HALF, COL_FOOT_HALF, PLATE_T / 2],
                pos=[0, COL_Y, TABLE_H + PLATE_T / 2], rgba=COL_PLATE)
    col_z0 = TABLE_H + PLATE_T
    wb.add_geom(name="column", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[COL_HALF, COL_HALF, (COL_TOP_Z - col_z0) / 2],
                pos=[0, COL_Y, (col_z0 + COL_TOP_Z) / 2], rgba=COL_PLATE)
    wb.add_geom(name="gantry_head", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[HEAD_HALF_X, HEAD_HALF_Y, HEAD_T / 2],
                pos=[0, COL_Y, HEAD_Z], rgba=COL_PLATE)

    # --- arms: hung upside down off the head, splayed 45 deg outward --------
    # (Unlike a frame's euler, a body quat DOES propagate through attach, so we
    # orient each base via its mount body.)
    for side, sx, quat in (("left", -1, LEFT_BASE_QUAT), ("right", 1, RIGHT_BASE_QUAT)):
        mount = wb.add_body(name=f"{side}_robot_mount",
                            pos=[sx * ARM_MOUNT_X, COL_Y, ARM_Z], quat=quat)
        mount.add_geom(name=f"plate_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.075, 0.075, PLATE_T / 2], pos=[0, 0, -PLATE_T / 2],
                       rgba=COL_PLATE, contype=0, conaffinity=0)
        mount.add_frame().attach_body(_arm_with_hand(side, hand).body("base"), f"{side}_", "")

    # Wrist F/T: force+torque sensors at each flange (attachment_site) report the wrench
    # transmitted between the adapter+hand subtree and wrist_3, in the flange
    # frame -- the same quantity a UR wrist F/T sensor measures. Reading is
    # nonzero at rest (tool weight), so consumers should tare against a
    # no-contact reference pose.
    for side in ("left", "right"):
        spec.body(f"{side}_flange_adapter").add_site(name=f"{side}_ft_site")
        spec.add_sensor(name=f"{side}_ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_ft_site")
        spec.add_sensor(name=f"{side}_ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{side}_ft_site")

    if spawn_props:
        _add_props(wb)

    # --- camera (Intel RealSense D435) --------------------------------------
    quat = _lookat_quat(CAM_POS, CAM_TARGET)
    cam_body = wb.add_body(name="top1_body", pos=list(CAM_POS), quat=list(quat))
    # D435 housing: 90 x 25 x 25 mm; camera looks down its local -z.
    cam_body.add_geom(name="top1_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.045, 0.0125, 0.0125], rgba=[1, 1, 1, 1],
                      contype=0, conaffinity=0)
    cam_body.add_camera(name="top1", fovy=CAM_FOVY)

    return spec


def build_model(spawn_props: bool = False, hand: str = DEFAULT_HAND) -> mujoco.MjModel:
    return build_spec(spawn_props, hand).compile()


# ======================================================================== #
#  HARD-CODED INITIAL STATE — edit these blocks to change what the viewer   #
#  opens at. To recapture them after rearranging the scene, run test_viz.py #
#  and press 's' (see capture_state); paste its output back here.           #
# ======================================================================== #

# Robot pose is set per-joint by name (see set_initial_pose) from LEFT_HOME_POSE /
# RIGHT_HOME_POSE, so it survives layout changes (added hand/object joints). Edit
# those / BLOCK_INIT_POS above to change the starting state.

# Viewport: the free camera's orbit angle / zoom / look-at target.
INITIAL_VIEW = {
    "azimuth": 90.0,
    "elevation": -12.0,
    "distance": 2.4,
    "lookat": [0.0, Y0 + 0.3 * EXPOSED_DEPTH, BOARD_TOP + 0.30],
}


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms at the home pose with hands open, command the actuators to
    hold it (so nothing sags under gravity), and place the props at their rest
    poses. Set by joint/body name so it survives layout changes."""
    for prefix, home_pose in (("left_", LEFT_HOME_POSE), ("right_", RIGHT_HOME_POSE)):
        for joint, angle in zip(ARM_JOINTS, home_pose):
            data.qpos[model.joint(f"{prefix}{joint}_joint").qposadr[0]] = angle
            data.ctrl[model.actuator(f"{prefix}{joint}").id] = angle
        set_hand(model, data, prefix[:-1], HAND_OPEN)

    # free joint qpos is [x, y, z, qw, qx, qy, qz]
    if has_props(model):
        for joint, pos in (("block_joint", BLOCK_INIT_POS), ("cardboard_box_joint", BOX_INIT_POS)):
            adr = model.joint(joint).qposadr[0]
            data.qpos[adr:adr + 7] = [*pos, 1, 0, 0, 0]


def hand_type(model: mujoco.MjModel) -> str:
    """'wuji' or 'sharpa', read off the compiled model's actuator names."""
    wuji = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hand_finger1_joint1")
    return "wuji" if wuji >= 0 else "sharpa"


def hand_actuators(model: mujoco.MjModel, side: str) -> list[str]:
    """Names of one hand's position actuators (20 Wuji, 22 Sharpa), in actuator
    order."""
    names = [model.actuator(i).name for i in range(model.nu)]
    return [n for n in names if n.startswith(f"{side}_hand_")]


def hand_pose(model: mujoco.MjModel, side: str, closure: float) -> np.ndarray:
    """Ctrl targets for one hand, in hand_actuators order: closure 0 (HAND_OPEN)
    is the flat open hand, 1 (HAND_CLOSED) a fist. Clipped to each ctrlrange."""
    kind = hand_type(model)
    out = []
    for name in hand_actuators(model, side):
        lo, hi = model.actuator(name).ctrlrange
        if kind == "wuji":
            target = closure * WUJI_CURL_CLOSED
        elif name.endswith(("_AA", "pinky_CMC")):
            target = 0.0
        else:
            flex_limit = lo if side == "right" else hi
            target = closure * SHARPA_CLOSED_FRAC * flex_limit
        out.append(float(np.clip(target, lo, hi)))
    return np.array(out)


def set_hand(model: mujoco.MjModel, data: mujoco.MjData, side: str, closure: float) -> None:
    """Command one hand to hand_pose(closure): 0 open, 1 fist."""
    ids = [model.actuator(n).id for n in hand_actuators(model, side)]
    data.ctrl[ids] = hand_pose(model, side, closure)


def has_props(model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block") >= 0


def block_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Current height (z, meters) of the block's center."""
    return float(data.body("block").xpos[2])


def pick_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """True once the block has been lifted LIFT_SUCCESS_H above its rest height."""
    return block_height(model, data) > BLOCK_REST_Z + LIFT_SUCCESS_H


def apply_initial_view(viewer) -> None:
    """Point the viewer's free camera at INITIAL_VIEW. Call right after
    launch_passive(), then viewer.sync(). Edit INITIAL_VIEW above to change it."""
    cam = viewer.cam
    cam.azimuth = INITIAL_VIEW["azimuth"]
    cam.elevation = INITIAL_VIEW["elevation"]
    cam.distance = INITIAL_VIEW["distance"]
    cam.lookat[:] = INITIAL_VIEW["lookat"]


def capture_state(data: mujoco.MjData, viewer) -> None:
    """Print the current viewport as a paste-ready INITIAL_VIEW block, plus the
    current arm joint angles (for LEFT_HOME_POSE), so you can hard-code the state
    you've navigated to."""
    cam = viewer.cam
    print("\n# --- paste INITIAL_VIEW into build_urgantry.py ---")
    print("INITIAL_VIEW = {")
    print(f'    "azimuth": {cam.azimuth:.3f},')
    print(f'    "elevation": {cam.elevation:.3f},')
    print(f'    "distance": {cam.distance:.4f},')
    print(f'    "lookat": [{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}],')
    print("}")
    print("# current left-arm joints:",
          [round(float(data.qpos[i]), 4) for i in range(6)], "\n")


def build_scene(spawn_props: bool = False,
                hand: str = DEFAULT_HAND) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the scene and return model + data initialized to the home pose, with
    the position actuators commanded to hold it and forward kinematics evaluated."""
    model = build_model(spawn_props, hand)
    data = mujoco.MjData(model)
    set_initial_pose(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def parse_scene_args(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hand", choices=HANDS, default=DEFAULT_HAND)
    p.add_argument("--props", action="store_true", help="spawn the block and tray")
    return p.parse_args(argv)


def main() -> None:
    import time
    import mujoco.viewer

    args = parse_scene_args()
    model, data = build_scene(spawn_props=args.props, hand=args.hand)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_initial_view(viewer)
        viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
