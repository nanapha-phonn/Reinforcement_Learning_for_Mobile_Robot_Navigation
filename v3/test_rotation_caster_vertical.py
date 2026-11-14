import math
import csv
import pybullet as p
import pybullet_data


# ============================================================
# V3.17.5 CASTER VERTICAL / CONTACT DIAGNOSTIC
# ============================================================

WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.04
WHEEL_BASE = 0.30

CHASSIS_LENGTH = 0.40
CHASSIS_WIDTH = 0.24
CHASSIS_HEIGHT = 0.10

CHASSIS_MASS = 5.0
WHEEL_MASS = 0.5

CASTER_FORK_MASS = 0.05
CASTER_WHEEL_MASS = 0.15

CASTER_X = -0.15
CASTER_TRAIL = 0.05

CASTER_WHEEL_RADIUS = 0.04
CASTER_WHEEL_WIDTH = 0.025

MOTOR_FORCE = 20.0

TIME_STEP = 1.0 / 240.0
GRAVITY = -9.81

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1
CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

LEFT_CMD = -4.0
RIGHT_CMD = 4.0

ROTATION_TIME = 3.0
SETTLE_TIME = 2.0

SAMPLE_INTERVAL = 0.1


# ============================================================
# HELPERS
# ============================================================

def quaternion_to_matrix(q):
    return p.getMatrixFromQuaternion(q)


def local_axis_world(orientation, local_axis):
    matrix = quaternion_to_matrix(orientation)

    x = (
        matrix[0] * local_axis[0]
        + matrix[3] * local_axis[1]
        + matrix[6] * local_axis[2]
    )

    y = (
        matrix[1] * local_axis[0]
        + matrix[4] * local_axis[1]
        + matrix[7] * local_axis[2]
    )

    z = (
        matrix[2] * local_axis[0]
        + matrix[5] * local_axis[1]
        + matrix[8] * local_axis[2]
    )

    return x, y, z


def vec_length(v):
    return math.sqrt(sum(x * x for x in v))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def angle_deg(a, b):
    la = vec_length(a)
    lb = vec_length(b)

    if la < 1e-12 or lb < 1e-12:
        return float("nan")

    value = dot(a, b) / (la * lb)
    value = max(-1.0, min(1.0, value))

    return math.degrees(math.acos(value))


# ============================================================
# CREATE ROBOT
# ============================================================

def create_robot():
    collision_shapes = []

    # Chassis
    chassis_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            CHASSIS_LENGTH / 2,
            CHASSIS_WIDTH / 2,
            CHASSIS_HEIGHT / 2
        ]
    )

    chassis_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            CHASSIS_LENGTH / 2,
            CHASSIS_WIDTH / 2,
            CHASSIS_HEIGHT / 2
        ]
    )

    # Drive wheels
    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # Caster fork
    fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.015, 0.015, 0.05]
    )

    fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.015, 0.015, 0.05]
    )

    # Caster wheel
    caster_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        height=CASTER_WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    caster_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        length=CASTER_WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # Working PyBullet hierarchy established in V3.15+
    link_masses = [
        WHEEL_MASS,
        WHEEL_MASS,
        CASTER_FORK_MASS,
        CASTER_WHEEL_MASS
    ]

    link_collision_shapes = [
        wheel_collision,
        wheel_collision,
        fork_collision,
        caster_collision
    ]

    link_visual_shapes = [
        wheel_visual,
        wheel_visual,
        fork_visual,
        caster_visual
    ]

    link_positions = [
        [0.0,  WHEEL_BASE / 2, -0.05],
        [0.0, -WHEEL_BASE / 2, -0.05],

        [CASTER_X, 0.0, -0.05],

        [-CASTER_TRAIL, 0.0, -0.04]
    ]

    link_orientations = [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]

    link_inertial_positions = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    link_inertial_orientations = [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]

    # IMPORTANT:
    # This is the empirically validated hierarchy for this
    # installed PyBullet build.
    link_parent_indices = [0, 0, 0, 3]

    link_joint_types = [
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE
    ]

    link_joint_axes = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]

    robot_id = p.createMultiBody(
        baseMass=CHASSIS_MASS,
        baseCollisionShapeIndex=chassis_collision,
        baseVisualShapeIndex=chassis_visual,
        basePosition=[0, 0, 0.13],
        baseOrientation=[0, 0, 0, 1],

        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shapes,
        linkVisualShapeIndices=link_visual_shapes,

        linkPositions=link_positions,
        linkOrientations=link_orientations,

        linkInertialFramePositions=link_inertial_positions,
        linkInertialFrameOrientations=link_inertial_orientations,

        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    p.changeDynamics(
        robot_id,
        LEFT_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    p.changeDynamics(
        robot_id,
        RIGHT_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    p.changeDynamics(
        robot_id,
        2,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    p.changeDynamics(
        robot_id,
        3,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # Disable self collisions
    for i in range(-1, 4):
        for j in range(i + 1, 4):
            try:
                p.setCollisionFilterPair(
                    robot_id,
                    robot_id,
                    i,
                    j,
                    enableCollision=0
                )
            except Exception:
                pass

    return robot_id


# ============================================================
# CONTACT INFORMATION
# ============================================================

def get_caster_contact(robot_id):
    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=CASTER_WHEEL_JOINT
    )

    total_normal_force = 0.0

    contact_points = []

    for c in contacts:
        normal_force = c[9]
        total_normal_force += normal_force

        # Position on caster wheel
        pos_a = c[5]

        # Position on plane
        pos_b = c[6]

        contact_points.append(
            (pos_a, pos_b, normal_force)
        )

    return contacts, total_normal_force, contact_points


# ============================================================
# SETUP
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, GRAVITY)
p.setTimeStep(TIME_STEP)

plane_id = p.loadURDF("plane.urdf")

robot_id = create_robot()

# No caster motors
p.setJointMotorControl2(
    robot_id,
    CASTER_SWIVEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0
)

p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0
)

print("=" * 80)
print("V3.17.5 CASTER VERTICAL / CONTACT DIAGNOSTIC")
print("=" * 80)

print()
print("Fixed parameters:")
print(f"CASTER_X     = {CASTER_X:.3f} m")
print(f"CASTER_TRAIL = {CASTER_TRAIL:.3f} m")
print()

print(f"Left wheel command  : {LEFT_CMD:.3f} rad/s")
print(f"Right wheel command : {RIGHT_CMD:.3f} rad/s")
print(f"Motor force         : {MOTOR_FORCE:.3f}")
print(f"Rotation time       : {ROTATION_TIME:.3f} s")

theoretical_yaw_rate = (
    WHEEL_RADIUS / WHEEL_BASE
) * (RIGHT_CMD - LEFT_CMD)

print(
    f"Theoretical yaw rate: "
    f"{theoretical_yaw_rate:.6f} rad/s"
)

# ============================================================
# SETTLE
# ============================================================

for _ in range(int(SETTLE_TIME / TIME_STEP)):
    p.stepSimulation()


# ============================================================
# ROTATION
# ============================================================

print()
print("=" * 80)
print("ROTATION START")
print("=" * 80)

history = []

steps = int(ROTATION_TIME / TIME_STEP)

sample_every = max(
    1,
    int(SAMPLE_INTERVAL / TIME_STEP)
)

for step in range(steps + 1):

    p.setJointMotorControl2(
        robot_id,
        LEFT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=LEFT_CMD,
        force=MOTOR_FORCE
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=RIGHT_CMD,
        force=MOTOR_FORCE
    )

    # Keep caster passive
    p.setJointMotorControl2(
        robot_id,
        CASTER_SWIVEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    p.setJointMotorControl2(
        robot_id,
        CASTER_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    p.stepSimulation()

    if step % sample_every != 0:
        continue

    # --------------------------------------------------------
    # Base state
    # --------------------------------------------------------

    base_pos, base_orn = p.getBasePositionAndOrientation(
        robot_id
    )

    base_euler = p.getEulerFromQuaternion(base_orn)

    roll = math.degrees(base_euler[0])
    pitch = math.degrees(base_euler[1])
    yaw = math.degrees(base_euler[2])

    # --------------------------------------------------------
    # Caster fork state
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True
    )

    fork_pos = fork_state[4]
    fork_orn = fork_state[5]

    fork_z = fork_pos[2]

    # --------------------------------------------------------
    # Caster wheel state
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_pos = wheel_state[4]
    wheel_orn = wheel_state[5]

    wheel_x = wheel_pos[0]
    wheel_y = wheel_pos[1]
    wheel_z = wheel_pos[2]

    # --------------------------------------------------------
    # Wheel world axes
    # --------------------------------------------------------

    # Wheel local Y is the axle direction.
    axle_axis = local_axis_world(
        wheel_orn,
        [0, 1, 0]
    )

    # Wheel local Z is approximately the wheel radial direction.
    wheel_z_axis = local_axis_world(
        wheel_orn,
        [0, 0, 1]
    )

    # Wheel local X
    wheel_x_axis = local_axis_world(
        wheel_orn,
        [1, 0, 0]
    )

    # --------------------------------------------------------
    # Wheel orientation relative to vertical/horizontal
    # --------------------------------------------------------

    vertical = [0, 0, 1]
    ground_normal = [0, 0, 1]

    axle_vertical_angle = angle_deg(
        axle_axis,
        vertical
    )

    wheel_radial_vertical_angle = angle_deg(
        wheel_z_axis,
        vertical
    )

    # A correctly oriented wheel should have its axle
    # approximately horizontal.
    axle_from_horizontal = abs(
        90.0 - axle_vertical_angle
    )

    # --------------------------------------------------------
    # Ground geometry
    # --------------------------------------------------------

    ground_gap = wheel_z - CASTER_WHEEL_RADIUS

    contacts, normal_force, contact_points = (
        get_caster_contact(robot_id)
    )

    contact = len(contacts) > 0

    # --------------------------------------------------------
    # Contact point
    # --------------------------------------------------------

    if contact_points:
        cp_world = contact_points[0][0]
        plane_cp = contact_points[0][1]
    else:
        cp_world = [float("nan")] * 3
        plane_cp = [float("nan")] * 3

    # --------------------------------------------------------
    # Caster-to-fork relative vertical geometry
    # --------------------------------------------------------

    relative_caster_z = wheel_z - fork_z

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    history.append({
        "time": step * TIME_STEP,

        "base_x": base_pos[0],
        "base_y": base_pos[1],
        "base_z": base_pos[2],

        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,

        "fork_x": fork_pos[0],
        "fork_y": fork_pos[1],
        "fork_z": fork_z,

        "wheel_x": wheel_x,
        "wheel_y": wheel_y,
        "wheel_z": wheel_z,

        "relative_caster_z": relative_caster_z,

        "ground_gap": ground_gap,

        "axle_vertical_angle_deg":
            axle_vertical_angle,

        "axle_from_horizontal_deg":
            axle_from_horizontal,

        "wheel_radial_vertical_angle_deg":
            wheel_radial_vertical_angle,

        "axle_x": axle_axis[0],
        "axle_y": axle_axis[1],
        "axle_z": axle_axis[2],

        "contact": int(contact),
        "contact_count": len(contacts),

        "normal_force": normal_force,

        "contact_x":
            cp_world[0],

        "contact_y":
            cp_world[1],

        "contact_z":
            cp_world[2],

        "plane_contact_x":
            plane_cp[0],

        "plane_contact_y":
            plane_cp[1],

        "plane_contact_z":
            plane_cp[2]
    })


# ============================================================
# ANALYSIS
# ============================================================

print()
print("=" * 80)
print("V3.17.5 VERTICAL / CONTACT HISTORY")
print("=" * 80)

print(
    "Time | BaseZ | ForkZ | WheelZ | Gap | "
    "Pitch | Roll | AxleTilt | Force | Contact"
)

print("-" * 100)

for row in history:
    print(
        f"{row['time']:4.2f} | "
        f"{row['base_z']:6.4f} | "
        f"{row['fork_z']:6.4f} | "
        f"{row['wheel_z']:6.4f} | "
        f"{row['ground_gap']:6.4f} | "
        f"{row['pitch_deg']:7.3f} | "
        f"{row['roll_deg']:7.3f} | "
        f"{row['axle_from_horizontal_deg']:8.3f} | "
        f"{row['normal_force']:7.3f} | "
        f"{row['contact']}"
    )


# ============================================================
# SUMMARY
# ============================================================

base_z_values = [r["base_z"] for r in history]
fork_z_values = [r["fork_z"] for r in history]
wheel_z_values = [r["wheel_z"] for r in history]

gap_values = [r["ground_gap"] for r in history]

pitch_values = [abs(r["pitch_deg"]) for r in history]
roll_values = [abs(r["roll_deg"]) for r in history]

axle_tilt_values = [
    r["axle_from_horizontal_deg"]
    for r in history
]

force_values = [
    r["normal_force"]
    for r in history
]

contact_values = [
    r["contact"]
    for r in history
]

relative_z_values = [
    r["relative_caster_z"]
    for r in history
]

print()
print("=" * 80)
print("V3.17.5 SUMMARY")
print("=" * 80)

print()
print("VERTICAL MOTION")
print("-" * 80)

print(
    f"Base Z minimum          : "
    f"{min(base_z_values):.6f} m"
)

print(
    f"Base Z maximum          : "
    f"{max(base_z_values):.6f} m"
)

print(
    f"Base Z variation        : "
    f"{max(base_z_values) - min(base_z_values):.6f} m"
)

print(
    f"Fork Z minimum          : "
    f"{min(fork_z_values):.6f} m"
)

print(
    f"Fork Z maximum          : "
    f"{max(fork_z_values):.6f} m"
)

print(
    f"Fork Z variation        : "
    f"{max(fork_z_values) - min(fork_z_values):.6f} m"
)

print(
    f"Wheel Z minimum         : "
    f"{min(wheel_z_values):.6f} m"
)

print(
    f"Wheel Z maximum         : "
    f"{max(wheel_z_values):.6f} m"
)

print(
    f"Wheel Z variation       : "
    f"{max(wheel_z_values) - min(wheel_z_values):.6f} m"
)

print()
print("CASTER RELATIVE VERTICAL MOTION")
print("-" * 80)

print(
    f"Relative wheel/fork Z min : "
    f"{min(relative_z_values):.6f} m"
)

print(
    f"Relative wheel/fork Z max : "
    f"{max(relative_z_values):.6f} m"
)

print(
    f"Relative Z variation      : "
    f"{max(relative_z_values) - min(relative_z_values):.6f} m"
)

print()
print("GROUND CONTACT")
print("-" * 80)

print(
    f"Ground gap minimum       : "
    f"{min(gap_values):.6f} m"
)

print(
    f"Ground gap maximum       : "
    f"{max(gap_values):.6f} m"
)

print(
    f"Maximum ground gap       : "
    f"{max(gap_values):.6f} m"
)

print(
    f"Contact samples          : "
    f"{sum(contact_values)}/{len(contact_values)}"
)

print(
    f"Contact percentage       : "
    f"{100.0 * sum(contact_values) / len(contact_values):.2f}%"
)

print()
print("WHEEL ORIENTATION")
print("-" * 80)

print(
    f"Maximum axle tilt        : "
    f"{max(axle_tilt_values):.6f} deg"
)

print(
    f"Average axle tilt        : "
    f"{sum(axle_tilt_values) / len(axle_tilt_values):.6f} deg"
)

print()
print("ROBOT ATTITUDE")
print("-" * 80)

print(
    f"Maximum |pitch|          : "
    f"{max(pitch_values):.6f} deg"
)

print(
    f"Maximum |roll|           : "
    f"{max(roll_values):.6f} deg"
)

print()
print("CONTACT FORCE")
print("-" * 80)

print(
    f"Maximum normal force     : "
    f"{max(force_values):.6f} N"
)

print(
    f"Average normal force     : "
    f"{sum(force_values) / len(force_values):.6f} N"
)

# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

print()
print("=" * 80)
print("DIAGNOSTIC INTERPRETATION")
print("=" * 80)

base_variation = (
    max(base_z_values) -
    min(base_z_values)
)

fork_variation = (
    max(fork_z_values) -
    min(fork_z_values)
)

wheel_variation = (
    max(wheel_z_values) -
    min(wheel_z_values)
)

relative_variation = (
    max(relative_z_values) -
    min(relative_z_values)
)

max_gap = max(gap_values)
max_axle_tilt = max(axle_tilt_values)

print()

if base_variation < 0.005:
    print("BASE VERTICAL MOTION: LOW")
else:
    print("BASE VERTICAL MOTION: SIGNIFICANT")

if fork_variation < 0.005:
    print("FORK VERTICAL MOTION: LOW")
else:
    print("FORK VERTICAL MOTION: SIGNIFICANT")

if relative_variation < 0.005:
    print("CASTER LINKAGE VERTICAL MOTION: LOW")
else:
    print("CASTER LINKAGE VERTICAL MOTION: SIGNIFICANT")

if max_axle_tilt < 10.0:
    print("WHEEL ORIENTATION: REASONABLY HORIZONTAL")
else:
    print("WHEEL ORIENTATION: SIGNIFICANTLY TILTED")

if max_gap < 0.005:
    print("GROUND GAP: SMALL")
else:
    print("GROUND GAP: LARGE")

print()

if max_gap >= 0.02:
    print(
        "PRIMARY INDICATION:"
    )
    print(
        "Caster contact loss involves substantial vertical separation."
    )

if base_variation < 0.005 and fork_variation < 0.005:
    print(
        "The chassis/fork is not responsible for most of the vertical motion."
    )

if relative_variation >= 0.005:
    print(
        "The caster wheel is moving vertically relative to the fork."
    )

if max_axle_tilt >= 10.0:
    print(
        "Wheel orientation may contribute to contact loss."
    )

print()
print("No robot.py parameters were permanently changed.")


# ============================================================
# SAVE CSV
# ============================================================

filename = "v3_17_5_caster_vertical_contact.csv"

fieldnames = list(history[0].keys())

with open(
    filename,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(history)

print()
print("=" * 80)
print(f"CSV SAVED: {filename}")
print("=" * 80)

p.disconnect()