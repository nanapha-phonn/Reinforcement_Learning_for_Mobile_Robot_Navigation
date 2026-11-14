import math
import csv
import pybullet as p
import pybullet_data


# ============================================================
# V3.17.6 CASTER RELATIVE-ORIENTATION DIAGNOSTIC
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

SETTLE_TIME = 2.0
ROTATION_TIME = 3.0

SAMPLE_INTERVAL = 0.1


# ============================================================
# VECTOR HELPERS
# ============================================================

def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = norm(v)

    if n < 1e-12:
        return [0.0, 0.0, 0.0]

    return [x / n for x in v]


def angle_deg(a, b):
    na = norm(a)
    nb = norm(b)

    if na < 1e-12 or nb < 1e-12:
        return float("nan")

    value = dot(a, b) / (na * nb)
    value = max(-1.0, min(1.0, value))

    return math.degrees(math.acos(value))


def vector_sub(a, b):
    return [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2]
    ]


# ============================================================
# QUATERNION HELPERS
# ============================================================

def quaternion_inverse(q):
    return p.invertTransform(
        [0, 0, 0],
        q
    )[1]


def relative_orientation(parent_orn, child_orn):
    """
    Rotation of child expressed relative to parent.
    """

    parent_inv = p.invertTransform(
        [0, 0, 0],
        parent_orn
    )[1]

    relative_pos, relative_orn = p.multiplyTransforms(
        [0, 0, 0],
        parent_inv,
        [0, 0, 0],
        child_orn
    )

    return relative_orn


def quaternion_angle_deg(q):
    """
    Total rotation represented by quaternion.
    """

    x, y, z, w = q

    w = max(-1.0, min(1.0, w))

    angle = 2.0 * math.acos(abs(w))

    return math.degrees(angle)


# ============================================================
# LOCAL AXIS -> WORLD
# ============================================================

def world_axis(orientation, axis):
    matrix = p.getMatrixFromQuaternion(orientation)

    return [
        matrix[0] * axis[0]
        + matrix[3] * axis[1]
        + matrix[6] * axis[2],

        matrix[1] * axis[0]
        + matrix[4] * axis[1]
        + matrix[7] * axis[2],

        matrix[2] * axis[0]
        + matrix[5] * axis[1]
        + matrix[8] * axis[2]
    ]


# ============================================================
# ROBOT
# ============================================================

def create_robot():

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

    fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.015, 0.015, 0.05]
    )

    fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.015, 0.015, 0.05]
    )

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
        [0.0, 0.15, -0.05],
        [0.0, -0.15, -0.05],
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

    # Empirically validated hierarchy for this PyBullet build.
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

    # Base
    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # Drive wheels
    for joint in [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]:

        p.changeDynamics(
            robot_id,
            joint,
            lateralFriction=0.5,
            rollingFriction=0.0,
            spinningFriction=0.0
        )

    # Fork
    p.changeDynamics(
        robot_id,
        CASTER_SWIVEL_JOINT,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # Caster wheel
    p.changeDynamics(
        robot_id,
        CASTER_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # Disable self-collision.
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
# CONTACT
# ============================================================

def get_caster_contacts(robot_id, plane_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=CASTER_WHEEL_JOINT
    )

    total_force = 0.0

    points = []

    for c in contacts:

        total_force += c[9]

        points.append({
            "wheel_point": c[5],
            "ground_point": c[6],
            "normal_force": c[9]
        })

    return contacts, points, total_force


# ============================================================
# SETUP
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(
    0,
    0,
    GRAVITY
)

p.setTimeStep(
    TIME_STEP
)

plane_id = p.loadURDF("plane.urdf")

robot_id = create_robot()


# Disable caster motors completely.
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
print("V3.17.6 CASTER RELATIVE-ORIENTATION DIAGNOSTIC")
print("=" * 80)

print()
print("Fixed parameters:")
print(f"CASTER_X     = {CASTER_X:.3f} m")
print(f"CASTER_TRAIL = {CASTER_TRAIL:.3f} m")

print()
print("Purpose:")
print("Determine whether the caster wheel maintains")
print("the intended orientation relative to the fork.")

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

for _ in range(
    int(SETTLE_TIME / TIME_STEP)
):
    p.stepSimulation()


# ============================================================
# INITIAL ORIENTATION
# ============================================================

fork_initial = p.getLinkState(
    robot_id,
    CASTER_SWIVEL_JOINT,
    computeForwardKinematics=True
)

wheel_initial = p.getLinkState(
    robot_id,
    CASTER_WHEEL_JOINT,
    computeForwardKinematics=True
)

fork_initial_orn = fork_initial[5]
wheel_initial_orn = wheel_initial[5]

initial_relative_orn = relative_orientation(
    fork_initial_orn,
    wheel_initial_orn
)

print()
print("=" * 80)
print("INITIAL ORIENTATION")
print("=" * 80)

print(
    "Fork quaternion:"
)
print(
    "  ",
    tuple(round(x, 6) for x in fork_initial_orn)
)

print(
    "Wheel quaternion:"
)
print(
    "  ",
    tuple(round(x, 6) for x in wheel_initial_orn)
)

print(
    "Relative wheel/fork quaternion:"
)
print(
    "  ",
    tuple(round(x, 6) for x in initial_relative_orn)
)

print(
    "Relative rotation magnitude:"
    f" {quaternion_angle_deg(initial_relative_orn):.6f} deg"
)


# ============================================================
# ROTATION
# ============================================================

print()
print("=" * 80)
print("ROTATION START")
print("=" * 80)

history = []

steps = int(
    ROTATION_TIME / TIME_STEP
)

sample_every = max(
    1,
    int(SAMPLE_INTERVAL / TIME_STEP)
)


for step in range(steps + 1):

    # Drive wheels.
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

    # Keep caster completely passive.
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
    # Base
    # --------------------------------------------------------

    base_pos, base_orn = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    base_euler = p.getEulerFromQuaternion(
        base_orn
    )

    roll_deg = math.degrees(
        base_euler[0]
    )

    pitch_deg = math.degrees(
        base_euler[1]
    )

    yaw_deg = math.degrees(
        base_euler[2]
    )


    # --------------------------------------------------------
    # Fork
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True
    )

    fork_pos = fork_state[4]
    fork_orn = fork_state[5]


    # --------------------------------------------------------
    # Wheel
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_pos = wheel_state[4]
    wheel_orn = wheel_state[5]


    # --------------------------------------------------------
    # World axes
    # --------------------------------------------------------

    fork_swivel_axis = world_axis(
        fork_orn,
        [0, 0, 1]
    )

    wheel_axle_axis = world_axis(
        wheel_orn,
        [0, 1, 0]
    )

    wheel_local_x = world_axis(
        wheel_orn,
        [1, 0, 0]
    )

    wheel_local_z = world_axis(
        wheel_orn,
        [0, 0, 1]
    )


    # --------------------------------------------------------
    # Relative wheel/fork orientation
    # --------------------------------------------------------

    relative_orn = relative_orientation(
        fork_orn,
        wheel_orn
    )

    relative_angle = quaternion_angle_deg(
        relative_orn
    )


    # --------------------------------------------------------
    # Axle tilt
    # --------------------------------------------------------

    axle_tilt = angle_deg(
        wheel_axle_axis,
        [0, 1, 0]
    )

    # Since the axle should remain horizontal,
    # measure its deviation from the horizontal plane.
    axle_vertical_component = abs(
        wheel_axle_axis[2]
    )

    axle_tilt_from_horizontal = math.degrees(
        math.asin(
            max(
                -1.0,
                min(
                    1.0,
                    axle_vertical_component
                )
            )
        )
    )


    # --------------------------------------------------------
    # Fork / wheel relative position
    # --------------------------------------------------------

    relative_position = vector_sub(
        wheel_pos,
        fork_pos
    )

    relative_position_z = (
        relative_position[2]
    )


    # --------------------------------------------------------
    # Ground gap
    # --------------------------------------------------------

    wheel_z = wheel_pos[2]

    ground_gap = (
        wheel_z -
        CASTER_WHEEL_RADIUS
    )


    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    contacts, contact_points, normal_force = (
        get_caster_contacts(
            robot_id,
            plane_id
        )
    )

    contact = int(
        len(contacts) > 0
    )


    # --------------------------------------------------------
    # Contact point
    # --------------------------------------------------------

    if contact_points:

        cp = contact_points[0]

        contact_x = cp["wheel_point"][0]
        contact_y = cp["wheel_point"][1]
        contact_z = cp["wheel_point"][2]

        ground_x = cp["ground_point"][0]
        ground_y = cp["ground_point"][1]
        ground_z = cp["ground_point"][2]

    else:

        contact_x = float("nan")
        contact_y = float("nan")
        contact_z = float("nan")

        ground_x = float("nan")
        ground_y = float("nan")
        ground_z = float("nan")


    # --------------------------------------------------------
    # Swivel joint
    # --------------------------------------------------------

    swivel_state = p.getJointState(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    swivel_angle_deg = math.degrees(
        swivel_state[0]
    )

    swivel_velocity = swivel_state[1]


    # --------------------------------------------------------
    # Caster wheel joint
    # --------------------------------------------------------

    wheel_joint_state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    wheel_angle_deg = math.degrees(
        wheel_joint_state[0]
    )

    wheel_velocity = wheel_joint_state[1]


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    history.append({

        "time":
            step * TIME_STEP,

        "base_z":
            base_pos[2],

        "roll_deg":
            roll_deg,

        "pitch_deg":
            pitch_deg,

        "yaw_deg":
            yaw_deg,

        "fork_x":
            fork_pos[0],

        "fork_y":
            fork_pos[1],

        "fork_z":
            fork_pos[2],

        "fork_qx":
            fork_orn[0],

        "fork_qy":
            fork_orn[1],

        "fork_qz":
            fork_orn[2],

        "fork_qw":
            fork_orn[3],

        "wheel_x":
            wheel_pos[0],

        "wheel_y":
            wheel_pos[1],

        "wheel_z":
            wheel_pos[2],

        "wheel_qx":
            wheel_orn[0],

        "wheel_qy":
            wheel_orn[1],

        "wheel_qz":
            wheel_orn[2],

        "wheel_qw":
            wheel_orn[3],

        "relative_qx":
            relative_orn[0],

        "relative_qy":
            relative_orn[1],

        "relative_qz":
            relative_orn[2],

        "relative_qw":
            relative_orn[3],

        "relative_angle_deg":
            relative_angle,

        "fork_axis_x":
            fork_swivel_axis[0],

        "fork_axis_y":
            fork_swivel_axis[1],

        "fork_axis_z":
            fork_swivel_axis[2],

        "axle_x":
            wheel_axle_axis[0],

        "axle_y":
            wheel_axle_axis[1],

        "axle_z":
            wheel_axle_axis[2],

        "axle_tilt_deg":
            axle_tilt_from_horizontal,

        "relative_position_z":
            relative_position_z,

        "ground_gap":
            ground_gap,

        "contact":
            contact,

        "contact_count":
            len(contacts),

        "normal_force":
            normal_force,

        "contact_x":
            contact_x,

        "contact_y":
            contact_y,

        "contact_z":
            contact_z,

        "ground_x":
            ground_x,

        "ground_y":
            ground_y,

        "ground_z":
            ground_z,

        "swivel_angle_deg":
            swivel_angle_deg,

        "swivel_velocity":
            swivel_velocity,

        "wheel_angle_deg":
            wheel_angle_deg,

        "wheel_velocity":
            wheel_velocity
    })


# ============================================================
# PRINT HISTORY
# ============================================================

print()
print("=" * 80)
print("V3.17.6 RELATIVE-ORIENTATION HISTORY")
print("=" * 80)

print(
    "Time | Swivel | RelRot | "
    "AxleTilt | ForkZ | WheelZ | Gap | Force | Contact"
)

print("-" * 105)

for r in history:

    print(
        f"{r['time']:4.2f} | "
        f"{r['swivel_angle_deg']:8.2f} | "
        f"{r['relative_angle_deg']:7.2f} | "
        f"{r['axle_tilt_deg']:8.2f} | "
        f"{r['fork_z']:6.4f} | "
        f"{r['wheel_z']:6.4f} | "
        f"{r['ground_gap']:6.4f} | "
        f"{r['normal_force']:7.3f} | "
        f"{r['contact']}"
    )


# ============================================================
# SUMMARY
# ============================================================

relative_angles = [
    r["relative_angle_deg"]
    for r in history
]

axle_tilts = [
    r["axle_tilt_deg"]
    for r in history
]

gaps = [
    r["ground_gap"]
    for r in history
]

forces = [
    r["normal_force"]
    for r in history
]

contact_values = [
    r["contact"]
    for r in history
]

fork_z = [
    r["fork_z"]
    for r in history
]

wheel_z = [
    r["wheel_z"]
    for r in history
]

relative_z = [
    r["relative_position_z"]
    for r in history
]

swivel_angles = [
    r["swivel_angle_deg"]
    for r in history
]


print()
print("=" * 80)
print("V3.17.6 SUMMARY")
print("=" * 80)

print()
print("SWIVEL")
print("-" * 80)

print(
    f"Swivel angle minimum       : "
    f"{min(swivel_angles):.6f} deg"
)

print(
    f"Swivel angle maximum       : "
    f"{max(swivel_angles):.6f} deg"
)

print(
    f"Swivel angle range         : "
    f"{max(swivel_angles) - min(swivel_angles):.6f} deg"
)

print()
print("RELATIVE WHEEL/FORK ORIENTATION")
print("-" * 80)

print(
    f"Relative rotation minimum  : "
    f"{min(relative_angles):.6f} deg"
)

print(
    f"Relative rotation maximum  : "
    f"{max(relative_angles):.6f} deg"
)

print(
    f"Relative rotation range    : "
    f"{max(relative_angles) - min(relative_angles):.6f} deg"
)

print()
print("WHEEL AXLE")
print("-" * 80)

print(
    f"Maximum axle tilt          : "
    f"{max(axle_tilts):.6f} deg"
)

print(
    f"Average axle tilt          : "
    f"{sum(axle_tilts) / len(axle_tilts):.6f} deg"
)

print()
print("VERTICAL GEOMETRY")
print("-" * 80)

print(
    f"Fork Z variation           : "
    f"{max(fork_z) - min(fork_z):.6f} m"
)

print(
    f"Wheel Z variation          : "
    f"{max(wheel_z) - min(wheel_z):.6f} m"
)

print(
    f"Relative wheel/fork Z var. : "
    f"{max(relative_z) - min(relative_z):.6f} m"
)

print()
print("GROUND CONTACT")
print("-" * 80)

print(
    f"Maximum ground gap         : "
    f"{max(gaps):.6f} m"
)

print(
    f"Contact samples            : "
    f"{sum(contact_values)}/{len(contact_values)}"
)

print(
    f"Contact percentage         : "
    f"{100 * sum(contact_values) / len(contact_values):.2f}%"
)

print()
print("FORCE")
print("-" * 80)

print(
    f"Maximum normal force       : "
    f"{max(forces):.6f} N"
)

print(
    f"Average normal force       : "
    f"{sum(forces) / len(forces):.6f} N"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 80)
print("V3.17.6 INTERPRETATION")
print("=" * 80)

print()

relative_range = (
    max(relative_angles) -
    min(relative_angles)
)

max_axle_tilt = max(axle_tilts)

if relative_range < 10.0:
    print(
        "RELATIVE ORIENTATION: STABLE"
    )
    print(
        "The caster wheel maintains approximately"
    )
    print(
        "the same orientation relative to the fork."
    )
else:
    print(
        "RELATIVE ORIENTATION: CHANGING"
    )
    print(
        "The caster wheel orientation changes"
    )
    print(
        "substantially relative to the fork."
    )

print()

if max_axle_tilt < 10.0:
    print(
        "AXLE ORIENTATION: ACCEPTABLE"
    )
else:
    print(
        "AXLE ORIENTATION: SIGNIFICANTLY TILTED"
    )

print()

if max_axle_tilt >= 45.0:
    print(
        "IMPORTANT:"
    )
    print(
        "The caster wheel reaches a strongly tilted"
    )
    print(
        "orientation during the rotation."
    )

print()

if max(gaps) > 0.005:
    print(
        "CONTACT:"
    )
    print(
        "Ground separation remains significant."
    )

print()
print(
    "No robot.py parameters were permanently changed."
)


# ============================================================
# SAVE CSV
# ============================================================

filename = (
    "v3_17_6_caster_relative_orientation.csv"
)

with open(
    filename,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


print()
print("=" * 80)
print(
    f"CSV SAVED: {filename}"
)
print("=" * 80)

p.disconnect()