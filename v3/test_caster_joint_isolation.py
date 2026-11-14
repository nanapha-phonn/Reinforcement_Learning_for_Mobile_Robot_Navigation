import math
import csv
import pybullet as p
import pybullet_data


# ============================================================
# V3.17.7 — CASTER JOINT-AXIS ISOLATION
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

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1
CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

TIME_STEP = 1.0 / 240.0

TEST_ANGLES_DEG = [
    0,
    -30,
    -60,
    -90,
    -120,
    -150,
    0
]

HOLD_TIME = 0.5


# ============================================================
# VECTOR FUNCTIONS
# ============================================================

def dot(a, b):
    return sum(
        a[i] * b[i]
        for i in range(3)
    )


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):

    n = norm(v)

    if n < 1e-12:
        return [0.0, 0.0, 0.0]

    return [
        x / n
        for x in v
    ]


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    value = dot(a, b)

    value = max(
        -1.0,
        min(1.0, value)
    )

    return math.degrees(
        math.acos(value)
    )


# ============================================================
# WORLD AXIS
# ============================================================

def world_axis(orientation, local_axis):

    matrix = p.getMatrixFromQuaternion(
        orientation
    )

    return [
        matrix[0] * local_axis[0]
        + matrix[3] * local_axis[1]
        + matrix[6] * local_axis[2],

        matrix[1] * local_axis[0]
        + matrix[4] * local_axis[1]
        + matrix[7] * local_axis[2],

        matrix[2] * local_axis[0]
        + matrix[5] * local_axis[1]
        + matrix[8] * local_axis[2]
    ]


# ============================================================
# RELATIVE ORIENTATION
# ============================================================

def get_relative_orientation(
    parent_orn,
    child_orn
):

    inverse_pos, inverse_orn = (
        p.invertTransform(
            [0, 0, 0],
            parent_orn
        )
    )

    _, relative_orn = (
        p.multiplyTransforms(
            [0, 0, 0],
            inverse_orn,
            [0, 0, 0],
            child_orn
        )
    )

    return relative_orn


def quaternion_angle_deg(q):

    x, y, z, w = q

    w = max(
        -1.0,
        min(1.0, abs(w))
    )

    return math.degrees(
        2.0 * math.acos(w)
    )


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
        halfExtents=[
            0.015,
            0.015,
            0.05
        ]
    )

    fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.015,
            0.015,
            0.05
        ]
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

    # IMPORTANT:
    # This is the hierarchy already established
    # for the current PyBullet installation.
    link_parent_indices = [
        0,
        0,
        0,
        3
    ]

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

        baseCollisionShapeIndex=(
            chassis_collision
        ),

        baseVisualShapeIndex=(
            chassis_visual
        ),

        basePosition=[
            0,
            0,
            0
        ],

        baseOrientation=[
            0,
            0,
            0,
            1
        ],

        linkMasses=link_masses,

        linkCollisionShapeIndices=(
            link_collision_shapes
        ),

        linkVisualShapeIndices=(
            link_visual_shapes
        ),

        linkPositions=link_positions,

        linkOrientations=link_orientations,

        linkInertialFramePositions=(
            link_inertial_positions
        ),

        linkInertialFrameOrientations=(
            link_inertial_orientations
        ),

        linkParentIndices=(
            link_parent_indices
        ),

        linkJointTypes=(
            link_joint_types
        ),

        linkJointAxis=(
            link_joint_axes
        )
    )

    return robot_id


# ============================================================
# CONNECT
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

# IMPORTANT:
# No gravity.
p.setGravity(
    0,
    0,
    0
)

p.setTimeStep(
    TIME_STEP
)

robot_id = create_robot()


# ============================================================
# DISABLE ALL MOTORS EXCEPT SWIVEL
# ============================================================

for joint in [
    LEFT_WHEEL_JOINT,
    RIGHT_WHEEL_JOINT,
    CASTER_WHEEL_JOINT
]:

    p.setJointMotorControl2(
        robot_id,
        joint,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )


# ============================================================
# INITIAL SETTLE
# ============================================================

for _ in range(100):

    p.stepSimulation()


# ============================================================
# INITIAL RELATIVE ORIENTATION
# ============================================================

fork_state = p.getLinkState(
    robot_id,
    CASTER_SWIVEL_JOINT,
    computeForwardKinematics=True
)

wheel_state = p.getLinkState(
    robot_id,
    CASTER_WHEEL_JOINT,
    computeForwardKinematics=True
)

fork_orn_initial = fork_state[5]
wheel_orn_initial = wheel_state[5]

relative_initial = (
    get_relative_orientation(
        fork_orn_initial,
        wheel_orn_initial
    )
)

initial_relative_angle = (
    quaternion_angle_deg(
        relative_initial
    )
)


print("=" * 80)
print("V3.17.7 — CASTER JOINT-AXIS ISOLATION TEST")
print("=" * 80)

print()
print("NO GRAVITY")
print("NO GROUND")
print("NO DRIVE MOTORS")
print("NO CONTACT")

print()
print(
    f"Initial relative wheel/fork rotation: "
    f"{initial_relative_angle:.6f} deg"
)

print()
print(
    "Commanding caster swivel through:"
)

print(
    " → ".join(
        f"{x}°"
        for x in TEST_ANGLES_DEG
    )
)


# ============================================================
# TEST
# ============================================================

results = []

for target_deg in TEST_ANGLES_DEG:

    target_rad = math.radians(
        target_deg
    )

    print()
    print("-" * 80)

    print(
        f"TARGET SWIVEL = "
        f"{target_deg:+.1f}°"
    )

    # Command swivel position.
    p.setJointMotorControl2(
        robot_id,
        CASTER_SWIVEL_JOINT,
        p.POSITION_CONTROL,
        targetPosition=target_rad,
        force=10.0
    )

    # Hold wheel joint passive.
    p.setJointMotorControl2(
        robot_id,
        CASTER_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    # Let the swivel settle.
    for _ in range(
        int(HOLD_TIME / TIME_STEP)
    ):

        p.stepSimulation()


    # --------------------------------------------------------
    # Joint states
    # --------------------------------------------------------

    swivel_state = p.getJointState(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    wheel_state_joint = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    actual_swivel = math.degrees(
        swivel_state[0]
    )

    swivel_velocity = (
        swivel_state[1]
    )

    wheel_angle = math.degrees(
        wheel_state_joint[0]
    )

    wheel_velocity = (
        wheel_state_joint[1]
    )


    # --------------------------------------------------------
    # Link states
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    fork_pos = fork_state[4]
    fork_orn = fork_state[5]

    wheel_pos = wheel_state[4]
    wheel_orn = wheel_state[5]


    # --------------------------------------------------------
    # Relative orientation
    # --------------------------------------------------------

    relative_orn = (
        get_relative_orientation(
            fork_orn,
            wheel_orn
        )
    )

    relative_angle = (
        quaternion_angle_deg(
            relative_orn
        )
    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    fork_z = world_axis(
        fork_orn,
        [0, 0, 1]
    )

    wheel_y = world_axis(
        wheel_orn,
        [0, 1, 0]
    )

    # Angle between caster wheel axle
    # and fork swivel axis.
    axle_vs_swivel = angle_deg(
        wheel_y,
        fork_z
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Actual swivel angle      : "
        f"{actual_swivel:+.6f}°"
    )

    print(
        f"Wheel joint angle        : "
        f"{wheel_angle:+.6f}°"
    )

    print(
        f"Relative wheel/fork rot. : "
        f"{relative_angle:.6f}°"
    )

    print(
        f"Swivel velocity          : "
        f"{swivel_velocity:.6f} rad/s"
    )

    print(
        f"Wheel velocity           : "
        f"{wheel_velocity:.6f} rad/s"
    )

    print(
        f"Axle-vs-fork-Z angle     : "
        f"{axle_vs_swivel:.6f}°"
    )


    results.append({

        "target_swivel_deg":
            target_deg,

        "actual_swivel_deg":
            actual_swivel,

        "wheel_joint_deg":
            wheel_angle,

        "relative_rotation_deg":
            relative_angle,

        "swivel_velocity":
            swivel_velocity,

        "wheel_velocity":
            wheel_velocity,

        "axle_vs_swivel_deg":
            axle_vs_swivel
    })


# ============================================================
# SUMMARY
# ============================================================

relative_values = [
    x["relative_rotation_deg"]
    for x in results
]

swivel_values = [
    x["actual_swivel_deg"]
    for x in results
]


print()
print("=" * 80)
print("V3.17.7 SUMMARY")
print("=" * 80)

print()

print(
    f"Initial relative rotation : "
    f"{initial_relative_angle:.6f}°"
)

print(
    f"Minimum relative rotation : "
    f"{min(relative_values):.6f}°"
)

print(
    f"Maximum relative rotation : "
    f"{max(relative_values):.6f}°"
)

print(
    f"Relative rotation range   : "
    f"{max(relative_values) - min(relative_values):.6f}°"
)

print()

print(
    f"Actual swivel range       : "
    f"{min(swivel_values):.6f}° "
    f"to "
    f"{max(swivel_values):.6f}°"
)

# ============================================================
# DIAGNOSIS
# ============================================================

relative_range = (
    max(relative_values)
    -
    min(relative_values)
)


print()
print("=" * 80)
print("V3.17.7 DIAGNOSIS")
print("=" * 80)

print()

if relative_range < 5.0:

    print(
        "PASS — RELATIVE ORIENTATION STABLE"
    )

    print()
    print(
        "The caster wheel maintains approximately"
    )

    print(
        "the same orientation relative to the fork."
    )

    print()
    print(
        "Therefore the previous large relative"
    )

    print(
        "orientation excursion is likely caused by"
    )

    print(
        "3D dynamic/contact interaction rather than"
    )

    print(
        "the basic joint-axis configuration."
    )

else:

    print(
        "FAIL — RELATIVE ORIENTATION NOT STABLE"
    )

    print()
    print(
        "The caster wheel changes orientation"
    )

    print(
        "substantially relative to the fork even"
    )

    print(
        "without gravity or ground contact."
    )

    print()
    print(
        "This strongly indicates a problem with"
    )

    print(
        "the caster wheel joint/link-frame geometry."
    )


# ============================================================
# SAVE CSV
# ============================================================

filename = (
    "v3_17_7_caster_joint_isolation.csv"
)

with open(
    filename,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print(
    f"CSV saved: {filename}"
)

print("=" * 80)

p.disconnect()