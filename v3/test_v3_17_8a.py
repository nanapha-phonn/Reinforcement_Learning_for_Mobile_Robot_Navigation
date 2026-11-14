import math

import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_SWIVEL_JOINT,
    CASTER_WHEEL_JOINT,
    TIME_STEP,
    GRAVITY,
)


# ============================================================
# TEST PARAMETERS
# ============================================================

SIM_TIME = 3.0

CASTER_WHEEL_RADIUS = 0.04


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_relative_angle(parent_quat, child_quat):
    """
    Calculate the relative rotation magnitude between
    the caster wheel and caster fork.
    """

    _, inverse_quat = p.invertTransform(
        [0, 0, 0],
        parent_quat,
    )

    _, relative_quat = p.multiplyTransforms(
        [0, 0, 0],
        inverse_quat,
        [0, 0, 0],
        child_quat,
    )

    _, angle = p.getAxisAngleFromQuaternion(
        relative_quat
    )

    return math.degrees(angle)


def get_joint_angle(robot_id, joint_index):
    """
    Return joint position in degrees.
    """

    state = p.getJointState(
        robot_id,
        joint_index,
    )

    return math.degrees(state[0])


def get_link_pose(robot_id, link_index):
    """
    Return world position and world orientation of a link.
    """

    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True,
    )

    position = state[4]
    orientation = state[5]

    return position, orientation


def get_robot_attitude(robot_id):
    """
    Return robot roll, pitch, and yaw in degrees.
    """

    _, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def get_contact_force(body_id, link_index, ground_id):
    """
    Calculate total normal contact force between
    the specified robot link and the ground.
    """

    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=ground_id,
        linkIndexA=link_index,
    )

    total_normal_force = 0.0

    for contact in contacts:
        total_normal_force += contact[9]

    return total_normal_force, len(contacts)


# ============================================================
# CONNECT TO PYBULLET
# ============================================================

physics_client = p.connect(
    p.DIRECT
)

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

p.setPhysicsEngineParameter(
    fixedTimeStep=TIME_STEP,
    numSolverIterations=100,
)


# ============================================================
# GROUND
# ============================================================

ground_id = p.loadURDF(
    "plane.urdf",
    [0, 0, 0],
)


# ============================================================
# ROBOT
# ============================================================

# IMPORTANT:
# The current robot.py returns ONLY robot_id.
# Do not treat it as a tuple.

robot_id = create_robot()


# ============================================================
# DRIVE WHEELS OFF
# ============================================================

# Left wheel
p.setJointMotorControl2(
    robot_id,
    0,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)

# Right wheel
p.setJointMotorControl2(
    robot_id,
    1,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# CASTER SWIVEL — PASSIVE
# ============================================================

p.setJointMotorControl2(
    robot_id,
    CASTER_SWIVEL_JOINT,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# CASTER WHEEL — PASSIVE
# ============================================================

p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# TEST INFORMATION
# ============================================================

print()
print("=" * 80)
print("V3.17.8A — STATIC GRAVITY + GROUND CONTACT TEST")
print("=" * 80)
print()

print("Gravity       : ON")
print("Ground        : ON")
print("Drive motors  : OFF")
print("Robot yaw     : 0")
print("Simulation    : 3.0 s")
print("Caster trail  : 0.050 m")
print("Caster X      : -0.150 m")
print()

print("Objective:")
print(
    "Determine whether gravity + static ground contact "
    "alone causes caster instability."
)
print()


# ============================================================
# INITIAL SETTLE
# ============================================================

print("Settling robot for 1.0 s...")

settle_steps = int(
    1.0 / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()


print("Settlement complete.")
print()


# ============================================================
# DATA COLLECTION
# ============================================================

samples = []

steps = int(
    SIM_TIME / TIME_STEP
)


for step in range(steps):

    p.stepSimulation()

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    base_pos, base_quat = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    # --------------------------------------------------------
    # CASTER FORK
    # --------------------------------------------------------

    fork_pos, fork_quat = get_link_pose(
        robot_id,
        CASTER_SWIVEL_JOINT,
    )

    # --------------------------------------------------------
    # CASTER WHEEL
    # --------------------------------------------------------

    wheel_pos, wheel_quat = get_link_pose(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    # --------------------------------------------------------
    # RELATIVE WHEEL/FORK ROTATION
    # --------------------------------------------------------

    relative_rotation = (
        quaternion_relative_angle(
            fork_quat,
            wheel_quat,
        )
    )

    # --------------------------------------------------------
    # JOINT ANGLES
    # --------------------------------------------------------

    swivel_angle = get_joint_angle(
        robot_id,
        CASTER_SWIVEL_JOINT,
    )

    wheel_joint_angle = get_joint_angle(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    # --------------------------------------------------------
    # ROBOT ATTITUDE
    # --------------------------------------------------------

    roll, pitch, yaw = (
        get_robot_attitude(
            robot_id
        )
    )

    # --------------------------------------------------------
    # CASTER GROUND GAP
    # --------------------------------------------------------

    ground_gap = (
        wheel_pos[2]
        - CASTER_WHEEL_RADIUS
    )

    # --------------------------------------------------------
    # CASTER CONTACT FORCE
    # --------------------------------------------------------

    caster_force, caster_contacts = (
        get_contact_force(
            robot_id,
            CASTER_WHEEL_JOINT,
            ground_id,
        )
    )

    # --------------------------------------------------------
    # SAVE SAMPLE
    # --------------------------------------------------------

    samples.append(
        {
            "time": step * TIME_STEP,

            "relative_rotation":
                relative_rotation,

            "swivel_angle":
                swivel_angle,

            "wheel_joint_angle":
                wheel_joint_angle,

            "wheel_z":
                wheel_pos[2],

            "fork_z":
                fork_pos[2],

            "ground_gap":
                ground_gap,

            "caster_force":
                caster_force,

            "contact":
                caster_contacts > 0,

            "roll":
                roll,

            "pitch":
                pitch,

            "yaw":
                yaw,
        }
    )


# ============================================================
# EXTRACT DATA
# ============================================================

relative_rotations = [
    s["relative_rotation"]
    for s in samples
]

swivel_angles = [
    s["swivel_angle"]
    for s in samples
]

wheel_z = [
    s["wheel_z"]
    for s in samples
]

fork_z = [
    s["fork_z"]
    for s in samples
]

ground_gaps = [
    s["ground_gap"]
    for s in samples
]

caster_forces = [
    s["caster_force"]
    for s in samples
]

rolls = [
    abs(s["roll"])
    for s in samples
]

pitches = [
    abs(s["pitch"])
    for s in samples
]

contacts = [
    s["contact"]
    for s in samples
]


# ============================================================
# SUMMARY
# ============================================================

max_relative_rotation = max(
    relative_rotations
)

average_relative_rotation = (
    sum(relative_rotations)
    / len(relative_rotations)
)

swivel_min = min(
    swivel_angles
)

swivel_max = max(
    swivel_angles
)

fork_z_variation = (
    max(fork_z)
    - min(fork_z)
)

wheel_z_variation = (
    max(wheel_z)
    - min(wheel_z)
)

maximum_ground_gap = max(
    ground_gaps
)

contact_count = sum(
    contacts
)

contact_percentage = (
    100.0
    * contact_count
    / len(contacts)
)

maximum_caster_force = max(
    caster_forces
)

average_caster_force = (
    sum(caster_forces)
    / len(caster_forces)
)

maximum_roll = max(
    rolls
)

maximum_pitch = max(
    pitches
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("V3.17.8A RESULTS")
print("=" * 80)

print()

print(
    f"Relative wheel/fork rotation max : "
    f"{max_relative_rotation:.6f} deg"
)

print(
    f"Relative wheel/fork rotation avg : "
    f"{average_relative_rotation:.6f} deg"
)

print(
    f"Swivel angle range               : "
    f"{swivel_min:.6f} to "
    f"{swivel_max:.6f} deg"
)

print()

print(
    f"Fork Z variation                 : "
    f"{fork_z_variation:.6f} m"
)

print(
    f"Wheel Z variation                : "
    f"{wheel_z_variation:.6f} m"
)

print(
    f"Maximum ground gap               : "
    f"{maximum_ground_gap:.6f} m"
)

print()

print(
    f"Contact samples                  : "
    f"{contact_count}/{len(contacts)}"
)

print(
    f"Contact percentage               : "
    f"{contact_percentage:.2f}%"
)

print()

print(
    f"Maximum caster normal force      : "
    f"{maximum_caster_force:.6f} N"
)

print(
    f"Average caster normal force      : "
    f"{average_caster_force:.6f} N"
)

print()

print(
    f"Maximum |roll|                   : "
    f"{maximum_roll:.6f} deg"
)

print(
    f"Maximum |pitch|                  : "
    f"{maximum_pitch:.6f} deg"
)


# ============================================================
# TIME HISTORY
# ============================================================

print()
print("=" * 80)
print("TIME HISTORY")
print("=" * 80)

history_interval = max(
    1,
    len(samples) // 10
)

for i in range(
    0,
    len(samples),
    history_interval,
):

    s = samples[i]

    print(
        f"{s['time']:5.2f} s | "
        f"RelRot {s['relative_rotation']:7.3f}° | "
        f"Swivel {s['swivel_angle']:7.2f}° | "
        f"ForkZ {s['fork_z']:.5f} | "
        f"WheelZ {s['wheel_z']:.5f} | "
        f"Gap {s['ground_gap']:.5f} | "
        f"Force {s['caster_force']:7.3f} | "
        f"Contact {int(s['contact'])}"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("=" * 80)
print("V3.17.8A DIAGNOSTIC")
print("=" * 80)

if (
    max_relative_rotation < 5.0
    and contact_percentage > 95.0
):

    print(
        "PASS — caster remains stable under "
        "static gravity/contact."
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "Gravity and static ground contact do not "
        "produce significant caster instability."
    )

    print(
        "Proceed to V3.17.8B — controlled slow yaw."
    )


elif max_relative_rotation >= 20.0:

    print(
        "FAIL — large caster relative rotation "
        "appears without robot yaw."
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "The instability exists under static "
        "gravity/contact conditions."
    )

    print(
        "Do NOT proceed to yaw testing yet."
    )


elif contact_percentage <= 95.0:

    print(
        "PARTIAL — caster contact is intermittent, "
        "but relative orientation remains limited."
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "The caster may have a vertical/contact "
        "stability problem without large "
        "wheel/fork rotation."
    )

    print(
        "Investigate contact behavior before "
        "introducing yaw."
    )


else:

    print(
        "INVESTIGATE — behavior is between the "
        "expected stable and unstable regimes."
    )


print()
print("=" * 80)

p.disconnect()