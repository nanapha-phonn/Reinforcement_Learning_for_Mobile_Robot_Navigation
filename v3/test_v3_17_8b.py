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
# V3.17.8B
# GRAVITY + GROUND + CONTROLLED SLOW YAW
#
# Purpose:
# Determine how the caster behaves when a slow robot yaw
# motion is introduced.
#
# IMPORTANT:
# This is a diagnostic experiment.
# No robot.py parameters are changed.
# ============================================================


# ============================================================
# PARAMETERS
# ============================================================

SIMULATION_TIME = 3.0
SETTLE_TIME = 1.0

WHEEL_RADIUS = 0.08
WHEEL_BASE = 0.30

LEFT_WHEEL_SPEED = -1.0
RIGHT_WHEEL_SPEED = 1.0

MOTOR_FORCE = 20.0

CASTER_WHEEL_RADIUS = 0.04

SAMPLE_INTERVAL = 0.10


# Theoretical differential-drive yaw rate.
THEORETICAL_YAW_RATE = (
    WHEEL_RADIUS / WHEEL_BASE
) * (
    RIGHT_WHEEL_SPEED - LEFT_WHEEL_SPEED
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_relative_angle(parent_quat, child_quat):
    """
    Relative rotation magnitude between child and parent.
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


def signed_yaw(quaternion):
    """
    Return robot yaw in degrees.
    """

    _, _, yaw = p.getEulerFromQuaternion(
        quaternion
    )

    return math.degrees(yaw)


def get_robot_state(robot_id):

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return (
        position,
        orientation,
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def get_joint_state(robot_id, joint_index):

    state = p.getJointState(
        robot_id,
        joint_index,
    )

    angle_deg = math.degrees(
        state[0]
    )

    velocity = state[1]

    reaction_forces = state[2]

    motor_torque = state[3]

    return (
        angle_deg,
        velocity,
        reaction_forces,
        motor_torque,
    )


def get_caster_contact(robot_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=CASTER_WHEEL_JOINT,
    )

    normal_force = 0.0

    for contact in contacts:
        normal_force += contact[9]

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True,
    )

    wheel_position = wheel_state[4]

    wheel_bottom_z = (
        wheel_position[2]
        - CASTER_WHEEL_RADIUS
    )

    ground_gap = wheel_bottom_z

    contact = len(contacts) > 0

    return (
        contact,
        normal_force,
        ground_gap,
    )


# ============================================================
# CONNECT
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
    numSolverIterations=100
)


# ============================================================
# GROUND
# ============================================================

ground_id = p.loadURDF(
    "plane.urdf"
)


# ============================================================
# CREATE ROBOT
# ============================================================

robot_id = create_robot()


# ============================================================
# INITIAL MOTOR STATE
# ============================================================

# Drive motors OFF during settling.

p.setJointMotorControl2(
    robot_id,
    0,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)

p.setJointMotorControl2(
    robot_id,
    1,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# Caster swivel remains passive.

p.setJointMotorControl2(
    robot_id,
    CASTER_SWIVEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# Caster wheel remains passive.

p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("V3.17.8B — CONTROLLED SLOW YAW TEST")
print("=" * 80)

print()

print("Gravity       : ON")
print("Ground        : ON")
print("Drive motors  : CONTROLLED")
print("Caster swivel : PASSIVE")
print("Caster wheel  : PASSIVE")

print()

print(
    f"Left wheel command  : "
    f"{LEFT_WHEEL_SPEED:.3f} rad/s"
)

print(
    f"Right wheel command : "
    f"{RIGHT_WHEEL_SPEED:.3f} rad/s"
)

print(
    f"Motor force         : "
    f"{MOTOR_FORCE:.3f} N*m"
)

print(
    f"Wheel radius        : "
    f"{WHEEL_RADIUS:.3f} m"
)

print(
    f"Wheel base          : "
    f"{WHEEL_BASE:.3f} m"
)

print()

print(
    f"Theoretical yaw rate: "
    f"{THEORETICAL_YAW_RATE:.6f} rad/s"
)

print(
    f"Theoretical yaw rate: "
    f"{math.degrees(THEORETICAL_YAW_RATE):.6f} deg/s"
)

print()

print("Objective:")
print(
    "Determine how the passive caster responds to "
    "controlled slow robot yaw."
)

print()
print("=" * 80)


# ============================================================
# SETTLE
# ============================================================

print()
print("Settling robot for 1.0 s...")

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):

    p.stepSimulation()


print("Settlement complete.")


# ============================================================
# INITIAL STATE
# ============================================================

(
    initial_position,
    initial_base_quat,
    initial_roll,
    initial_pitch,
    initial_yaw,
) = get_robot_state(
    robot_id
)


fork_state = p.getLinkState(
    robot_id,
    CASTER_SWIVEL_JOINT,
    computeForwardKinematics=True,
)

wheel_state = p.getLinkState(
    robot_id,
    CASTER_WHEEL_JOINT,
    computeForwardKinematics=True,
)

initial_fork_quat = fork_state[5]
initial_wheel_quat = wheel_state[5]

initial_relative_rotation = (
    quaternion_relative_angle(
        initial_fork_quat,
        initial_wheel_quat,
    )
)


(
    initial_wheel_joint_angle,
    initial_wheel_joint_velocity,
    initial_wheel_reaction,
    initial_wheel_torque,
) = get_joint_state(
    robot_id,
    CASTER_WHEEL_JOINT,
)


(
    initial_swivel_angle,
    initial_swivel_velocity,
    initial_swivel_reaction,
    initial_swivel_torque,
) = get_joint_state(
    robot_id,
    CASTER_SWIVEL_JOINT,
)


# ============================================================
# STORAGE
# ============================================================

history = []

relative_rotations = []

wheel_joint_angles = []
wheel_joint_velocities = []

swivel_angles = []

robot_yaws = []
robot_rolls = []
robot_pitches = []

fork_z_values = []
wheel_z_values = []

ground_gaps = []
caster_forces = []

yaw_rates = []

contact_count = 0
total_samples = 0


# ============================================================
# APPLY SLOW YAW COMMAND
# ============================================================

p.setJointMotorControl2(
    robot_id,
    0,
    p.VELOCITY_CONTROL,
    targetVelocity=LEFT_WHEEL_SPEED,
    force=MOTOR_FORCE,
)

p.setJointMotorControl2(
    robot_id,
    1,
    p.VELOCITY_CONTROL,
    targetVelocity=RIGHT_WHEEL_SPEED,
    force=MOTOR_FORCE,
)


# ============================================================
# SIMULATION
# ============================================================

simulation_steps = int(
    SIMULATION_TIME / TIME_STEP
)

sample_every = max(
    1,
    int(
        SAMPLE_INTERVAL
        / TIME_STEP
    )
)


previous_yaw = initial_yaw
previous_time = 0.0


for step in range(simulation_steps):

    p.stepSimulation()

    if step % sample_every != 0:
        continue

    time = step * TIME_STEP


    # --------------------------------------------------------
    # Robot state
    # --------------------------------------------------------

    (
        base_position,
        base_quat,
        roll_deg,
        pitch_deg,
        yaw_deg,
    ) = get_robot_state(
        robot_id
    )


    # --------------------------------------------------------
    # Fork
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True,
    )

    fork_position = fork_state[4]
    fork_quat = fork_state[5]


    # --------------------------------------------------------
    # Caster wheel
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True,
    )

    wheel_position = wheel_state[4]
    wheel_quat = wheel_state[5]


    # --------------------------------------------------------
    # Relative rotation
    # --------------------------------------------------------

    relative_rotation = (
        quaternion_relative_angle(
            fork_quat,
            wheel_quat,
        )
    )


    # --------------------------------------------------------
    # Wheel joint
    # --------------------------------------------------------

    (
        wheel_joint_angle,
        wheel_joint_velocity,
        wheel_reaction,
        wheel_torque,
    ) = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT,
    )


    # --------------------------------------------------------
    # Swivel
    # --------------------------------------------------------

    (
        swivel_angle,
        swivel_velocity,
        swivel_reaction,
        swivel_torque,
    ) = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT,
    )


    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    (
        contact,
        caster_force,
        ground_gap,
    ) = get_caster_contact(
        robot_id
    )


    # --------------------------------------------------------
    # Yaw rate
    # --------------------------------------------------------

    if time > previous_time:

        delta_yaw = (
            yaw_deg
            - previous_yaw
        )

        # Correct wrapping across +/-180 deg.
        while delta_yaw > 180:
            delta_yaw -= 360

        while delta_yaw < -180:
            delta_yaw += 360

        measured_yaw_rate = (
            math.radians(delta_yaw)
            / (time - previous_time)
        )

    else:

        measured_yaw_rate = 0.0


    previous_yaw = yaw_deg
    previous_time = time


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    relative_rotations.append(
        abs(relative_rotation)
    )

    wheel_joint_angles.append(
        wheel_joint_angle
    )

    wheel_joint_velocities.append(
        abs(wheel_joint_velocity)
    )

    swivel_angles.append(
        swivel_angle
    )

    robot_yaws.append(
        yaw_deg
    )

    robot_rolls.append(
        abs(roll_deg)
    )

    robot_pitches.append(
        abs(pitch_deg)
    )

    fork_z_values.append(
        fork_position[2]
    )

    wheel_z_values.append(
        wheel_position[2]
    )

    ground_gaps.append(
        ground_gap
    )

    caster_forces.append(
        caster_force
    )

    yaw_rates.append(
        measured_yaw_rate
    )


    if contact:
        contact_count += 1

    total_samples += 1


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history.append(
        (
            time,
            relative_rotation,
            wheel_joint_angle,
            wheel_joint_velocity,
            swivel_angle,
            yaw_deg,
            measured_yaw_rate,
            fork_position[2],
            wheel_position[2],
            ground_gap,
            caster_force,
            contact,
            roll_deg,
            pitch_deg,
        )
    )


# ============================================================
# FINAL STATE
# ============================================================

(
    final_position,
    final_base_quat,
    final_roll,
    final_pitch,
    final_yaw,
) = get_robot_state(
    robot_id
)


(
    final_wheel_joint_angle,
    final_wheel_joint_velocity,
    final_wheel_reaction,
    final_wheel_torque,
) = get_joint_state(
    robot_id,
    CASTER_WHEEL_JOINT,
)


(
    final_swivel_angle,
    final_swivel_velocity,
    final_swivel_reaction,
    final_swivel_torque,
) = get_joint_state(
    robot_id,
    CASTER_SWIVEL_JOINT,
)


# ============================================================
# CALCULATIONS
# ============================================================

max_relative_rotation = max(
    relative_rotations
)

avg_relative_rotation = (
    sum(relative_rotations)
    / len(relative_rotations)
)


final_relative_rotation = (
    relative_rotations[-1]
)


wheel_joint_drift = abs(
    final_wheel_joint_angle
    - initial_wheel_joint_angle
)


wheel_joint_range = (
    max(wheel_joint_angles)
    - min(wheel_joint_angles)
)


max_wheel_velocity = max(
    wheel_joint_velocities
)


swivel_range = (
    max(swivel_angles)
    - min(swivel_angles)
)


yaw_change = (
    final_yaw
    - initial_yaw
)


# Handle yaw wrapping.
while yaw_change > 180:
    yaw_change -= 360

while yaw_change < -180:
    yaw_change += 360


actual_average_yaw_rate = (
    sum(yaw_rates)
    / len(yaw_rates)
)


final_yaw_rate = (
    yaw_rates[-1]
)


rotation_ratio = (
    abs(
        actual_average_yaw_rate
    )
    / abs(
        THEORETICAL_YAW_RATE
    )
    * 100.0
)


fork_z_variation = (
    max(fork_z_values)
    - min(fork_z_values)
)


wheel_z_variation = (
    max(wheel_z_values)
    - min(wheel_z_values)
)


max_ground_gap = max(
    ground_gaps
)


max_caster_force = max(
    caster_forces
)


avg_caster_force = (
    sum(caster_forces)
    / len(caster_forces)
)


contact_percentage = (
    100.0
    * contact_count
    / total_samples
)


max_roll = max(
    robot_rolls
)


max_pitch = max(
    robot_pitches
)


# ============================================================
# RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("V3.17.8B RESULTS")
print("=" * 80)

print()

print(
    f"Theoretical yaw rate       : "
    f"{THEORETICAL_YAW_RATE:.6f} rad/s"
)

print(
    f"Average measured yaw rate : "
    f"{actual_average_yaw_rate:.6f} rad/s"
)

print(
    f"Final measured yaw rate   : "
    f"{final_yaw_rate:.6f} rad/s"
)

print(
    f"Rotation ratio            : "
    f"{rotation_ratio:.2f}%"
)

print()

print(
    f"Robot yaw change          : "
    f"{yaw_change:.6f} deg"
)

print(
    f"Initial yaw               : "
    f"{initial_yaw:.6f} deg"
)

print(
    f"Final yaw                 : "
    f"{final_yaw:.6f} deg"
)

print()

print(
    f"Relative wheel/fork max   : "
    f"{max_relative_rotation:.6f} deg"
)

print(
    f"Relative wheel/fork avg   : "
    f"{avg_relative_rotation:.6f} deg"
)

print(
    f"Relative wheel/fork final : "
    f"{final_relative_rotation:.6f} deg"
)

print()

print(
    f"Wheel joint initial       : "
    f"{initial_wheel_joint_angle:.6f} deg"
)

print(
    f"Wheel joint final         : "
    f"{final_wheel_joint_angle:.6f} deg"
)

print(
    f"Wheel joint drift         : "
    f"{wheel_joint_drift:.6f} deg"
)

print(
    f"Wheel joint range         : "
    f"{wheel_joint_range:.6f} deg"
)

print(
    f"Maximum wheel velocity    : "
    f"{max_wheel_velocity:.6f} rad/s"
)

print()

print(
    f"Swivel initial            : "
    f"{initial_swivel_angle:.6f} deg"
)

print(
    f"Swivel final              : "
    f"{final_swivel_angle:.6f} deg"
)

print(
    f"Swivel range              : "
    f"{swivel_range:.6f} deg"
)

print()

print(
    f"Fork Z variation          : "
    f"{fork_z_variation:.6f} m"
)

print(
    f"Wheel Z variation         : "
    f"{wheel_z_variation:.6f} m"
)

print(
    f"Maximum ground gap        : "
    f"{max_ground_gap:.6f} m"
)

print()

print(
    f"Contact samples           : "
    f"{contact_count}/{total_samples}"
)

print(
    f"Contact percentage        : "
    f"{contact_percentage:.2f}%"
)

print()

print(
    f"Maximum caster force      : "
    f"{max_caster_force:.6f} N"
)

print(
    f"Average caster force      : "
    f"{avg_caster_force:.6f} N"
)

print()

print(
    f"Maximum |roll|            : "
    f"{max_roll:.6f} deg"
)

print(
    f"Maximum |pitch|           : "
    f"{max_pitch:.6f} deg"
)


# ============================================================
# TIME HISTORY
# ============================================================

print()
print("=" * 80)
print("TIME HISTORY")
print("=" * 80)

print()

print(
    " Time | RelRot | WheelJoint | WheelVel | "
    "Swivel | Yaw | YawRate | Gap | Force | Contact"
)

print("-" * 100)


for record in history:

    (
        time,
        relative_rotation,
        wheel_joint_angle,
        wheel_joint_velocity,
        swivel_angle,
        yaw_deg,
        measured_yaw_rate,
        fork_z,
        wheel_z,
        ground_gap,
        caster_force,
        contact,
        roll_deg,
        pitch_deg,
    ) = record


    print(
        f"{time:5.2f} | "
        f"{relative_rotation:7.3f}° | "
        f"{wheel_joint_angle:10.3f}° | "
        f"{wheel_joint_velocity:8.3f} | "
        f"{swivel_angle:7.3f}° | "
        f"{yaw_deg:7.2f}° | "
        f"{measured_yaw_rate:7.3f} | "
        f"{ground_gap:7.5f} | "
        f"{caster_force:6.3f} | "
        f"{int(contact)}"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("=" * 80)
print("V3.17.8B DIAGNOSTIC")
print("=" * 80)

print()

# ------------------------------------------------------------
# Robot yaw
# ------------------------------------------------------------

if rotation_ratio >= 90.0:

    print(
        f"PASS — slow yaw tracks theory "
        f"({rotation_ratio:.2f}%)."
    )

elif rotation_ratio >= 75.0:

    print(
        f"PARTIAL — yaw tracking is reasonable but "
        f"below 90% ({rotation_ratio:.2f}%)."
    )

else:

    print(
        f"FAIL — yaw tracking is poor "
        f"({rotation_ratio:.2f}%)."
    )


# ------------------------------------------------------------
# Caster contact
# ------------------------------------------------------------

print()

if contact_percentage >= 95.0:

    print(
        f"PASS — caster contact remains stable "
        f"({contact_percentage:.2f}%)."
    )

else:

    print(
        f"FAIL — caster contact becomes intermittent "
        f"({contact_percentage:.2f}%)."
    )


# ------------------------------------------------------------
# Robot attitude
# ------------------------------------------------------------

print()

if max_roll < 2.0 and max_pitch < 2.0:

    print(
        "PASS — robot remains approximately level."
    )

else:

    print(
        "FAIL — robot experiences significant "
        "roll/pitch motion."
    )


# ------------------------------------------------------------
# Caster relative rotation
# ------------------------------------------------------------

print()

if max_relative_rotation < 5.0:

    print(
        "PASS — caster relative rotation remains small."
    )

elif max_relative_rotation < 20.0:

    print(
        "PARTIAL — caster experiences moderate "
        "relative rotation."
    )

else:

    print(
        "FAIL — caster experiences large "
        "relative rotation."
    )


# ------------------------------------------------------------
# Wheel joint
# ------------------------------------------------------------

print()

if wheel_joint_drift > 20.0:

    print(
        "CONFIRMED — caster wheel joint undergoes "
        "large physical rotation."
    )

elif wheel_joint_drift > 5.0:

    print(
        "WARNING — caster wheel joint undergoes "
        "moderate physical rotation."
    )

else:

    print(
        "PASS — caster wheel joint remains "
        "approximately stable."
    )


# ============================================================
# OVERALL INTERPRETATION
# ============================================================

print()
print("=" * 80)
print("OVERALL INTERPRETATION")
print("=" * 80)

print()

print(
    "This test introduces robot yaw while keeping gravity "
    "and ground contact active."
)

print()

print(
    "Compare the caster behavior against V3.17.8A/A.1."
)

print()

print(
    "IMPORTANT:"
)

print(
    "A large caster-wheel rotation in this test does not "
    "automatically mean the yaw model is wrong."
)

print(
    "V3.17.8A.1 already demonstrated that the caster wheel "
    "can rotate under static ground contact."
)

print(
    "Therefore this experiment is used to determine whether "
    "robot yaw changes the magnitude or character of that "
    "existing caster behavior."
)

print()

print("=" * 80)
print("END V3.17.8B")
print("=" * 80)

print()


# ============================================================
# DISCONNECT
# ============================================================

p.disconnect()