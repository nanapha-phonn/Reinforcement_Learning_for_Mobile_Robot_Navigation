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
# V3.17.8A.1
# STATIC GRAVITY + GROUND CONTACT
# CASTER WHEEL JOINT DIAGNOSTIC
#
# Purpose:
# Determine whether the large caster wheel/fork relative
# rotation is caused by actual caster-wheel joint rotation.
#
# Physics:
#   Gravity      = ON
#   Ground       = ON
#   Drive motors = OFF
#   Robot yaw    = 0
#   Swivel       = passive
#   Caster wheel = passive
#
# NO robot.py parameters are changed.
# ============================================================


SIMULATION_TIME = 3.0
SETTLE_TIME = 1.0

SAMPLE_INTERVAL = 0.10

# Used only for reporting the caster wheel geometry.
CASTER_WHEEL_RADIUS = 0.04


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_relative_angle(parent_quat, child_quat):
    """
    Calculate the relative rotation angle between child and parent.

    Returns:
        angle in degrees
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


def get_joint_state(robot_id, joint_index):
    """
    Return:
        joint_angle_deg
        joint_velocity
        reaction_force_vector
        applied_motor_torque
    """

    state = p.getJointState(
        robot_id,
        joint_index,
    )

    joint_angle = state[0]
    joint_velocity = state[1]
    reaction_forces = state[2]
    motor_torque = state[3]

    return (
        math.degrees(joint_angle),
        joint_velocity,
        reaction_forces,
        motor_torque,
    )


def get_contact_data(robot_id, caster_wheel_link):
    """
    Measure caster-wheel ground contact.

    Returns:
        contact
        normal_force
        ground_gap
    """

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=caster_wheel_link,
    )

    normal_force = 0.0

    if contacts:
        for contact in contacts:
            normal_force += contact[9]

    # Caster wheel world position.
    wheel_state = p.getLinkState(
        robot_id,
        caster_wheel_link,
        computeForwardKinematics=True,
    )

    wheel_pos = wheel_state[4]

    wheel_bottom_z = wheel_pos[2] - CASTER_WHEEL_RADIUS

    ground_gap = wheel_bottom_z

    contact = len(contacts) > 0

    return (
        contact,
        normal_force,
        ground_gap,
    )


def get_robot_attitude(robot_id):
    """
    Return:
        roll_deg
        pitch_deg
        yaw_deg
    """

    position, orientation = p.getBasePositionAndOrientation(
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
# DISABLE DRIVE MOTORS
# ============================================================

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


# ============================================================
# DISABLE CASTER SWIVEL MOTOR
# ============================================================

p.setJointMotorControl2(
    robot_id,
    CASTER_SWIVEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# DISABLE CASTER WHEEL MOTOR
# ============================================================

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
print("V3.17.8A.1 — STATIC CASTER WHEEL JOINT DIAGNOSTIC")
print("=" * 80)

print()
print("Gravity       : ON")
print("Ground        : ON")
print("Drive motors  : OFF")
print("Robot yaw     : 0")
print(f"Simulation    : {SIMULATION_TIME:.1f} s")
print(f"Caster trail  : 0.050 m")
print(f"Caster X      : -0.150 m")
print(f"Caster wheel radius : {CASTER_WHEEL_RADIUS:.3f} m")

print()
print("Objective:")
print(
    "Determine whether the caster wheel/fork relative rotation "
    "is caused by actual caster-wheel joint rotation."
)

print()
print("Caster wheel joint index :", CASTER_WHEEL_JOINT)
print("Caster swivel joint index:", CASTER_SWIVEL_JOINT)

print()
print("=" * 80)


# ============================================================
# SETTLE ROBOT
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

base_pos, base_quat = p.getBasePositionAndOrientation(
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

fork_quat = fork_state[5]
wheel_quat = wheel_state[5]

initial_relative_rotation = quaternion_relative_angle(
    fork_quat,
    wheel_quat,
)


(
    initial_joint_angle,
    initial_joint_velocity,
    initial_joint_reaction,
    initial_joint_torque,
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
wheel_joint_torques = []

swivel_angles = []

fork_z_values = []
wheel_z_values = []

ground_gaps = []
caster_forces = []

roll_values = []
pitch_values = []
yaw_values = []

contact_count = 0
total_samples = 0


# ============================================================
# SIMULATION
# ============================================================

simulation_steps = int(
    SIMULATION_TIME / TIME_STEP
)

sample_every = max(
    1,
    int(SAMPLE_INTERVAL / TIME_STEP)
)


for step in range(simulation_steps):

    # --------------------------------------------------------
    # Physics step
    # --------------------------------------------------------

    p.stepSimulation()


    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    if step % sample_every != 0:
        continue


    time = step * TIME_STEP


    # --------------------------------------------------------
    # Base attitude
    # --------------------------------------------------------

    (
        roll_deg,
        pitch_deg,
        yaw_deg,
    ) = get_robot_attitude(
        robot_id
    )


    # --------------------------------------------------------
    # Fork state
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True,
    )

    fork_pos = fork_state[4]
    fork_quat = fork_state[5]


    # --------------------------------------------------------
    # Caster wheel state
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True,
    )

    wheel_pos = wheel_state[4]
    wheel_quat = wheel_state[5]


    # --------------------------------------------------------
    # Relative wheel/fork rotation
    # --------------------------------------------------------

    relative_rotation = quaternion_relative_angle(
        fork_quat,
        wheel_quat,
    )


    # --------------------------------------------------------
    # Direct caster wheel joint state
    # --------------------------------------------------------

    (
        wheel_joint_angle,
        wheel_joint_velocity,
        wheel_joint_reaction,
        wheel_joint_torque,
    ) = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT,
    )


    # --------------------------------------------------------
    # Swivel joint state
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
    # Ground contact
    # --------------------------------------------------------

    (
        contact,
        caster_force,
        ground_gap,
    ) = get_contact_data(
        robot_id,
        CASTER_WHEEL_JOINT,
    )


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

    wheel_joint_torques.append(
        abs(wheel_joint_torque)
    )

    swivel_angles.append(
        swivel_angle
    )

    fork_z_values.append(
        fork_pos[2]
    )

    wheel_z_values.append(
        wheel_pos[2]
    )

    ground_gaps.append(
        ground_gap
    )

    caster_forces.append(
        caster_force
    )

    roll_values.append(
        abs(roll_deg)
    )

    pitch_values.append(
        abs(pitch_deg)
    )

    yaw_values.append(
        yaw_deg
    )


    if contact:
        contact_count += 1

    total_samples += 1


    # --------------------------------------------------------
    # Time history
    # --------------------------------------------------------

    history.append(
        (
            time,
            relative_rotation,
            wheel_joint_angle,
            wheel_joint_velocity,
            wheel_joint_torque,
            swivel_angle,
            fork_pos[2],
            wheel_pos[2],
            ground_gap,
            caster_force,
            contact,
            roll_deg,
            pitch_deg,
            yaw_deg,
        )
    )


# ============================================================
# FINAL STATE
# ============================================================

(
    final_joint_angle,
    final_joint_velocity,
    final_joint_reaction,
    final_joint_torque,
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


final_relative_rotation = relative_rotations[-1]


# ============================================================
# CALCULATIONS
# ============================================================

max_relative_rotation = max(
    relative_rotations
)

avg_relative_rotation = sum(
    relative_rotations
) / len(relative_rotations)


relative_rotation_range = (
    max(relative_rotations)
    - min(relative_rotations)
)


wheel_joint_angle_range = (
    max(wheel_joint_angles)
    - min(wheel_joint_angles)
)


max_wheel_joint_velocity = max(
    wheel_joint_velocities
)


max_wheel_joint_torque = max(
    wheel_joint_torques
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


avg_caster_force = sum(
    caster_forces
) / len(caster_forces)


contact_percentage = (
    100.0
    * contact_count
    / total_samples
)


max_roll = max(
    roll_values
)

max_pitch = max(
    pitch_values
)


swivel_range = (
    max(swivel_angles)
    - min(swivel_angles)
)


# ============================================================
# RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("V3.17.8A.1 RESULTS")
print("=" * 80)

print()

print(
    f"Relative wheel/fork rotation max : "
    f"{max_relative_rotation:.6f} deg"
)

print(
    f"Relative wheel/fork rotation avg : "
    f"{avg_relative_rotation:.6f} deg"
)

print(
    f"Relative rotation final          : "
    f"{final_relative_rotation:.6f} deg"
)

print(
    f"Relative rotation range         : "
    f"{relative_rotation_range:.6f} deg"
)

print()

print(
    f"Caster wheel joint initial angle : "
    f"{initial_joint_angle:.6f} deg"
)

print(
    f"Caster wheel joint final angle   : "
    f"{final_joint_angle:.6f} deg"
)

print(
    f"Caster wheel joint angle range   : "
    f"{wheel_joint_angle_range:.6f} deg"
)

print(
    f"Maximum wheel joint velocity     : "
    f"{max_wheel_joint_velocity:.6f} rad/s"
)

print(
    f"Maximum wheel reaction torque    : "
    f"{max_wheel_joint_torque:.6f} N*m"
)

print(
    f"Final wheel joint velocity       : "
    f"{final_joint_velocity:.6f} rad/s"
)

print(
    f"Final wheel reaction torque      : "
    f"{final_joint_torque:.6f} N*m"
)

print()

print(
    f"Swivel angle initial             : "
    f"{initial_swivel_angle:.6f} deg"
)

print(
    f"Swivel angle final               : "
    f"{final_swivel_angle:.6f} deg"
)

print(
    f"Swivel angle range               : "
    f"{swivel_range:.6f} deg"
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
    f"{max_ground_gap:.6f} m"
)

print()

print(
    f"Contact samples                  : "
    f"{contact_count}/{total_samples}"
)

print(
    f"Contact percentage               : "
    f"{contact_percentage:.2f}%"
)

print()

print(
    f"Maximum caster normal force      : "
    f"{max_caster_force:.6f} N"
)

print(
    f"Average caster normal force      : "
    f"{avg_caster_force:.6f} N"
)

print()

print(
    f"Maximum |roll|                   : "
    f"{max_roll:.6f} deg"
)

print(
    f"Maximum |pitch|                  : "
    f"{max_pitch:.6f} deg"
)

print()

print("=" * 80)
print("INITIAL → FINAL COMPARISON")
print("=" * 80)

print()

print(
    f"Relative rotation : "
    f"{initial_relative_rotation:.6f}°"
    f" → "
    f"{final_relative_rotation:.6f}°"
)

print(
    f"Wheel joint angle : "
    f"{initial_joint_angle:.6f}°"
    f" → "
    f"{final_joint_angle:.6f}°"
)

print(
    f"Swivel angle      : "
    f"{initial_swivel_angle:.6f}°"
    f" → "
    f"{final_swivel_angle:.6f}°"
)

print()


# ============================================================
# TIME HISTORY
# ============================================================

print("=" * 80)
print("TIME HISTORY")
print("=" * 80)

print()

print(
    " Time | RelRot | WheelJoint | WheelVel | "
    "Swivel | ForkZ | WheelZ | Gap | Force | Contact"
)

print("-" * 80)


for record in history:

    (
        time,
        relative_rotation,
        wheel_joint_angle,
        wheel_joint_velocity,
        wheel_joint_torque,
        swivel_angle,
        fork_z,
        wheel_z,
        ground_gap,
        caster_force,
        contact,
        roll_deg,
        pitch_deg,
        yaw_deg,
    ) = record


    print(
        f"{time:5.2f} | "
        f"{relative_rotation:7.3f}° | "
        f"{wheel_joint_angle:10.3f}° | "
        f"{wheel_joint_velocity:8.3f} | "
        f"{swivel_angle:7.3f}° | "
        f"{fork_z:6.4f} | "
        f"{wheel_z:6.4f} | "
        f"{ground_gap:7.5f} | "
        f"{caster_force:6.3f} | "
        f"{int(contact)}"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("=" * 80)
print("V3.17.8A.1 DIAGNOSTIC")
print("=" * 80)

print()

# ------------------------------------------------------------
# Primary diagnostic:
# Compare actual joint rotation with measured relative
# wheel/fork rotation.
# ------------------------------------------------------------

joint_drift = abs(
    final_joint_angle
    - initial_joint_angle
)

rotation_difference = abs(
    max_relative_rotation
    - abs(joint_drift)
)


print(
    f"Actual caster wheel joint drift : "
    f"{joint_drift:.6f} deg"
)

print(
    f"Measured relative rotation max  : "
    f"{max_relative_rotation:.6f} deg"
)

print(
    f"Difference                      : "
    f"{rotation_difference:.6f} deg"
)

print()


# ============================================================
# INTERPRETATION
# ============================================================

if joint_drift > 5.0:

    print(
        "CONFIRMED — the caster wheel joint is physically "
        "rotating under static gravity/contact conditions."
    )

    print()
    print(
        "The large wheel/fork relative rotation is therefore "
        "not merely a measurement artifact."
    )

    print()
    print(
        "Next investigation:"
    )

    print(
        "  1. Determine why the passive wheel joint receives "
        "rotational motion under static contact."
    )

    print(
        "  2. Inspect contact/friction torque on the caster."
    )

    print(
        "  3. Do NOT proceed to yaw testing yet."
    )


elif max_relative_rotation > 20.0:

    print(
        "INVESTIGATE — large relative rotation exists, "
        "but direct wheel-joint drift is small."
    )

    print()
    print(
        "This suggests the relative quaternion measurement "
        "or link-frame interpretation requires further analysis."
    )

    print()
    print(
        "Do NOT modify robot.py yet."
    )


elif max_relative_rotation < 5.0:

    print(
        "PASS — caster relative rotation remains small."
    )

    print()
    print(
        "Static gravity/contact does not reproduce the "
        "previous caster instability."
    )


else:

    print(
        "PARTIAL — moderate relative rotation detected."
    )

    print()
    print(
        "Further joint/contact investigation is required."
    )


# ============================================================
# CONTACT DIAGNOSTIC
# ============================================================

print()
print("=" * 80)
print("CONTACT DIAGNOSTIC")
print("=" * 80)

print()

if contact_percentage >= 95.0:

    print(
        f"PASS — caster contact is stable "
        f"({contact_percentage:.2f}%)."
    )

else:

    print(
        f"FAIL — caster contact is intermittent "
        f"({contact_percentage:.2f}%)."
    )


# ============================================================
# ATTITUDE DIAGNOSTIC
# ============================================================

print()

if max_roll < 2.0 and max_pitch < 2.0:

    print(
        "PASS — robot body remains approximately level."
    )

else:

    print(
        "FAIL — robot body experiences significant attitude change."
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("END V3.17.8A.1")
print("=" * 80)
print()


# ============================================================
# DISCONNECT
# ============================================================

p.disconnect()