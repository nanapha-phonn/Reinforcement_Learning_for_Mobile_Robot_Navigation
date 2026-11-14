import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_WHEEL_JOINT,
)


# ============================================================
# V3.17.8A.3
# Zero-Friction Caster Isolation Test
#
# Purpose:
# Determine whether caster-wheel friction is responsible
# for the static caster-wheel rotation observed in A.2.
#
# IMPORTANT:
# - Do NOT modify robot.py
# - Gravity remains ON
# - Ground remains ON
# - Drive motors remain OFF
# - Robot yaw remains fixed
# - Only caster-wheel friction is changed to zero
# ============================================================


TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 1.0
TEST_TIME = 3.0


def quaternion_relative_angle(parent_quat, child_quat):
    """
    Calculate the relative rotation angle between two bodies.
    Returns degrees.
    """

    _, inverse_quat = p.invertTransform(
        [0, 0, 0],
        parent_quat
    )

    _, relative_quat = p.multiplyTransforms(
        [0, 0, 0],
        inverse_quat,
        [0, 0, 0],
        child_quat
    )

    _, angle = p.getAxisAngleFromQuaternion(relative_quat)

    return math.degrees(angle)


# ============================================================
# CONNECT
# ============================================================

physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

p.setTimeStep(TIME_STEP)

p.setPhysicsEngineParameter(
    fixedTimeStep=TIME_STEP,
    numSolverIterations=100
)


# ============================================================
# GROUND
# ============================================================

ground_id = p.loadURDF(
    "plane.urdf",
    [0, 0, 0]
)


# ============================================================
# CREATE ROBOT
# ============================================================

robot_id = create_robot()


# ============================================================
# A.3 CONTROLLED VARIABLE
#
# ONLY CHANGE:
# caster wheel friction = 0
#
# Do NOT change:
# - chassis friction
# - drive-wheel friction
# - caster geometry
# - caster masses
# - joint axes
# - trail
# - caster position
# ============================================================

p.changeDynamics(
    robot_id,
    CASTER_WHEEL_JOINT,
    lateralFriction=0.0,
    spinningFriction=0.0,
    rollingFriction=0.0
)


# ============================================================
# IMPORTANT:
# Drive motors OFF
# Caster joints PASSIVE
# ============================================================

p.setJointMotorControl2(
    robot_id,
    0,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0
)

p.setJointMotorControl2(
    robot_id,
    1,
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

# Caster swivel passive
# Do not apply motor torque.


# ============================================================
# INITIAL STATE
# ============================================================

base_pos, base_quat = p.getBasePositionAndOrientation(robot_id)

joint_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

initial_wheel_angle = joint_state[0]


# ============================================================
# SETTLE
# ============================================================

settle_steps = int(SETTLE_TIME / TIME_STEP)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# MEASURE INITIAL STATE AFTER SETTLING
# ============================================================

base_pos, base_quat = p.getBasePositionAndOrientation(robot_id)

joint_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

settled_wheel_angle = joint_state[0]


# ============================================================
# STORAGE
# ============================================================

relative_angles = []

wheel_angles = []
wheel_velocities = []

normal_forces = []
friction_forces = []

axle_torques = []

contact_count = 0
total_samples = 0

roll_values = []
pitch_values = []

time_history = []


# ============================================================
# SIMULATION
# ============================================================

test_steps = int(TEST_TIME / TIME_STEP)

sample_interval = max(1, int(0.1 / TIME_STEP))


for step in range(test_steps):

    p.stepSimulation()

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    base_pos, base_quat = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        base_quat
    )

    roll_values.append(math.degrees(roll))
    pitch_values.append(math.degrees(pitch))


    # --------------------------------------------------------
    # CASTER FORK
    # --------------------------------------------------------

    fork_state = p.getLinkState(
        robot_id,
        2,
        computeForwardKinematics=True
    )

    fork_pos = fork_state[4]
    fork_quat = fork_state[5]


    # --------------------------------------------------------
    # CASTER WHEEL
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_pos = wheel_state[4]
    wheel_quat = wheel_state[5]


    # --------------------------------------------------------
    # RELATIVE WHEEL/FORK ROTATION
    # --------------------------------------------------------

    relative_angle = quaternion_relative_angle(
        fork_quat,
        wheel_quat
    )

    relative_angles.append(relative_angle)


    # --------------------------------------------------------
    # WHEEL JOINT
    # --------------------------------------------------------

    joint_state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    wheel_angle = joint_state[0]
    wheel_velocity = joint_state[1]

    wheel_angles.append(
        math.degrees(wheel_angle)
    )

    wheel_velocities.append(
        abs(wheel_velocity)
    )


    # --------------------------------------------------------
    # CONTACT ANALYSIS
    # --------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=CASTER_WHEEL_JOINT
    )

    total_samples += 1

    if contacts:
        contact_count += 1


    sample_normal_force = 0.0
    sample_friction_force = 0.0

    sample_axle_torque = 0.0


    for contact in contacts:

        normal_force = contact[9]

        friction_force_1 = contact[10]
        friction_dir_1 = contact[11]

        friction_force_2 = contact[12]
        friction_dir_2 = contact[13]

        contact_position = contact[5]

        sample_normal_force += abs(
            normal_force
        )

        sample_friction_force += math.sqrt(
            friction_force_1 ** 2
            +
            friction_force_2 ** 2
        )


        # ----------------------------------------------------
        # NORMAL FORCE TORQUE
        # ----------------------------------------------------

        rx = contact_position[0] - wheel_pos[0]
        ry = contact_position[1] - wheel_pos[1]
        rz = contact_position[2] - wheel_pos[2]

        normal_vector = [
            contact[7][0] * normal_force,
            contact[7][1] * normal_force,
            contact[7][2] * normal_force
        ]

        normal_torque = [
            ry * normal_vector[2] - rz * normal_vector[1],
            rz * normal_vector[0] - rx * normal_vector[2],
            rx * normal_vector[1] - ry * normal_vector[0]
        ]


        # ----------------------------------------------------
        # FRICTION TORQUE
        # ----------------------------------------------------

        friction_vector_1 = [
            friction_force_1 * friction_dir_1[0],
            friction_force_1 * friction_dir_1[1],
            friction_force_1 * friction_dir_1[2]
        ]

        friction_vector_2 = [
            friction_force_2 * friction_dir_2[0],
            friction_force_2 * friction_dir_2[1],
            friction_force_2 * friction_dir_2[2]
        ]

        friction_vector = [
            friction_vector_1[0] + friction_vector_2[0],
            friction_vector_1[1] + friction_vector_2[1],
            friction_vector_1[2] + friction_vector_2[2]
        ]

        friction_torque = [
            ry * friction_vector[2] - rz * friction_vector[1],
            rz * friction_vector[0] - rx * friction_vector[2],
            rx * friction_vector[1] - ry * friction_vector[0]
        ]


        total_torque = [
            normal_torque[0] + friction_torque[0],
            normal_torque[1] + friction_torque[1],
            normal_torque[2] + friction_torque[2]
        ]


        # ----------------------------------------------------
        # WHEEL AXLE
        #
        # Wheel axle is local Y axis.
        # ----------------------------------------------------

        wheel_rotation_matrix = p.getMatrixFromQuaternion(
            wheel_quat
        )

        wheel_rotation_matrix = [
            wheel_rotation_matrix[0:3],
            wheel_rotation_matrix[3:6],
            wheel_rotation_matrix[6:9]
        ]

        axle = [
            wheel_rotation_matrix[1][0],
            wheel_rotation_matrix[1][1],
            wheel_rotation_matrix[1][2]
        ]


        axle_torque = abs(
            total_torque[0] * axle[0]
            +
            total_torque[1] * axle[1]
            +
            total_torque[2] * axle[2]
        )

        sample_axle_torque += axle_torque


    normal_forces.append(
        sample_normal_force
    )

    friction_forces.append(
        sample_friction_force
    )

    axle_torques.append(
        sample_axle_torque
    )


    # --------------------------------------------------------
    # TIME HISTORY
    # --------------------------------------------------------

    if step % sample_interval == 0:

        time_history.append(
            (
                step * TIME_STEP,
                math.degrees(wheel_angle),
                relative_angle,
                wheel_velocity,
                sample_normal_force,
                sample_friction_force
            )
        )


# ============================================================
# FINAL VALUES
# ============================================================

final_joint_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

final_wheel_angle = final_joint_state[0]


wheel_angle_drift = math.degrees(
    final_wheel_angle - settled_wheel_angle
)


max_relative_angle = max(
    relative_angles
)

avg_relative_angle = sum(
    relative_angles
) / len(relative_angles)


max_wheel_velocity = max(
    wheel_velocities
)

avg_wheel_velocity = sum(
    wheel_velocities
) / len(wheel_velocities)


max_normal_force = max(
    normal_forces
)

avg_normal_force = sum(
    normal_forces
) / len(normal_forces)


max_friction_force = max(
    friction_forces
)

avg_friction_force = sum(
    friction_forces
) / len(friction_forces)


max_axle_torque = max(
    axle_torques
)

avg_axle_torque = sum(
    axle_torques
) / len(axle_torques)


contact_percentage = (
    contact_count / total_samples
) * 100.0


max_roll = max(
    abs(value)
    for value in roll_values
)

max_pitch = max(
    abs(value)
    for value in pitch_values
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("V3.17.8A.3 — ZERO-FRICTION CASTER ISOLATION")
print("=" * 65)

print()
print("CONTROLLED VARIABLE")
print("-" * 65)

print("Caster lateral friction : 0.0")
print("Caster spinning friction: 0.0")
print("Caster rolling friction : 0.0")

print()
print("CASTER WHEEL")
print("-" * 65)

print(
    f"Initial wheel angle       : "
    f"{math.degrees(settled_wheel_angle):.6f} deg"
)

print(
    f"Final wheel angle         : "
    f"{math.degrees(final_wheel_angle):.6f} deg"
)

print(
    f"Wheel angle drift         : "
    f"{wheel_angle_drift:.6f} deg"
)

print(
    f"Maximum wheel velocity    : "
    f"{max_wheel_velocity:.6f} rad/s"
)

print(
    f"Average wheel velocity    : "
    f"{avg_wheel_velocity:.6f} rad/s"
)

print()
print("RELATIVE WHEEL/FORK ROTATION")
print("-" * 65)

print(
    f"Maximum relative rotation : "
    f"{max_relative_angle:.6f} deg"
)

print(
    f"Average relative rotation : "
    f"{avg_relative_angle:.6f} deg"
)

print()
print("CONTACT")
print("-" * 65)

print(
    f"Maximum normal force      : "
    f"{max_normal_force:.6f} N"
)

print(
    f"Average normal force      : "
    f"{avg_normal_force:.6f} N"
)

print(
    f"Maximum friction force    : "
    f"{max_friction_force:.9f} N"
)

print(
    f"Average friction force    : "
    f"{avg_friction_force:.9f} N"
)

print(
    f"Maximum axle torque       : "
    f"{max_axle_torque:.9f} N*m"
)

print(
    f"Average axle torque       : "
    f"{avg_axle_torque:.9f} N*m"
)

print(
    f"Contact percentage        : "
    f"{contact_percentage:.2f}%"
)

print()
print("ROBOT STABILITY")
print("-" * 65)

print(
    f"Maximum |roll|            : "
    f"{max_roll:.6f} deg"
)

print(
    f"Maximum |pitch|           : "
    f"{max_pitch:.6f} deg"
)

print()
print("A.2 BASELINE")
print("-" * 65)

print(
    "A.2 wheel drift           : "
    "24.513634 deg"
)

print()
print("A.2 → A.3 COMPARISON")
print("-" * 65)

drift_change = wheel_angle_drift - 24.513634

drift_change_percent = (
    abs(drift_change) / 24.513634
) * 100.0

print(
    f"A.3 wheel drift           : "
    f"{wheel_angle_drift:.6f} deg"
)

print(
    f"Drift change              : "
    f"{drift_change:+.6f} deg"
)

print(
    f"Absolute change           : "
    f"{abs(drift_change):.6f} deg"
)

print(
    f"Percentage change         : "
    f"{drift_change_percent:.2f}%"
)

print()
print("TIME HISTORY")
print("-" * 65)

print(
    "Time(s) | Wheel(deg) | Rel(deg) | "
    "Velocity | Normal(N) | Friction(N)"
)

for row in time_history:

    print(
        f"{row[0]:6.2f} | "
        f"{row[1]:10.3f} | "
        f"{row[2]:8.3f} | "
        f"{row[3]:8.4f} | "
        f"{row[4]:9.4f} | "
        f"{row[5]:10.7f}"
    )

print()
print("=" * 65)
print("END OF V3.17.8A.3")
print("=" * 65)

p.disconnect()