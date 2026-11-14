import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_WHEEL_JOINT,
)


# ============================================================
# V3.17.8A.4
#
# NORMAL-FORCE / CONTACT-TORQUE ISOLATION
#
# Based on V3.17.8A.3:
#   caster friction = 0
#
# Purpose:
# Measure whether the normal contact force creates a
# significant torque about the caster wheel rolling axle.
#
# IMPORTANT:
# Do NOT modify robot.py.
# Do NOT modify caster geometry.
# ============================================================


TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 1.0
TEST_TIME = 3.0


def quaternion_relative_angle(parent_quat, child_quat):
    """
    Relative rotation between child and parent.
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

    _, angle = p.getAxisAngleFromQuaternion(
        relative_quat
    )

    return math.degrees(angle)


def dot(a, b):
    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
    )


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(v):
    return math.sqrt(dot(v, v))


# ============================================================
# CONNECT
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(
    0,
    0,
    -9.81
)

p.setTimeStep(
    TIME_STEP
)

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
# ROBOT
# ============================================================

robot_id = create_robot()


# ============================================================
# A.3 CONDITION
#
# ONLY experimental setting:
# caster friction = zero
# ============================================================

p.changeDynamics(
    robot_id,
    CASTER_WHEEL_JOINT,
    lateralFriction=0.0,
    spinningFriction=0.0,
    rollingFriction=0.0
)


# ============================================================
# DRIVE MOTORS OFF
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


# Caster wheel passive
p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0
)


# ============================================================
# SETTLE
# ============================================================

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# MEASUREMENT ARRAYS
# ============================================================

wheel_angles = []
wheel_velocities = []

normal_forces = []

normal_axle_torques = []

contact_offsets = []
contact_radial_distances = []

contact_x = []
contact_y = []
contact_z = []

wheel_x = []
wheel_y = []
wheel_z = []

axle_angles = []

relative_angles = []

contact_count = 0
total_samples = 0


# ============================================================
# SIMULATION
# ============================================================

test_steps = int(
    TEST_TIME / TIME_STEP
)

sample_interval = max(
    1,
    int(0.1 / TIME_STEP)
)

time_history = []


for step in range(test_steps):

    p.stepSimulation()


    # --------------------------------------------------------
    # CASTER WHEEL STATE
    # --------------------------------------------------------

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_position = wheel_state[4]
    wheel_quaternion = wheel_state[5]

    wx, wy, wz = wheel_position

    wheel_x.append(wx)
    wheel_y.append(wy)
    wheel_z.append(wz)


    # --------------------------------------------------------
    # WHEEL AXLE
    #
    # Local Y axis of caster wheel.
    # --------------------------------------------------------

    rotation_matrix = p.getMatrixFromQuaternion(
        wheel_quaternion
    )

    rotation_matrix = [
        rotation_matrix[0:3],
        rotation_matrix[3:6],
        rotation_matrix[6:9]
    ]

    axle = [
        rotation_matrix[1][0],
        rotation_matrix[1][1],
        rotation_matrix[1][2]
    ]

    axle = [
        value / norm(axle)
        for value in axle
    ]


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
    # CONTACTS
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
    sample_axle_torque = 0.0

    sample_offset = 0.0
    sample_radial_distance = 0.0


    for contact in contacts:

        # ----------------------------------------------------
        # CONTACT POINT
        # ----------------------------------------------------

        cp = contact[5]

        cx, cy, cz = cp

        contact_x.append(cx)
        contact_y.append(cy)
        contact_z.append(cz)


        # ----------------------------------------------------
        # CONTACT → WHEEL CENTER VECTOR
        # ----------------------------------------------------

        r = [
            cx - wx,
            cy - wy,
            cz - wz
        ]

        radial_distance = norm(r)

        sample_radial_distance += radial_distance


        # ----------------------------------------------------
        # NORMAL FORCE
        # ----------------------------------------------------

        normal_force = contact[9]

        sample_normal_force += abs(
            normal_force
        )


        # ----------------------------------------------------
        # NORMAL FORCE VECTOR
        #
        # contact[7] = contact normal on body B
        # ====================================================

        normal_direction = contact[7]

        F_normal = [
            normal_direction[0] * normal_force,
            normal_direction[1] * normal_force,
            normal_direction[2] * normal_force
        ]


        # ----------------------------------------------------
        # NORMAL TORQUE
        #
        # τ = r × F
        # ----------------------------------------------------

        torque = cross(
            r,
            F_normal
        )


        # ----------------------------------------------------
        # PROJECT TORQUE ONTO CASTER AXLE
        # ----------------------------------------------------

        axle_torque = dot(
            torque,
            axle
        )

        sample_axle_torque += abs(
            axle_torque
        )


        # ----------------------------------------------------
        # TORQUE-RELEVANT CONTACT OFFSET
        #
        # Distance of contact point from wheel center
        # perpendicular to wheel axle.
        # ----------------------------------------------------

        axial_component = dot(
            r,
            axle
        )

        axial_vector = [
            axle[0] * axial_component,
            axle[1] * axial_component,
            axle[2] * axial_component
        ]

        radial_vector = [
            r[0] - axial_vector[0],
            r[1] - axial_vector[1],
            r[2] - axial_vector[2]
        ]

        radial_offset = norm(
            radial_vector
        )

        sample_offset += radial_offset


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    normal_forces.append(
        sample_normal_force
    )

    normal_axle_torques.append(
        sample_axle_torque
    )

    if contacts:

        contact_offsets.append(
            sample_offset / len(contacts)
        )

        contact_radial_distances.append(
            sample_radial_distance / len(contacts)
        )


    # --------------------------------------------------------
    # TIME HISTORY
    # --------------------------------------------------------

    if step % sample_interval == 0:

        time_history.append(
            (
                step * TIME_STEP,
                math.degrees(wheel_angle),
                wheel_velocity,
                sample_normal_force,
                sample_axle_torque,
                sample_offset,
                sample_radial_distance,
            )
        )


# ============================================================
# FINAL RESULTS
# ============================================================

final_joint_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

final_wheel_angle = final_joint_state[0]


# Use last settled angle as baseline.
# The simulation has already undergone the same
# 1-second settling condition as A.3.

first_angle = wheel_angles[0]

wheel_drift = (
    wheel_angles[-1]
    - first_angle
)


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


max_axle_torque = max(
    normal_axle_torques
)

avg_axle_torque = sum(
    normal_axle_torques
) / len(normal_axle_torques)


if contact_offsets:

    max_contact_offset = max(
        contact_offsets
    )

    avg_contact_offset = sum(
        contact_offsets
    ) / len(contact_offsets)

else:

    max_contact_offset = 0.0
    avg_contact_offset = 0.0


if contact_radial_distances:

    max_radial_distance = max(
        contact_radial_distances
    )

    avg_radial_distance = sum(
        contact_radial_distances
    ) / len(contact_radial_distances)

else:

    max_radial_distance = 0.0
    avg_radial_distance = 0.0


contact_percentage = (
    contact_count
    / total_samples
) * 100.0


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 72)
print("V3.17.8A.4 — NORMAL-FORCE / CONTACT-TORQUE ISOLATION")
print("=" * 72)

print()
print("EXPERIMENT CONDITION")
print("-" * 72)

print("Gravity                    : ON")
print("Ground                     : ON")
print("Drive motors               : OFF")
print("Caster wheel friction     : 0")
print("Caster swivel              : PASSIVE")
print("Caster wheel               : PASSIVE")


print()
print("CASTER WHEEL")
print("-" * 72)

print(
    f"Initial wheel angle       : "
    f"{wheel_angles[0]:.6f} deg"
)

print(
    f"Final wheel angle         : "
    f"{wheel_angles[-1]:.6f} deg"
)

print(
    f"Wheel angle drift         : "
    f"{wheel_drift:.6f} deg"
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
print("NORMAL CONTACT FORCE")
print("-" * 72)

print(
    f"Maximum normal force      : "
    f"{max_normal_force:.9f} N"
)

print(
    f"Average normal force      : "
    f"{avg_normal_force:.9f} N"
)


print()
print("NORMAL TORQUE ABOUT WHEEL AXLE")
print("-" * 72)

print(
    f"Maximum normal axle torque: "
    f"{max_axle_torque:.12f} N*m"
)

print(
    f"Average normal axle torque: "
    f"{avg_axle_torque:.12f} N*m"
)


print()
print("CONTACT GEOMETRY")
print("-" * 72)

print(
    f"Maximum radial contact offset : "
    f"{max_contact_offset:.9f} m"
)

print(
    f"Average radial contact offset : "
    f"{avg_contact_offset:.9f} m"
)

print(
    f"Maximum contact radial distance: "
    f"{max_radial_distance:.9f} m"
)

print(
    f"Average contact radial distance: "
    f"{avg_radial_distance:.9f} m"
)


print()
print("CONTACT")
print("-" * 72)

print(
    f"Contact percentage        : "
    f"{contact_percentage:.2f}%"
)


print()
print("A.3 BASELINE")
print("-" * 72)

print(
    "A.3 wheel drift           : "
    "33.511243 deg"
)

print(
    "A.3 max wheel velocity    : "
    "0.217067 rad/s"
)

print(
    "A.3 avg normal force      : "
    "1.839528 N"
)

print(
    "A.3 max axle torque       : "
    "0.000313005 N*m"
)

print(
    "A.3 avg axle torque       : "
    "0.000191867 N*m"
)


print()
print("A.3 → A.4")
print("-" * 72)

drift_change = (
    wheel_drift - 33.511243
)

drift_change_percent = (
    abs(drift_change)
    / 33.511243
) * 100.0

print(
    f"A.4 wheel drift           : "
    f"{wheel_drift:.6f} deg"
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
print("-" * 72)

print(
    "Time(s) | Wheel(deg) | "
    "Velocity | Normal(N) | "
    "AxleTorque(Nm) | Offset(m)"
)

for row in time_history:

    print(
        f"{row[0]:6.2f} | "
        f"{row[1]:10.3f} | "
        f"{row[2]:8.4f} | "
        f"{row[3]:9.5f} | "
        f"{row[4]:14.9f} | "
        f"{row[5]:9.6f}"
    )


print()
print("=" * 72)
print("END OF V3.17.8A.4")
print("=" * 72)


p.disconnect()
