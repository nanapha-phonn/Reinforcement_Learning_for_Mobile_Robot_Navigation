import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_SWIVEL_JOINT,
    CASTER_WHEEL_JOINT,
)


# ============================================================
# V3.17.8A.5
#
# PASSIVE JOINT-DYNAMICS DIAGNOSTIC
#
# Purpose:
# Determine why the passive caster wheel joint continuously
# rotates under gravity + ground contact.
#
# This test DOES NOT change caster geometry.
# ============================================================

TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 1.0
TEST_TIME = 3.0


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


def quaternion_relative_angle(parent_quat, child_quat):

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
# CREATE ROBOT
# ============================================================

robot_id = create_robot()


# ============================================================
# ZERO CASTER FRICTION
#
# Same condition as A.3 / A.4
# ============================================================

p.changeDynamics(
    robot_id,
    CASTER_WHEEL_JOINT,
    lateralFriction=0.0,
    spinningFriction=0.0,
    rollingFriction=0.0
)


# ============================================================
# DRIVE WHEELS OFF
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


# ============================================================
# CASTER WHEEL PASSIVE
# ============================================================

p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0
)


# ============================================================
# JOINT INFORMATION
# ============================================================

print()
print("=" * 75)
print("V3.17.8A.5 — PASSIVE JOINT-DYNAMICS DIAGNOSTIC")
print("=" * 75)

print()
print("JOINT INFORMATION")
print("-" * 75)


for joint_index in [
    CASTER_SWIVEL_JOINT,
    CASTER_WHEEL_JOINT
]:

    info = p.getJointInfo(
        robot_id,
        joint_index
    )

    joint_name = info[1].decode(
        "utf-8",
        errors="replace"
    )

    joint_type = info[2]

    joint_axis = info[13]

    parent_link = info[16]

    damping = info[6]

    friction = info[7]

    lower_limit = info[8]

    upper_limit = info[9]

    print()
    print(f"Joint index        : {joint_index}")
    print(f"Joint name         : {joint_name}")
    print(f"Joint type         : {joint_type}")
    print(f"Joint axis         : {joint_axis}")
    print(f"Parent link        : {parent_link}")
    print(f"Joint damping      : {damping}")
    print(f"Joint friction     : {friction}")
    print(f"Lower limit        : {lower_limit}")
    print(f"Upper limit        : {upper_limit}")


# ============================================================
# SETTLE
# ============================================================

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# INITIAL JOINT STATES
# ============================================================

wheel_state_initial = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

swivel_state_initial = p.getJointState(
    robot_id,
    CASTER_SWIVEL_JOINT
)

initial_wheel_angle = (
    wheel_state_initial[0]
)

initial_wheel_velocity = (
    wheel_state_initial[1]
)

initial_swivel_angle = (
    swivel_state_initial[0]
)


# ============================================================
# ARRAYS
# ============================================================

wheel_angles = []
wheel_velocities = []

swivel_angles = []

applied_torques = []
reaction_torques = []

normal_forces = []
contact_torques = []

relative_angles = []

time_history = []

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


for step in range(test_steps):

    p.stepSimulation()


    # --------------------------------------------------------
    # JOINT STATES
    # --------------------------------------------------------

    wheel_state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    swivel_state = p.getJointState(
        robot_id,
        CASTER_SWIVEL_JOINT
    )


    wheel_angle = wheel_state[0]
    wheel_velocity = wheel_state[1]

    swivel_angle = swivel_state[0]


    # PyBullet:
    # jointState[3] = applied motor torque
    applied_torque = wheel_state[3]


    # jointState[2] = reaction force/torque
    reaction = wheel_state[2]

    reaction_torque_vector = [
        reaction[3],
        reaction[4],
        reaction[5]
    ]


    reaction_torque_magnitude = norm(
        reaction_torque_vector
    )


    wheel_angles.append(
        math.degrees(wheel_angle)
    )

    wheel_velocities.append(
        abs(wheel_velocity)
    )

    swivel_angles.append(
        math.degrees(swivel_angle)
    )

    applied_torques.append(
        applied_torque
    )

    reaction_torques.append(
        reaction_torque_magnitude
    )


    # --------------------------------------------------------
    # WHEEL LINK STATE
    # --------------------------------------------------------

    wheel_link = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    wheel_position = wheel_link[4]
    wheel_quaternion = wheel_link[5]


    # --------------------------------------------------------
    # FORK LINK STATE
    # --------------------------------------------------------

    fork_link = p.getLinkState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        computeForwardKinematics=True
    )

    fork_quaternion = fork_link[5]


    relative_angle = quaternion_relative_angle(
        fork_quaternion,
        wheel_quaternion
    )

    relative_angles.append(
        relative_angle
    )


    # --------------------------------------------------------
    # WHEEL AXLE
    #
    # Local Y axis
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

    axle_norm = norm(axle)

    axle = [
        value / axle_norm
        for value in axle
    ]


    # --------------------------------------------------------
    # CONTACT
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
    sample_contact_torque = 0.0


    for contact in contacts:

        contact_position = contact[5]

        normal_force = contact[9]

        normal_direction = contact[7]

        force_vector = [
            normal_direction[0] * normal_force,
            normal_direction[1] * normal_force,
            normal_direction[2] * normal_force
        ]


        r = [
            contact_position[0] - wheel_position[0],
            contact_position[1] - wheel_position[1],
            contact_position[2] - wheel_position[2]
        ]


        torque = cross(
            r,
            force_vector
        )


        axle_torque = dot(
            torque,
            axle
        )


        sample_normal_force += abs(
            normal_force
        )

        sample_contact_torque += abs(
            axle_torque
        )


    normal_forces.append(
        sample_normal_force
    )

    contact_torques.append(
        sample_contact_torque
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
                swivel_angle,
                applied_torque,
                reaction_torque_magnitude,
                sample_normal_force,
                sample_contact_torque
            )
        )


# ============================================================
# FINAL VALUES
# ============================================================

final_wheel_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

final_swivel_state = p.getJointState(
    robot_id,
    CASTER_SWIVEL_JOINT
)


final_wheel_angle = final_wheel_state[0]
final_wheel_velocity = final_wheel_state[1]

final_swivel_angle = final_swivel_state[0]


wheel_drift = math.degrees(
    final_wheel_angle
    - initial_wheel_angle
)


swivel_change = math.degrees(
    final_swivel_angle
    - initial_swivel_angle
)


max_wheel_velocity = max(
    wheel_velocities
)

avg_wheel_velocity = sum(
    wheel_velocities
) / len(wheel_velocities)


max_applied_torque = max(
    abs(value)
    for value in applied_torques
)


max_reaction_torque = max(
    reaction_torques
)

avg_reaction_torque = sum(
    reaction_torques
) / len(reaction_torques)


max_normal_force = max(
    normal_forces
)

avg_normal_force = sum(
    normal_forces
) / len(normal_forces)


max_contact_torque = max(
    contact_torques
)

avg_contact_torque = sum(
    contact_torques
) / len(contact_torques)


contact_percentage = (
    contact_count
    / total_samples
) * 100.0


# ============================================================
# RESULTS
# ============================================================

print()
print("JOINT DYNAMICS")
print("-" * 75)

print(
    f"Initial wheel angle       : "
    f"{math.degrees(initial_wheel_angle):.6f} deg"
)

print(
    f"Final wheel angle         : "
    f"{math.degrees(final_wheel_angle):.6f} deg"
)

print(
    f"Wheel angle drift         : "
    f"{wheel_drift:.6f} deg"
)

print(
    f"Final wheel velocity      : "
    f"{final_wheel_velocity:.9f} rad/s"
)

print(
    f"Maximum wheel velocity    : "
    f"{max_wheel_velocity:.9f} rad/s"
)

print(
    f"Average wheel velocity    : "
    f"{avg_wheel_velocity:.9f} rad/s"
)

print(
    f"Maximum applied torque    : "
    f"{max_applied_torque:.12f} N*m"
)

print(
    f"Maximum reaction torque   : "
    f"{max_reaction_torque:.12f} N*m"
)

print(
    f"Average reaction torque   : "
    f"{avg_reaction_torque:.12f} N*m"
)


print()
print("CASTER SWIVEL")
print("-" * 75)

print(
    f"Initial swivel angle      : "
    f"{math.degrees(initial_swivel_angle):.6f} deg"
)

print(
    f"Final swivel angle        : "
    f"{math.degrees(final_swivel_angle):.6f} deg"
)

print(
    f"Swivel change             : "
    f"{swivel_change:.6f} deg"
)


print()
print("CONTACT")
print("-" * 75)

print(
    f"Maximum normal force      : "
    f"{max_normal_force:.9f} N"
)

print(
    f"Average normal force      : "
    f"{avg_normal_force:.9f} N"
)

print(
    f"Maximum contact axle torque: "
    f"{max_contact_torque:.12f} N*m"
)

print(
    f"Average contact axle torque: "
    f"{avg_contact_torque:.12f} N*m"
)

print(
    f"Contact percentage        : "
    f"{contact_percentage:.2f}%"
)


print()
print("REFERENCE: A.3")
print("-" * 75)

print(
    "A.3 wheel drift           : "
    "33.511243 deg"
)

print(
    "A.3 max wheel velocity    : "
    "0.217067 rad/s"
)

print(
    "A.3 max axle torque       : "
    "0.000313005 N*m"
)


print()
print("TIME HISTORY")
print("-" * 75)

print(
    "Time | Wheel | Velocity | Swivel | "
    "AppliedTorque | ReactionTorque | "
    "NormalForce | ContactTorque"
)

for row in time_history:

    print(
        f"{row[0]:4.2f} | "
        f"{row[1]:8.3f} | "
        f"{row[2]:8.4f} | "
        f"{math.degrees(row[3]):8.3f} | "
        f"{row[4]:13.8f} | "
        f"{row[5]:14.8f} | "
        f"{row[6]:11.6f} | "
        f"{row[7]:13.9f}"
    )


print()
print("=" * 75)
print("END OF V3.17.8A.5")
print("=" * 75)

p.disconnect()

