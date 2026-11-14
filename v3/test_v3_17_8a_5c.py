import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_WHEEL_JOINT,
)


TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 1.0
TEST_TIME = 3.0


# ============================================================
# CONNECT
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

# ============================================================
# GRAVITY OFF
# ============================================================

p.setGravity(
    0,
    0,
    0
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
# CASTER FRICTION = 0
#
# Same as A.5B
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

for joint in [0, 1]:

    p.setJointMotorControl2(
        robot_id,
        joint,
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
# SETTLE
#
# With gravity OFF, this is mainly a consistency check.
# ============================================================

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# STATE BEFORE RESET
# ============================================================

before = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

print()
print("=" * 75)
print("V3.17.8A.5C — ZERO-VELOCITY RELEASE, GRAVITY OFF")
print("=" * 75)

print()
print("STATE AFTER SETTLING")
print("-" * 75)

print(
    f"Wheel angle     : "
    f"{math.degrees(before[0]):.9f} deg"
)

print(
    f"Wheel velocity  : "
    f"{before[1]:.12f} rad/s"
)


# ============================================================
# CONTROLLED ZERO-VELOCITY RESET
# ============================================================

p.resetJointState(
    robot_id,
    CASTER_WHEEL_JOINT,
    targetValue=0.0,
    targetVelocity=0.0
)


# ============================================================
# VERIFY RESET
# ============================================================

reset_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

initial_angle = reset_state[0]
initial_velocity = reset_state[1]


print()
print("CONTROLLED RESET")
print("-" * 75)

print(
    f"Initial angle       : "
    f"{math.degrees(initial_angle):.9f} deg"
)

print(
    f"Initial velocity    : "
    f"{initial_velocity:.12f} rad/s"
)


# ============================================================
# ARRAYS
# ============================================================

angles = []
velocities = []
accelerations = []
normal_forces = []
times = []

previous_velocity = initial_velocity

contact_samples = 0
total_samples = 0


# ============================================================
# RELEASE
# ============================================================

test_steps = int(
    TEST_TIME / TIME_STEP
)

for step in range(test_steps):

    p.stepSimulation()

    state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    angle = state[0]
    velocity = state[1]

    acceleration = (
        velocity - previous_velocity
    ) / TIME_STEP

    previous_velocity = velocity


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
        contact_samples += 1

    normal_force = sum(
        abs(contact[9])
        for contact in contacts
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    angles.append(
        math.degrees(angle)
    )

    velocities.append(
        velocity
    )

    accelerations.append(
        acceleration
    )

    normal_forces.append(
        normal_force
    )

    times.append(
        step * TIME_STEP
    )


# ============================================================
# FINAL STATE
# ============================================================

final_state = p.getJointState(
    robot_id,
    CASTER_WHEEL_JOINT
)

final_angle = final_state[0]
final_velocity = final_state[1]


# ============================================================
# RESULTS
# ============================================================

drift = math.degrees(
    final_angle - initial_angle
)

max_velocity = max(
    abs(v)
    for v in velocities
)

max_acceleration = max(
    abs(a)
    for a in accelerations
)

average_acceleration = sum(
    accelerations
) / len(accelerations)

max_normal_force = max(
    normal_forces
)

avg_normal_force = sum(
    normal_forces
) / len(normal_forces)

contact_percentage = (
    contact_samples
    / total_samples
) * 100.0


# ============================================================
# RESULTS
# ============================================================

print()
print("A.5C RESULTS")
print("-" * 75)

print(
    "Gravity                 : OFF"
)

print(
    f"Initial angle          : "
    f"{math.degrees(initial_angle):.9f} deg"
)

print(
    f"Final angle            : "
    f"{math.degrees(final_angle):.9f} deg"
)

print(
    f"Wheel drift            : "
    f"{drift:.9f} deg"
)

print(
    f"Initial velocity       : "
    f"{initial_velocity:.12f} rad/s"
)

print(
    f"Final velocity         : "
    f"{final_velocity:.12f} rad/s"
)

print(
    f"Maximum |velocity|     : "
    f"{max_velocity:.12f} rad/s"
)

print(
    f"Maximum |acceleration| : "
    f"{max_acceleration:.12f} rad/s^2"
)

print(
    f"Average acceleration   : "
    f"{average_acceleration:.12f} rad/s^2"
)

print()
print("CONTACT")
print("-" * 75)

print(
    f"Maximum normal force   : "
    f"{max_normal_force:.9f} N"
)

print(
    f"Average normal force   : "
    f"{avg_normal_force:.9f} N"
)

print(
    f"Contact percentage     : "
    f"{contact_percentage:.2f}%"
)


# ============================================================
# TIME HISTORY
# ============================================================

print()
print("TIME HISTORY")
print("-" * 75)

print(
    "Time | Angle(deg) | Velocity(rad/s) | "
    "Acceleration(rad/s²) | Normal(N)"
)

sample_interval = max(
    1,
    int(0.1 / TIME_STEP)
)

for i in range(
    0,
    len(times),
    sample_interval
):

    print(
        f"{times[i]:4.2f} | "
        f"{angles[i]:10.6f} | "
        f"{velocities[i]:16.9f} | "
        f"{accelerations[i]:20.9f} | "
        f"{normal_forces[i]:10.6f}"
    )


# ============================================================
# COMPARISON
# ============================================================

print()
print("COMPARISON")
print("-" * 75)

print(
    "A.5B gravity ON drift  : "
    "-1.245639509 deg"
)

print(
    f"A.5C gravity OFF drift : "
    f"{drift:.9f} deg"
)

print(
    f"Difference             : "
    f"{drift - (-1.245639509):.9f} deg"
)


print()
print("=" * 75)
print("END OF V3.17.8A.5C")
print("=" * 75)

p.disconnect()