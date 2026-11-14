import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    WHEEL_RADIUS,
    WHEEL_BASE,
    TIME_STEP,
    MOTOR_FORCE,
    CASTER_WHEEL_JOINT,
)


# ============================================================
# V3.17.8C
# FULL DIFFERENTIAL-DRIVE YAW
# ============================================================

LEFT_WHEEL = 0
RIGHT_WHEEL = 1

LEFT_COMMAND = -2.0
RIGHT_COMMAND = 2.0

SETTLE_TIME = 1.0
TEST_TIME = 3.0


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
# CASTER
#
# Keep validated configuration.
# Do NOT modify geometry.
# ============================================================

p.changeDynamics(
    robot_id,
    CASTER_WHEEL_JOINT,
    lateralFriction=0.0,
    spinningFriction=0.0,
    rollingFriction=0.0
)


# ============================================================
# CASTER PASSIVE
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
# Drive motors OFF during settling.
# ============================================================

for joint in [LEFT_WHEEL, RIGHT_WHEEL]:

    p.setJointMotorControl2(
        robot_id,
        joint,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )


settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# INITIAL STATE
# ============================================================

base_initial = p.getBasePositionAndOrientation(
    robot_id
)

initial_position = base_initial[0]
initial_orientation = base_initial[1]

initial_euler = p.getEulerFromQuaternion(
    initial_orientation
)

initial_yaw = initial_euler[2]


# ============================================================
# THEORETICAL VALUES
# ============================================================

theoretical_yaw_rate = (
    WHEEL_RADIUS
    * (RIGHT_COMMAND - LEFT_COMMAND)
    / WHEEL_BASE
)

theoretical_yaw_change = (
    theoretical_yaw_rate
    * TEST_TIME
)


print()
print("=" * 80)
print("V3.17.8C — FULL DIFFERENTIAL-DRIVE YAW")
print("=" * 80)

print()
print("COMMAND")
print("-" * 80)

print(
    f"Left wheel command      : "
    f"{LEFT_COMMAND:.6f} rad/s"
)

print(
    f"Right wheel command     : "
    f"{RIGHT_COMMAND:.6f} rad/s"
)

print(
    f"Wheel radius            : "
    f"{WHEEL_RADIUS:.6f} m"
)

print(
    f"Wheel base              : "
    f"{WHEEL_BASE:.6f} m"
)


print()
print("THEORETICAL")
print("-" * 80)

print(
    f"Theoretical yaw rate    : "
    f"{theoretical_yaw_rate:.6f} rad/s"
)

print(
    f"Theoretical yaw change  : "
    f"{theoretical_yaw_change:.6f} rad"
)

print(
    f"Theoretical yaw change  : "
    f"{math.degrees(theoretical_yaw_change):.6f} deg"
)


# ============================================================
# ARRAYS
# ============================================================

left_velocities = []
right_velocities = []

yaw_rates = []

roll_values = []
pitch_values = []
yaw_values = []

left_contacts = []
right_contacts = []
caster_contacts = []

left_forces = []
right_forces = []
caster_forces = []

times = []


# ============================================================
# APPLY DRIVE
# ============================================================

p.setJointMotorControl2(
    robot_id,
    LEFT_WHEEL,
    p.VELOCITY_CONTROL,
    targetVelocity=LEFT_COMMAND,
    force=MOTOR_FORCE
)

p.setJointMotorControl2(
    robot_id,
    RIGHT_WHEEL,
    p.VELOCITY_CONTROL,
    targetVelocity=RIGHT_COMMAND,
    force=MOTOR_FORCE
)


# ============================================================
# SIMULATION
# ============================================================

test_steps = int(
    TEST_TIME / TIME_STEP
)

for step in range(test_steps):

    p.stepSimulation()


    # --------------------------------------------------------
    # JOINT STATES
    # --------------------------------------------------------

    left_state = p.getJointState(
        robot_id,
        LEFT_WHEEL
    )

    right_state = p.getJointState(
        robot_id,
        RIGHT_WHEEL
    )

    left_velocity = left_state[1]
    right_velocity = right_state[1]


    left_velocities.append(
        left_velocity
    )

    right_velocities.append(
        right_velocity
    )


    # --------------------------------------------------------
    # BASE STATE
    # --------------------------------------------------------

    base_position, base_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        base_orientation
    )

    angular_velocity = (
        p.getBaseVelocity(robot_id)[1]
    )

    yaw_rate = angular_velocity[2]


    yaw_rates.append(
        yaw_rate
    )

    roll_values.append(
        math.degrees(roll)
    )

    pitch_values.append(
        math.degrees(pitch)
    )

    yaw_values.append(
        math.degrees(yaw)
    )


    # --------------------------------------------------------
    # CONTACTS
    # --------------------------------------------------------

    left_contact_points = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=LEFT_WHEEL
    )

    right_contact_points = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=RIGHT_WHEEL
    )

    caster_contact_points = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=CASTER_WHEEL_JOINT
    )


    left_contacts.append(
        1 if left_contact_points else 0
    )

    right_contacts.append(
        1 if right_contact_points else 0
    )

    caster_contacts.append(
        1 if caster_contact_points else 0
    )


    # --------------------------------------------------------
    # NORMAL FORCES
    # --------------------------------------------------------

    left_force = sum(
        abs(contact[9])
        for contact in left_contact_points
    )

    right_force = sum(
        abs(contact[9])
        for contact in right_contact_points
    )

    caster_force = sum(
        abs(contact[9])
        for contact in caster_contact_points
    )


    left_forces.append(
        left_force
    )

    right_forces.append(
        right_force
    )

    caster_forces.append(
        caster_force
    )


    times.append(
        step * TIME_STEP
    )


# ============================================================
# FINAL STATE
# ============================================================

final_position, final_orientation = (
    p.getBasePositionAndOrientation(
        robot_id
    )
)

final_roll, final_pitch, final_yaw = (
    p.getEulerFromQuaternion(
        final_orientation
    )
)

final_angular_velocity = (
    p.getBaseVelocity(robot_id)[1]
)

final_yaw_rate = final_angular_velocity[2]


# ============================================================
# AVERAGES
# ============================================================

average_left = sum(
    left_velocities
) / len(left_velocities)

average_right = sum(
    right_velocities
) / len(right_velocities)

average_yaw_rate = sum(
    yaw_rates
) / len(yaw_rates)


# ============================================================
# YAW CHANGE
# ============================================================

# Use integrated yaw rate instead of Euler yaw because
# Euler yaw wraps around at ±180 degrees.

yaw_change = sum(
    yaw_rates
) * TIME_STEP


# ============================================================
# ROTATION RATIO
# ============================================================

rotation_ratio = (
    abs(average_yaw_rate)
    / abs(theoretical_yaw_rate)
) * 100.0

yaw_change_ratio = (
    abs(yaw_change)
    / abs(theoretical_yaw_change)
) * 100.0


# ============================================================
# CONTACT PERCENTAGES
# ============================================================

left_contact_percentage = (
    sum(left_contacts)
    / len(left_contacts)
) * 100.0

right_contact_percentage = (
    sum(right_contacts)
    / len(right_contacts)
) * 100.0

caster_contact_percentage = (
    sum(caster_contacts)
    / len(caster_contacts)
) * 100.0


# ============================================================
# NORMAL FORCES
# ============================================================

average_left_force = (
    sum(left_forces)
    / len(left_forces)
)

average_right_force = (
    sum(right_forces)
    / len(right_forces)
)

average_caster_force = (
    sum(caster_forces)
    / len(caster_forces)
)

average_total_force = (
    average_left_force
    + average_right_force
    + average_caster_force
)


# ============================================================
# RESULTS
# ============================================================

print()
print("WHEEL VELOCITIES")
print("-" * 80)

print(
    f"Average left wheel    : "
    f"{average_left:.6f} rad/s"
)

print(
    f"Average right wheel   : "
    f"{average_right:.6f} rad/s"
)


print()
print("YAW")
print("-" * 80)

print(
    f"Theoretical yaw rate  : "
    f"{theoretical_yaw_rate:.6f} rad/s"
)

print(
    f"Average yaw rate      : "
    f"{average_yaw_rate:.6f} rad/s"
)

print(
    f"Final yaw rate        : "
    f"{final_yaw_rate:.6f} rad/s"
)

print(
    f"Rotation ratio        : "
    f"{rotation_ratio:.3f}%"
)

print(
    f"Yaw change            : "
    f"{yaw_change:.6f} rad"
)

print(
    f"Yaw change ratio      : "
    f"{yaw_change_ratio:.3f}%"
)


print()
print("ATTITUDE")
print("-" * 80)

print(
    f"Final roll            : "
    f"{math.degrees(final_roll):.6f} deg"
)

print(
    f"Final pitch           : "
    f"{math.degrees(final_pitch):.6f} deg"
)

print(
    f"Final yaw             : "
    f"{math.degrees(final_yaw):.6f} deg"
)

print(
    f"Maximum |roll|        : "
    f"{max(abs(x) for x in roll_values):.6f} deg"
)

print(
    f"Maximum |pitch|       : "
    f"{max(abs(x) for x in pitch_values):.6f} deg"
)


print()
print("CONTACT")
print("-" * 80)

print(
    f"Left wheel contacts   : "
    f"{left_contact_percentage:.2f}%"
)

print(
    f"Right wheel contacts  : "
    f"{right_contact_percentage:.2f}%"
)

print(
    f"Caster contacts       : "
    f"{caster_contact_percentage:.2f}%"
)


print()
print("NORMAL FORCES")
print("-" * 80)

print(
    f"Average left force    : "
    f"{average_left_force:.6f} N"
)

print(
    f"Average right force   : "
    f"{average_right_force:.6f} N"
)

print(
    f"Average caster force  : "
    f"{average_caster_force:.6f} N"
)

print(
    f"Average total force   : "
    f"{average_total_force:.6f} N"
)


print()
print("VALIDATION")
print("-" * 80)

yaw_pass = rotation_ratio >= 95.0

attitude_pass = (
    max(abs(x) for x in roll_values) < 2.0
    and
    max(abs(x) for x in pitch_values) < 2.0
)

wheel_contact_pass = (
    left_contact_percentage >= 95.0
    and
    right_contact_percentage >= 95.0
)

caster_contact_pass = (
    caster_contact_percentage >= 95.0
)


print(
    f"Yaw tracking          : "
    f"{'PASS' if yaw_pass else 'FAIL'}"
)

print(
    f"Roll/Pitch stability  : "
    f"{'PASS' if attitude_pass else 'FAIL'}"
)

print(
    f"Drive wheel contact   : "
    f"{'PASS' if wheel_contact_pass else 'FAIL'}"
)

print(
    f"Caster contact        : "
    f"{'PASS' if caster_contact_pass else 'FAIL'}"
)


print()
print("=" * 80)
print("V3.17.8D — MODERATE-SPEED DIFFERENTIAL-DRIVE YAW")
print("=" * 80)

p.disconnect()