import pybullet as p
import pybullet_data
import math

import robot


# ============================================================
# V3.17 PURE-ROTATION VALIDATION
# ============================================================

LEFT_COMMAND = -4.0
RIGHT_COMMAND = 4.0

MOTOR_FORCE = 20.0
ROTATION_TIME = 3.0

THEORETICAL_YAW_RATE = (
    robot.WHEEL_RADIUS
    * (RIGHT_COMMAND - LEFT_COMMAND)
    / robot.WHEEL_BASE
)

THEORETICAL_YAW_CHANGE = (
    THEORETICAL_YAW_RATE * ROTATION_TIME
)

print("=" * 70)
print("V3.17 PURE-ROTATION VALIDATION")
print("=" * 70)

print()
print("TEST CONDITIONS")
print("-" * 70)
print(f"Left wheel command       : {LEFT_COMMAND:.3f} rad/s")
print(f"Right wheel command      : {RIGHT_COMMAND:.3f} rad/s")
print(f"Motor force              : {MOTOR_FORCE:.3f} N*m")
print(f"Rotation time            : {ROTATION_TIME:.3f} s")
print(f"Wheel radius             : {robot.WHEEL_RADIUS:.3f} m")
print(f"Wheel base               : {robot.WHEEL_BASE:.3f} m")
print(f"Theoretical yaw rate     : {THEORETICAL_YAW_RATE:.6f} rad/s")
print(f"Theoretical yaw change   : {THEORETICAL_YAW_CHANGE:.6f} rad")
print(
    f"Theoretical yaw change   : "
    f"{math.degrees(THEORETICAL_YAW_CHANGE):.3f} deg"
)


# ============================================================
# CONNECT
# ============================================================

physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(
    0,
    0,
    robot.GRAVITY
)

p.setTimeStep(robot.TIME_STEP)

plane_id = p.loadURDF("plane.urdf")

robot_id = robot.create_robot()

print()
print(f"Robot ID: {robot_id}")


# ============================================================
# SETTLE
# ============================================================

SETTLE_TIME = 2.0
settle_steps = int(
    SETTLE_TIME / robot.TIME_STEP
)

print()
print(
    f"Settling for {SETTLE_TIME:.1f} seconds..."
)

for _ in range(settle_steps):
    p.stepSimulation()


# ============================================================
# INITIAL STATE
# ============================================================

start_pos, start_orn = (
    p.getBasePositionAndOrientation(robot_id)
)

start_x = start_pos[0]
start_y = start_pos[1]

start_roll, start_pitch, start_yaw = (
    p.getEulerFromQuaternion(start_orn)
)


# ============================================================
# ROTATION
# ============================================================

steps = int(
    ROTATION_TIME / robot.TIME_STEP
)

left_velocities = []
right_velocities = []

yaw_rates = []
roll_values = []
pitch_values = []

print()
print("Rotating robot...")
print()

for step in range(steps):

    robot.set_wheel_velocity(
        robot_id,
        LEFT_COMMAND,
        RIGHT_COMMAND
    )

    p.stepSimulation()

    left_state = p.getJointState(
        robot_id,
        robot.LEFT_WHEEL_JOINT
    )

    right_state = p.getJointState(
        robot_id,
        robot.RIGHT_WHEEL_JOINT
    )

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    linear_velocity, angular_velocity = (
        p.getBaseVelocity(robot_id)
    )

    left_velocities.append(
        left_state[1]
    )

    right_velocities.append(
        right_state[1]
    )

    yaw_rates.append(
        angular_velocity[2]
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    roll_values.append(
        math.degrees(roll)
    )

    pitch_values.append(
        math.degrees(pitch)
    )


# ============================================================
# FINAL STATE
# ============================================================

final_pos, final_orn = (
    p.getBasePositionAndOrientation(
        robot_id
    )
)

final_lin_vel, final_ang_vel = (
    p.getBaseVelocity(robot_id)
)

final_x = final_pos[0]
final_y = final_pos[1]

dx = final_x - start_x
dy = final_y - start_y

final_roll, final_pitch, final_yaw = (
    p.getEulerFromQuaternion(
        final_orn
    )
)


# ============================================================
# WHEEL VELOCITIES
# ============================================================

avg_left = (
    sum(left_velocities)
    / len(left_velocities)
)

avg_right = (
    sum(right_velocities)
    / len(right_velocities)
)

final_left = left_velocities[-1]
final_right = right_velocities[-1]


# ============================================================
# YAW RATE
# ============================================================

avg_yaw_rate = (
    sum(yaw_rates)
    / len(yaw_rates)
)

final_yaw_rate = final_ang_vel[2]

rotation_ratio = (
    abs(avg_yaw_rate)
    / abs(THEORETICAL_YAW_RATE)
    * 100.0
)


# ============================================================
# ORIENTATION
# ============================================================

final_roll_deg = math.degrees(
    final_roll
)

final_pitch_deg = math.degrees(
    final_pitch
)

final_yaw_deg = math.degrees(
    final_yaw
)

# Convert yaw difference to an unwrapped estimate
# using the integrated measured yaw rate.

measured_yaw_change = (
    avg_yaw_rate * ROTATION_TIME
)

measured_yaw_change_deg = math.degrees(
    measured_yaw_change
)

yaw_change_ratio = (
    abs(measured_yaw_change)
    / abs(THEORETICAL_YAW_CHANGE)
    * 100.0
)


# ============================================================
# MAX ROLL / PITCH
# ============================================================

max_roll = max(
    abs(value)
    for value in roll_values
)

max_pitch = max(
    abs(value)
    for value in pitch_values
)


# ============================================================
# CONTACT ANALYSIS
# ============================================================

contacts = p.getContactPoints(
    bodyA=robot_id,
    bodyB=plane_id
)

left_contacts = 0
right_contacts = 0
caster_contacts = 0

left_force = 0.0
right_force = 0.0
caster_force = 0.0

for contact in contacts:

    link_index = contact[3]

    normal_force = contact[9]

    if link_index == robot.LEFT_WHEEL_JOINT:

        left_contacts += 1
        left_force += normal_force

    elif link_index == robot.RIGHT_WHEEL_JOINT:

        right_contacts += 1
        right_force += normal_force

    elif link_index == robot.CASTER_WHEEL_JOINT:

        caster_contacts += 1
        caster_force += normal_force


total_normal_force = (
    left_force
    + right_force
    + caster_force
)

expected_weight = (
    robot.CHASSIS_MASS
    + robot.WHEEL_MASS
    + robot.WHEEL_MASS
    + robot.CASTER_FORK_MASS
    + robot.CASTER_WHEEL_MASS
) * abs(robot.GRAVITY)

force_error = (
    total_normal_force
    - expected_weight
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("V3.17 FINAL STATE")
print("=" * 70)

print()
print("POSITION")
print("-" * 70)
print(f"Start X              : {start_x:.6f} m")
print(f"Start Y              : {start_y:.6f} m")
print(f"Final X              : {final_x:.6f} m")
print(f"Final Y              : {final_y:.6f} m")
print(f"X displacement       : {dx:.6f} m")
print(f"Y displacement       : {dy:.6f} m")

print()
print("WHEEL VELOCITIES")
print("-" * 70)
print(
    f"Average left wheel   : "
    f"{avg_left:.6f} rad/s"
)

print(
    f"Average right wheel  : "
    f"{avg_right:.6f} rad/s"
)

print(
    f"Final left wheel     : "
    f"{final_left:.6f} rad/s"
)

print(
    f"Final right wheel    : "
    f"{final_right:.6f} rad/s"
)

print()
print("ROTATION")
print("-" * 70)

print(
    f"Theoretical yaw rate : "
    f"{THEORETICAL_YAW_RATE:.6f} rad/s"
)

print(
    f"Average yaw rate     : "
    f"{avg_yaw_rate:.6f} rad/s"
)

print(
    f"Final yaw rate       : "
    f"{final_yaw_rate:.6f} rad/s"
)

print(
    f"Rotation ratio       : "
    f"{rotation_ratio:.2f}%"
)

print(
    f"Theoretical yaw Δ    : "
    f"{THEORETICAL_YAW_CHANGE:.6f} rad"
)

print(
    f"Measured yaw Δ*      : "
    f"{measured_yaw_change:.6f} rad"
)

print(
    f"Yaw-change ratio*    : "
    f"{yaw_change_ratio:.2f}%"
)

print("*Estimated by integrating average yaw rate.")

print()
print("ORIENTATION")
print("-" * 70)
print(
    f"Final roll           : "
    f"{final_roll_deg:.6f} deg"
)

print(
    f"Final pitch          : "
    f"{final_pitch_deg:.6f} deg"
)

print(
    f"Final yaw            : "
    f"{final_yaw_deg:.6f} deg"
)

print(
    f"Maximum |roll|       : "
    f"{max_roll:.6f} deg"
)

print(
    f"Maximum |pitch|      : "
    f"{max_pitch:.6f} deg"
)

print()
print("CONTACTS")
print("-" * 70)
print(
    f"Left wheel contacts  : "
    f"{left_contacts}"
)

print(
    f"Right wheel contacts : "
    f"{right_contacts}"
)

print(
    f"Caster contacts      : "
    f"{caster_contacts}"
)

print()
print("NORMAL FORCES")
print("-" * 70)
print(
    f"Left wheel force     : "
    f"{left_force:.6f} N"
)

print(
    f"Right wheel force    : "
    f"{right_force:.6f} N"
)

print(
    f"Caster force         : "
    f"{caster_force:.6f} N"
)

print(
    f"Total normal force   : "
    f"{total_normal_force:.6f} N"
)

print(
    f"Expected weight      : "
    f"{expected_weight:.6f} N"
)

print(
    f"Force error          : "
    f"{force_error:.6f} N"
)


# ============================================================
# VALIDATION
# ============================================================

wheel_velocity_tolerance = 0.20

left_velocity_pass = (
    abs(abs(avg_left) - abs(LEFT_COMMAND))
    <= wheel_velocity_tolerance
)

right_velocity_pass = (
    abs(abs(avg_right) - abs(RIGHT_COMMAND))
    <= wheel_velocity_tolerance
)

# Require at least 90% of theoretical yaw rate.
rotation_pass = (
    rotation_ratio >= 90.0
)

# Pure rotation should produce very little translation.
translation_pass = (
    math.sqrt(dx**2 + dy**2) < 0.10
)

roll_pass = (
    max_roll < 2.0
)

pitch_pass = (
    max_pitch < 2.0
)

left_contact_pass = (
    left_contacts > 0
)

right_contact_pass = (
    right_contacts > 0
)

caster_contact_pass = (
    caster_contacts > 0
)

weight_balance_pass = (
    abs(force_error) < 0.10
)

overall_pass = all([
    left_velocity_pass,
    right_velocity_pass,
    rotation_pass,
    translation_pass,
    roll_pass,
    pitch_pass,
    left_contact_pass,
    right_contact_pass,
    caster_contact_pass,
    weight_balance_pass
])


# ============================================================
# VALIDATION REPORT
# ============================================================

print()
print("=" * 70)
print("PURE-ROTATION VALIDATION")
print("=" * 70)

print(
    f"Left wheel velocity     "
    f"{'PASS' if left_velocity_pass else 'FAIL'}"
)

print(
    f"Right wheel velocity    "
    f"{'PASS' if right_velocity_pass else 'FAIL'}"
)

print(
    f"Rotation ratio          "
    f"{'PASS' if rotation_pass else 'FAIL'}"
)

print(
    f"Minimal translation     "
    f"{'PASS' if translation_pass else 'FAIL'}"
)

print(
    f"Roll stability          "
    f"{'PASS' if roll_pass else 'FAIL'}"
)

print(
    f"Pitch stability         "
    f"{'PASS' if pitch_pass else 'FAIL'}"
)

print(
    f"Left wheel contact      "
    f"{'PASS' if left_contact_pass else 'FAIL'}"
)

print(
    f"Right wheel contact     "
    f"{'PASS' if right_contact_pass else 'FAIL'}"
)

print(
    f"Caster contact          "
    f"{'PASS' if caster_contact_pass else 'FAIL'}"
)

print(
    f"Weight balance          "
    f"{'PASS' if weight_balance_pass else 'FAIL'}"
)

print()
print(
    f"OVERALL PURE ROTATION: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

print("=" * 70)

p.disconnect()