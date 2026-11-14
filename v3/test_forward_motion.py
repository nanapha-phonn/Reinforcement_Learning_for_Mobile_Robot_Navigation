import pybullet as p
import pybullet_data
import time
import math

import robot


# ============================================================
# V3.16 FORWARD-MOTION VALIDATION
# ============================================================

TARGET_WHEEL_VELOCITY = 4.0
MOTOR_FORCE = 20.0
DRIVE_TIME = 3.0

THEORETICAL_SPEED = robot.WHEEL_RADIUS * TARGET_WHEEL_VELOCITY
THEORETICAL_DISTANCE = THEORETICAL_SPEED * DRIVE_TIME

print("=" * 70)
print("V3.16 FORWARD-MOTION VALIDATION")
print("=" * 70)

print()
print("TEST CONDITIONS")
print("-" * 70)
print(f"Target left wheel velocity  : {TARGET_WHEEL_VELOCITY:.3f} rad/s")
print(f"Target right wheel velocity : {TARGET_WHEEL_VELOCITY:.3f} rad/s")
print(f"Motor force                 : {MOTOR_FORCE:.3f} N*m")
print(f"Drive time                  : {DRIVE_TIME:.3f} s")
print(f"Wheel radius                : {robot.WHEEL_RADIUS:.3f} m")
print(f"Wheel base                  : {robot.WHEEL_BASE:.3f} m")
print(f"Theoretical speed           : {THEORETICAL_SPEED:.3f} m/s")
print(f"Theoretical displacement     : {THEORETICAL_DISTANCE:.3f} m")

# ------------------------------------------------------------
# CONNECT
# ------------------------------------------------------------

physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, robot.GRAVITY)
p.setTimeStep(robot.TIME_STEP)

plane_id = p.loadURDF("plane.urdf")

robot_id = robot.create_robot()

print()
print(f"Robot ID: {robot_id}")

# ------------------------------------------------------------
# STATIC SETTLING
# ------------------------------------------------------------

SETTLE_TIME = 2.0
settle_steps = int(SETTLE_TIME / robot.TIME_STEP)

print()
print(f"Settling for {SETTLE_TIME:.1f} seconds...")

for _ in range(settle_steps):
    p.stepSimulation()

# ------------------------------------------------------------
# INITIAL STATE
# ------------------------------------------------------------

start_pos, start_orn = p.getBasePositionAndOrientation(robot_id)
start_lin_vel, start_ang_vel = p.getBaseVelocity(robot_id)

start_x = start_pos[0]
start_y = start_pos[1]

# ------------------------------------------------------------
# DRIVE
# ------------------------------------------------------------

drive_steps = int(DRIVE_TIME / robot.TIME_STEP)

left_velocities = []
right_velocities = []

linear_velocities = []
angular_velocities = []

print()
print("Driving robot forward...")
print()

for step in range(drive_steps):

    robot.set_wheel_velocity(
        robot_id,
        TARGET_WHEEL_VELOCITY,
        TARGET_WHEEL_VELOCITY
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

    lin_vel, ang_vel = p.getBaseVelocity(robot_id)

    left_velocities.append(left_state[1])
    right_velocities.append(right_state[1])

    linear_velocities.append(lin_vel)
    angular_velocities.append(ang_vel)

# ------------------------------------------------------------
# FINAL STATE
# ------------------------------------------------------------

final_pos, final_orn = p.getBasePositionAndOrientation(robot_id)
final_lin_vel, final_ang_vel = p.getBaseVelocity(robot_id)

final_x = final_pos[0]
final_y = final_pos[1]

dx = final_x - start_x
dy = final_y - start_y

total_displacement = math.sqrt(dx**2 + dy**2)

# ------------------------------------------------------------
# ORIENTATION
# ------------------------------------------------------------

roll, pitch, yaw = p.getEulerFromQuaternion(final_orn)

roll_deg = math.degrees(roll)
pitch_deg = math.degrees(pitch)
yaw_deg = math.degrees(yaw)

# ------------------------------------------------------------
# AVERAGE WHEEL VELOCITIES
# ------------------------------------------------------------

avg_left = sum(left_velocities) / len(left_velocities)
avg_right = sum(right_velocities) / len(right_velocities)

# ------------------------------------------------------------
# AVERAGE ROBOT VELOCITY
# ------------------------------------------------------------

avg_vx = sum(v[0] for v in linear_velocities) / len(linear_velocities)
avg_vy = sum(v[1] for v in linear_velocities) / len(linear_velocities)
avg_vz = sum(v[2] for v in linear_velocities) / len(linear_velocities)

avg_wx = sum(v[0] for v in angular_velocities) / len(angular_velocities)
avg_wy = sum(v[1] for v in angular_velocities) / len(angular_velocities)
avg_wz = sum(v[2] for v in angular_velocities) / len(angular_velocities)

# ------------------------------------------------------------
# CONTACT ANALYSIS
# ------------------------------------------------------------

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

total_normal_force = left_force + right_force + caster_force

expected_weight = (
    robot.CHASSIS_MASS
    + robot.WHEEL_MASS
    + robot.WHEEL_MASS
    + robot.CASTER_FORK_MASS
    + robot.CASTER_WHEEL_MASS
) * abs(robot.GRAVITY)

force_error = total_normal_force - expected_weight

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("=" * 70)
print("V3.16 FINAL STATE")
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
print(f"Total displacement   : {total_displacement:.6f} m")

print()
print("WHEEL VELOCITIES")
print("-" * 70)
print(f"Average left wheel   : {avg_left:.6f} rad/s")
print(f"Average right wheel  : {avg_right:.6f} rad/s")

print()
print("ROBOT VELOCITY")
print("-" * 70)
print(
    f"Average linear      : "
    f"({avg_vx:.6f}, {avg_vy:.6f}, {avg_vz:.6f}) m/s"
)

print(
    f"Average angular     : "
    f"({avg_wx:.6f}, {avg_wy:.6f}, {avg_wz:.6f}) rad/s"
)

print()
print("ORIENTATION")
print("-" * 70)
print(f"Roll                 : {roll_deg:.6f} deg")
print(f"Pitch                : {pitch_deg:.6f} deg")
print(f"Yaw                  : {yaw_deg:.6f} deg")

print()
print("CONTACTS")
print("-" * 70)
print(f"Left wheel contacts  : {left_contacts}")
print(f"Right wheel contacts : {right_contacts}")
print(f"Caster contacts      : {caster_contacts}")

print()
print("NORMAL FORCES")
print("-" * 70)
print(f"Left wheel force     : {left_force:.6f} N")
print(f"Right wheel force    : {right_force:.6f} N")
print(f"Caster force         : {caster_force:.6f} N")
print(f"Total normal force   : {total_normal_force:.6f} N")
print(f"Expected weight      : {expected_weight:.6f} N")
print(f"Force error          : {force_error:.6f} N")

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

wheel_velocity_tolerance = 0.20

left_velocity_pass = (
    abs(avg_left - TARGET_WHEEL_VELOCITY)
    <= wheel_velocity_tolerance
)

right_velocity_pass = (
    abs(avg_right - TARGET_WHEEL_VELOCITY)
    <= wheel_velocity_tolerance
)

# Require meaningful forward motion.
forward_motion_pass = dx > 0.50

# Lateral drift should remain small relative to forward travel.
lateral_drift_pass = abs(dy) < 0.10

roll_pass = abs(roll_deg) < 2.0
pitch_pass = abs(pitch_deg) < 2.0

left_contact_pass = left_contacts > 0
right_contact_pass = right_contacts > 0
caster_contact_pass = caster_contacts > 0

weight_balance_pass = abs(force_error) < 0.10

overall_pass = all([
    left_velocity_pass,
    right_velocity_pass,
    forward_motion_pass,
    lateral_drift_pass,
    roll_pass,
    pitch_pass,
    left_contact_pass,
    right_contact_pass,
    caster_contact_pass,
    weight_balance_pass
])

print()
print("=" * 70)
print("FORWARD-MOTION VALIDATION")
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
    f"Forward motion          "
    f"{'PASS' if forward_motion_pass else 'FAIL'}"
)

print(
    f"Lateral drift           "
    f"{'PASS' if lateral_drift_pass else 'FAIL'}"
)

print(
    f"Roll                    "
    f"{'PASS' if roll_pass else 'FAIL'}"
)

print(
    f"Pitch                   "
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
    f"OVERALL FORWARD MOTION: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

print("=" * 70)

p.disconnect()