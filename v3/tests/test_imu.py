import sys
import time
import math
from pathlib import Path

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    TIME_STEP,
    GRAVITY,
)

from sensors.imu import read_imu


# --------------------------------------------------
# CONNECT TO PYBULLET
# --------------------------------------------------

physics_client = p.connect(p.GUI)

if physics_client < 0:
    raise RuntimeError("Failed to connect to PyBullet.")


# --------------------------------------------------
# PYBULLET SETUP
# --------------------------------------------------

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


# --------------------------------------------------
# CREATE GROUND
# --------------------------------------------------

plane_id = p.loadURDF(
    "plane.urdf"
)


# --------------------------------------------------
# CREATE ROBOT
# --------------------------------------------------

robot_id = create_robot()

print("\n========================================")
print("V3.19.2 IMU DIAGNOSTIC TEST")
print("========================================")

print(f"Robot ID: {robot_id}")


# --------------------------------------------------
# INITIAL STATE
# --------------------------------------------------

imu_initial = read_imu(robot_id)

position_initial, orientation_initial = (
    p.getBasePositionAndOrientation(robot_id)
)

roll_initial, pitch_initial, yaw_initial = (
    p.getEulerFromQuaternion(
        orientation_initial
    )
)


print("\n--- INITIAL STATE ---")

print(
    f"Base X : {position_initial[0]:.6f} m"
)

print(
    f"Base Y : {position_initial[1]:.6f} m"
)

print(
    f"Base Z : {position_initial[2]:.6f} m"
)

print(
    f"IMU Roll  : {imu_initial['roll']:.6f} rad"
)

print(
    f"IMU Pitch : {imu_initial['pitch']:.6f} rad"
)

print(
    f"IMU Yaw   : {imu_initial['yaw']:.6f} rad"
)

print(
    f"Base Roll  : {roll_initial:.6f} rad"
)

print(
    f"Base Pitch : {pitch_initial:.6f} rad"
)

print(
    f"Base Yaw   : {yaw_initial:.6f} rad"
)


# --------------------------------------------------
# STATIC SETTLING
# --------------------------------------------------

print("\n--- STATIC SETTLING ---")

for _ in range(240):

    p.stepSimulation()

    time.sleep(TIME_STEP)


# --------------------------------------------------
# STATE AFTER SETTLING
# --------------------------------------------------

imu_settled = read_imu(robot_id)

position_settled, orientation_settled = (
    p.getBasePositionAndOrientation(robot_id)
)

roll_settled, pitch_settled, yaw_settled = (
    p.getEulerFromQuaternion(
        orientation_settled
    )
)


print("\n--- AFTER SETTLING ---")

print(
    f"Base X : {position_settled[0]:.6f} m"
)

print(
    f"Base Y : {position_settled[1]:.6f} m"
)

print(
    f"Base Z : {position_settled[2]:.6f} m"
)

print(
    f"IMU Roll  : {imu_settled['roll']:.6f} rad"
)

print(
    f"IMU Pitch : {imu_settled['pitch']:.6f} rad"
)

print(
    f"IMU Yaw   : {imu_settled['yaw']:.6f} rad"
)

print(
    f"Base Roll  : {roll_settled:.6f} rad"
)

print(
    f"Base Pitch : {pitch_settled:.6f} rad"
)

print(
    f"Base Yaw   : {yaw_settled:.6f} rad"
)


# --------------------------------------------------
# COMPARE IMU WITH PYBULLET GROUND TRUTH
# --------------------------------------------------

print("\n--- IMU vs PYBULLET GROUND TRUTH ---")

roll_error = abs(
    imu_settled["roll"] - roll_settled
)

pitch_error = abs(
    imu_settled["pitch"] - pitch_settled
)

yaw_error = abs(
    imu_settled["yaw"] - yaw_settled
)


print(
    f"Roll error  : {roll_error:.12f} rad"
)

print(
    f"Pitch error : {pitch_error:.12f} rad"
)

print(
    f"Yaw error   : {yaw_error:.12f} rad"
)


# --------------------------------------------------
# ANGULAR VELOCITY
# --------------------------------------------------

print("\n--- ANGULAR VELOCITY ---")

print(
    f"Angular velocity X : "
    f"{imu_settled['angular_vel_x']:.6f} rad/s"
)

print(
    f"Angular velocity Y : "
    f"{imu_settled['angular_vel_y']:.6f} rad/s"
)

print(
    f"Angular velocity Z : "
    f"{imu_settled['angular_vel_z']:.6f} rad/s"
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n--- VALIDATION ---")

all_values = [
    imu_settled["roll"],
    imu_settled["pitch"],
    imu_settled["yaw"],
    imu_settled["angular_vel_x"],
    imu_settled["angular_vel_y"],
    imu_settled["angular_vel_z"],
]

finite_values = all(
    math.isfinite(value)
    for value in all_values
)


# Sensor should reproduce PyBullet orientation
# essentially exactly because this is currently
# an ideal/noiseless simulated IMU.

orientation_match = (
    roll_error < 1e-9
    and pitch_error < 1e-9
    and yaw_error < 1e-9
)


if finite_values and orientation_match:

    print("\n========================================")
    print("V3.19.2 IMU SENSOR TEST: PASS")
    print("========================================")

else:

    print("\n========================================")
    print("V3.19.2 IMU SENSOR TEST: FAIL")
    print("========================================")


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

input(
    "\nPress Enter to close..."
)

p.disconnect()