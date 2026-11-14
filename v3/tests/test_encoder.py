import sys
from pathlib import Path

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import time

import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    set_wheel_velocity,
    stop_robot,
    WHEEL_RADIUS,
    LEFT_WHEEL_JOINT,
    RIGHT_WHEEL_JOINT,
    TIME_STEP,
    GRAVITY,
)

from sensors.encoder import read_encoder


# --------------------------------------------------
# CONNECT TO PYBULLET
# --------------------------------------------------

physics_client = p.connect(p.GUI)

if physics_client < 0:
    raise RuntimeError("Failed to connect to PyBullet.")


# --------------------------------------------------
# PYBULLET SETUP
# --------------------------------------------------

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, GRAVITY)

p.setTimeStep(TIME_STEP)


# --------------------------------------------------
# CREATE GROUND
# --------------------------------------------------

plane_id = p.loadURDF("plane.urdf")


# --------------------------------------------------
# CREATE V3 ROBOT
# --------------------------------------------------

robot_id = create_robot()

print("\n=== V3.19.1 ENCODER TEST ===")
print(f"Robot ID: {robot_id}")


# --------------------------------------------------
# INITIAL ENCODER READING
# --------------------------------------------------

encoder_initial = read_encoder(
    robot_id,
    LEFT_WHEEL_JOINT,
    RIGHT_WHEEL_JOINT,
    WHEEL_RADIUS,
)

print("\n--- INITIAL ENCODER ---")

print(f"Left angle      : {encoder_initial['left_angle']:.6f} rad")
print(f"Right angle     : {encoder_initial['right_angle']:.6f} rad")

print(f"Left velocity   : {encoder_initial['left_velocity']:.6f} rad/s")
print(f"Right velocity  : {encoder_initial['right_velocity']:.6f} rad/s")

print(f"Left distance   : {encoder_initial['left_distance']:.6f} m")
print(f"Right distance  : {encoder_initial['right_distance']:.6f} m")


# --------------------------------------------------
# LET ROBOT SETTLE
# --------------------------------------------------

print("\nLetting robot settle...")

for _ in range(240):

    p.stepSimulation()
    time.sleep(TIME_STEP)


# --------------------------------------------------
# DRIVE BOTH WHEELS
# --------------------------------------------------

print("\nDriving both wheels at +4 rad/s...")

set_wheel_velocity(
    robot_id,
    4.0,
    4.0,
)


# --------------------------------------------------
# RUN SIMULATION FOR 2 SECONDS
# --------------------------------------------------

for _ in range(480):

    p.stepSimulation()
    time.sleep(TIME_STEP)


# --------------------------------------------------
# STOP ROBOT
# --------------------------------------------------

stop_robot(robot_id)


# --------------------------------------------------
# READ FINAL ENCODER
# --------------------------------------------------

encoder_final = read_encoder(
    robot_id,
    LEFT_WHEEL_JOINT,
    RIGHT_WHEEL_JOINT,
    WHEEL_RADIUS,
)

print("\n--- FINAL ENCODER ---")

print(f"Left angle      : {encoder_final['left_angle']:.6f} rad")
print(f"Right angle     : {encoder_final['right_angle']:.6f} rad")

print(f"Left velocity   : {encoder_final['left_velocity']:.6f} rad/s")
print(f"Right velocity  : {encoder_final['right_velocity']:.6f} rad/s")

print(f"Left distance   : {encoder_final['left_distance']:.6f} m")
print(f"Right distance  : {encoder_final['right_distance']:.6f} m")


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

left_angle_change = (
    encoder_final["left_angle"]
    - encoder_initial["left_angle"]
)

right_angle_change = (
    encoder_final["right_angle"]
    - encoder_initial["right_angle"]
)

left_distance = encoder_final["left_distance"]
right_distance = encoder_final["right_distance"]


print("\n--- VALIDATION ---")

print(f"Left angle change  : {left_angle_change:.6f} rad")
print(f"Right angle change : {right_angle_change:.6f} rad")

print(f"Left travel        : {left_distance:.6f} m")
print(f"Right travel       : {right_distance:.6f} m")


# --------------------------------------------------
# PASS / FAIL
# --------------------------------------------------

encoder_pass = (
    abs(left_angle_change) > 0.1
    and abs(right_angle_change) > 0.1
    and abs(left_distance) > 0.01
    and abs(right_distance) > 0.01
)


if encoder_pass:
    print("\n================================")
    print("V3.19.1 ENCODER TEST: PASS")
    print("================================")
else:
    print("\n================================")
    print("V3.19.1 ENCODER TEST: FAIL")
    print("================================")


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

input("\nPress Enter to close...")

p.disconnect()