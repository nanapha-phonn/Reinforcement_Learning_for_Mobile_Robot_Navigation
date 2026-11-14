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

from sensors.lidar import read_lidar


# --------------------------------------------------
# CONNECT TO PYBULLET
# --------------------------------------------------

physics_client = p.connect(p.GUI)

if physics_client < 0:
    raise RuntimeError(
        "Failed to connect to PyBullet."
    )


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


# --------------------------------------------------
# CREATE TEST OBSTACLE
# --------------------------------------------------

# The obstacle is intentionally tall enough
# for the horizontal LiDAR ray to intersect it.

obstacle_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[
        0.10,
        0.50,
        0.30,
    ],
)

obstacle_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[
        0.10,
        0.50,
        0.30,
    ],
)

obstacle_id = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=obstacle_collision,
    baseVisualShapeIndex=obstacle_visual,
    basePosition=[
        1.00,
        0.00,
        0.30,
    ],
)


# --------------------------------------------------
# LET ROBOT SETTLE
# --------------------------------------------------

print("\n========================================")
print("V3.19.3 LiDAR TEST")
print("========================================")

print("\nLetting robot settle...")

for _ in range(240):

    p.stepSimulation()

    time.sleep(TIME_STEP)


# --------------------------------------------------
# READ LIDAR
# --------------------------------------------------

lidar = read_lidar(
    robot_id,
    num_rays=36,
    max_range=5.0,
    sensor_height=0.20,
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

ranges = lidar["ranges"]
angles = lidar["angles"]
hit_objects = lidar["hit_objects"]


print("\n--- LiDAR CONFIGURATION ---")

print(
    f"Number of rays : {len(ranges)}"
)

print(
    f"Maximum range  : 5.000000 m"
)

print(
    f"Resolution     : "
    f"{360.0 / len(ranges):.1f} degrees"
)

print(
    f"Sensor height  : 0.200000 m"
)


print("\n--- LiDAR SCAN ---")


for i in range(len(ranges)):

    angle_deg = (
        math.degrees(
            angles[i]
        )
    )

    print(
        f"Ray {i:02d} | "
        f"Angle {angle_deg:6.1f} deg | "
        f"Range {ranges[i]:.6f} m | "
        f"Object {hit_objects[i]}"
    )


# --------------------------------------------------
# FIND TEST-OBSTACLE DETECTIONS
# --------------------------------------------------

obstacle_hits = []

for i, object_id in enumerate(hit_objects):

    if object_id == obstacle_id:

        obstacle_hits.append(i)


# --------------------------------------------------
# FIND CLOSEST OBSTACLE DETECTION
# --------------------------------------------------

closest_obstacle_index = None

if obstacle_hits:

    closest_obstacle_index = min(
        obstacle_hits,
        key=lambda i: ranges[i]
    )


# --------------------------------------------------
# DISPLAY CLOSEST OBSTACLE DETECTION
# --------------------------------------------------

print("\n--- CLOSEST OBSTACLE DETECTION ---")


if closest_obstacle_index is not None:

    closest_obstacle_range = ranges[
        closest_obstacle_index
    ]

    closest_obstacle_angle = math.degrees(
        angles[closest_obstacle_index]
    )

    print(
        f"Ray index : "
        f"{closest_obstacle_index}"
    )

    print(
        f"Angle     : "
        f"{closest_obstacle_angle:.1f} deg"
    )

    print(
        f"Range     : "
        f"{closest_obstacle_range:.6f} m"
    )

    print(
        f"Object ID : "
        f"{obstacle_id}"
    )

else:

    closest_obstacle_range = 5.0

    print(
        "No test obstacle detected."
    )


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n--- VALIDATION ---")


# 1. Correct number of rays
correct_ray_count = (
    len(ranges) == 36
)


# 2. Correct number of angles
correct_angle_count = (
    len(angles) == 36
)


# 3. Every range must be finite
finite_ranges = all(
    math.isfinite(distance)
    for distance in ranges
)


# 4. Every range must be within sensor limits
valid_ranges = all(
    0.0 <= distance <= 5.0
    for distance in ranges
)


# 5. Test obstacle should be detected
obstacle_detected = (
    len(obstacle_hits) > 0
)


# 6. Closest obstacle detection should
# be physically reasonable.
#
# The obstacle begins around x = 0.90 m.
# The robot may move slightly while settling,
# so use a reasonable tolerance.

if closest_obstacle_index is not None:

    reasonable_obstacle_distance = (
        0.70
        <= closest_obstacle_range
        <= 1.10
    )

else:

    reasonable_obstacle_distance = False


print(
    f"Ray count          : "
    f"{'PASS' if correct_ray_count else 'FAIL'}"
)

print(
    f"Angle count        : "
    f"{'PASS' if correct_angle_count else 'FAIL'}"
)

print(
    f"Finite ranges      : "
    f"{'PASS' if finite_ranges else 'FAIL'}"
)

print(
    f"Range limits       : "
    f"{'PASS' if valid_ranges else 'FAIL'}"
)

print(
    f"Obstacle detected  : "
    f"{'PASS' if obstacle_detected else 'FAIL'}"
)

print(
    f"Detection distance : "
    f"{'PASS' if reasonable_obstacle_distance else 'FAIL'}"
)


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

lidar_pass = (
    correct_ray_count
    and correct_angle_count
    and finite_ranges
    and valid_ranges
    and obstacle_detected
    and reasonable_obstacle_distance
)


if lidar_pass:

    print("\n========================================")
    print("V3.19.3 LiDAR TEST: PASS")
    print("========================================")

else:

    print("\n========================================")
    print("V3.19.3 LiDAR TEST: FAIL")
    print("========================================")


# --------------------------------------------------
# CLOSE
# --------------------------------------------------

input(
    "\nPress Enter to close..."
)

p.disconnect()

