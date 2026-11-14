import sys
from pathlib import Path

import pybullet as p
import pybullet_data

# Add the V3 project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robot import create_robot
from sensors.sensor_manager import SensorManager
from sensors.noise import set_noise_seed

# ---------------------------------------------------------
# Robot parameters
# ---------------------------------------------------------

WHEEL_RADIUS = 0.08

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1

TIME_STEP = 1.0 / 240.0


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def check(condition, message):
    if condition:
        print(f"{message}: PASS")
        return True

    print(f"{message}: FAIL")
    return False


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main():

    print()
    print("=== V3.19.5 SENSOR INTEGRATION TEST ===")
    print()

    physics_client = p.connect(p.DIRECT)

    try:

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(TIME_STEP)

        # -----------------------------------------------------
        # Ground
        # -----------------------------------------------------

        plane = p.loadURDF("plane.urdf")

        # -----------------------------------------------------
        # Robot
        # -----------------------------------------------------

        robot_id = create_robot()

        # -----------------------------------------------------
        # Test obstacle
        # -----------------------------------------------------

        obstacle_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.10, 0.50, 0.30],
        )

        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.10, 0.50, 0.30],
            rgbaColor=[1, 0, 0, 1],
        )

        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=[1.0, 0.0, 0.30],
        )

        # -----------------------------------------------------
        # Settle physics
        # -----------------------------------------------------

        for _ in range(240):
            p.stepSimulation()

        # -----------------------------------------------------
        # Sensor manager — ideal mode
        # -----------------------------------------------------

        ideal_manager = SensorManager(
            left_joint=LEFT_WHEEL_JOINT,
            right_joint=RIGHT_WHEEL_JOINT,
            wheel_radius=WHEEL_RADIUS,
            num_lidar_rays=36,
            lidar_max_range=5.0,
            lidar_sensor_height=0.20,
            use_noise=False,
        )

        ideal_data = ideal_manager.read(robot_id)

        # -----------------------------------------------------
        # Print ideal measurements
        # -----------------------------------------------------

        print()
        print("--- IDEAL SENSOR DATA ---")
        print()

        encoder = ideal_data["encoder"]
        imu = ideal_data["imu"]
        lidar = ideal_data["lidar"]

        print(f"Left angle     : {encoder['left_angle']:.6f} rad")
        print(f"Right angle    : {encoder['right_angle']:.6f} rad")

        print(f"Left velocity  : {encoder['left_velocity']:.6f} rad/s")
        print(f"Right velocity : {encoder['right_velocity']:.6f} rad/s")

        print(f"Roll           : {imu['roll']:.6f} rad")
        print(f"Pitch          : {imu['pitch']:.6f} rad")
        print(f"Yaw            : {imu['yaw']:.6f} rad")

        print(f"Angular vel Z  : {imu['angular_vel_z']:.6f} rad/s")

        print(f"LiDAR rays     : {len(lidar['ranges'])}")

        # -----------------------------------------------------
        # Basic structure validation
        # -----------------------------------------------------

        print()
        print("--- STRUCTURE VALIDATION ---")

        encoder_keys = {
            "left_angle",
            "right_angle",
            "left_velocity",
            "right_velocity",
            "left_distance",
            "right_distance",
        }

        imu_keys = {
            "roll",
            "pitch",
            "yaw",
            "angular_vel_x",
            "angular_vel_y",
            "angular_vel_z",
        }

        lidar_keys = {
            "ranges",
            "angles",
            "hit_points",
            "hit_objects",
        }

        check(
            encoder_keys.issubset(encoder.keys()),
            "Encoder structure",
        )

        check(
            imu_keys.issubset(imu.keys()),
            "IMU structure",
        )

        check(
            lidar_keys.issubset(lidar.keys()),
            "LiDAR structure",
        )

        check(
            len(lidar["ranges"]) == 36,
            "LiDAR ray count",
        )

        check(
            len(lidar["angles"]) == 36,
            "LiDAR angle count",
        )

        # -----------------------------------------------------
        # LiDAR obstacle validation
        # -----------------------------------------------------

        obstacle_detected = False

        for object_id, distance in zip(
            lidar["hit_objects"],
            lidar["ranges"],
        ):
            if object_id == obstacle_id:
                obstacle_detected = True

                print()
                print(
                    f"Obstacle detected at "
                    f"{distance:.6f} m"
                )

                break

        check(
            obstacle_detected,
            "LiDAR obstacle detection",
        )

        # -----------------------------------------------------
        # Direct sensor comparison
        # -----------------------------------------------------

        direct_manager = SensorManager(
            left_joint=LEFT_WHEEL_JOINT,
            right_joint=RIGHT_WHEEL_JOINT,
            wheel_radius=WHEEL_RADIUS,
            num_lidar_rays=36,
            lidar_max_range=5.0,
            lidar_sensor_height=0.20,
            use_noise=False,
        )

        direct_data = direct_manager.read(robot_id)

        encoder_same = (
            ideal_data["encoder"]
            == direct_data["encoder"]
        )

        imu_same = (
            ideal_data["imu"]
            == direct_data["imu"]
        )

        lidar_same = (
            ideal_data["lidar"]["ranges"]
            == direct_data["lidar"]["ranges"]
        )

        print()
        print("--- DETERMINISTIC IDEAL MODE ---")

        check(
            encoder_same,
            "Encoder repeatability",
        )

        check(
            imu_same,
            "IMU repeatability",
        )

        check(
            lidar_same,
            "LiDAR repeatability",
        )

        # -----------------------------------------------------
        # Noisy sensor manager
        # -----------------------------------------------------

        set_noise_seed(42)

        noisy_manager = SensorManager(
            left_joint=LEFT_WHEEL_JOINT,
            right_joint=RIGHT_WHEEL_JOINT,
            wheel_radius=WHEEL_RADIUS,
            num_lidar_rays=36,
            lidar_max_range=5.0,
            lidar_sensor_height=0.20,
            use_noise=True,
        )

        noisy_data = noisy_manager.read(robot_id)

        # -----------------------------------------------------
        # Print noisy measurements
        # -----------------------------------------------------

        print()
        print("--- NOISY SENSOR DATA ---")
        print()

        noisy_encoder = noisy_data["encoder"]
        noisy_imu = noisy_data["imu"]
        noisy_lidar = noisy_data["lidar"]

        print(
            f"Left angle     : "
            f"{noisy_encoder['left_angle']:.6f} rad"
        )

        print(
            f"Right angle    : "
            f"{noisy_encoder['right_angle']:.6f} rad"
        )

        print(
            f"Roll           : "
            f"{noisy_imu['roll']:.6f} rad"
        )

        print(
            f"Pitch          : "
            f"{noisy_imu['pitch']:.6f} rad"
        )

        print(
            f"Yaw            : "
            f"{noisy_imu['yaw']:.6f} rad"
        )

        # -----------------------------------------------------
        # Verify noise changes measurements
        # -----------------------------------------------------

        encoder_changed = (
            noisy_encoder["left_angle"]
            != ideal_data["encoder"]["left_angle"]
        )

        imu_changed = (
            noisy_imu["yaw"]
            != ideal_data["imu"]["yaw"]
        )

        lidar_changed = (
            noisy_lidar["ranges"]
            != ideal_data["lidar"]["ranges"]
        )

        print()
        print("--- NOISE VALIDATION ---")

        check(
            encoder_changed,
            "Encoder noise applied",
        )

        check(
            imu_changed,
            "IMU noise applied",
        )

        check(
            lidar_changed,
            "LiDAR noise applied",
        )

        # -----------------------------------------------------
        # Validate noisy LiDAR limits
        # -----------------------------------------------------

        lidar_limits = all(
            0.0 <= distance <= 5.0
            for distance in noisy_lidar["ranges"]
        )

        check(
            lidar_limits,
            "Noisy LiDAR range limits",
        )

        # -----------------------------------------------------
        # Final validation
        # -----------------------------------------------------

        all_passed = all([
            encoder_keys.issubset(encoder.keys()),
            imu_keys.issubset(imu.keys()),
            lidar_keys.issubset(lidar.keys()),
            len(lidar["ranges"]) == 36,
            len(lidar["angles"]) == 36,
            obstacle_detected,
            encoder_same,
            imu_same,
            lidar_same,
            encoder_changed,
            imu_changed,
            lidar_changed,
            lidar_limits,
        ])

        print()

        if all_passed:
            print(
                "V3.19.5 SENSOR INTEGRATION TEST: PASS"
            )
        else:
            print(
                "V3.19.5 SENSOR INTEGRATION TEST: FAIL"
            )

    finally:
        p.disconnect()


if __name__ == "__main__":
    main()