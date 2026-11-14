import sys
import math
from pathlib import Path


# --------------------------------------------------
# ADD PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from sensors.noise import (
    set_noise_seed,
    add_gaussian_noise,
    add_encoder_noise,
    add_imu_noise,
    add_lidar_noise,
)


# --------------------------------------------------
# TEST SETUP
# --------------------------------------------------

print("\n========================================")
print("V3.19.4 SENSOR NOISE TEST")
print("========================================")


# --------------------------------------------------
# FIXED RANDOM SEED
# --------------------------------------------------

set_noise_seed(42)


# --------------------------------------------------
# TEST GAUSSIAN NOISE
# --------------------------------------------------

print("\n--- GAUSSIAN NOISE ---")

original_value = 10.0

noisy_value = add_gaussian_noise(
    original_value,
    0.1,
)

print(
    f"Original value : "
    f"{original_value:.6f}"
)

print(
    f"Noisy value    : "
    f"{noisy_value:.6f}"
)

print(
    f"Noise           : "
    f"{noisy_value - original_value:.6f}"
)


# --------------------------------------------------
# TEST ENCODER
# --------------------------------------------------

print("\n--- ENCODER NOISE ---")

encoder_data = {
    "left_angle": 8.0,
    "right_angle": 8.0,

    "left_velocity": 4.0,
    "right_velocity": 4.0,

    "left_distance": 0.64,
    "right_distance": 0.64,
}

noisy_encoder = add_encoder_noise(
    encoder_data
)

for key in encoder_data:

    print(
        f"{key:16s} | "
        f"ideal = "
        f"{encoder_data[key]:.6f} | "
        f"noisy = "
        f"{noisy_encoder[key]:.6f}"
    )


# --------------------------------------------------
# TEST IMU
# --------------------------------------------------

print("\n--- IMU NOISE ---")

imu_data = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 1.0,

    "angular_vel_x": 0.0,
    "angular_vel_y": 0.0,
    "angular_vel_z": 1.0,
}

noisy_imu = add_imu_noise(
    imu_data
)

for key in imu_data:

    print(
        f"{key:16s} | "
        f"ideal = "
        f"{imu_data[key]:.6f} | "
        f"noisy = "
        f"{noisy_imu[key]:.6f}"
    )


# --------------------------------------------------
# TEST LIDAR
# --------------------------------------------------

print("\n--- LiDAR NOISE ---")

lidar_data = {
    "ranges": [
        0.5,
        1.0,
        2.0,
        5.0,
    ],

    "angles": [
        0.0,
        1.0,
        2.0,
        3.0,
    ],

    "hit_points": [
        [0.5, 0.0, 0.2],
        [1.0, 0.0, 0.2],
        [2.0, 0.0, 0.2],
        [5.0, 0.0, 0.2],
    ],

    "hit_objects": [
        2,
        2,
        2,
        -1,
    ],
}

noisy_lidar = add_lidar_noise(
    lidar_data
)

for i in range(
    len(lidar_data["ranges"])
):

    print(
        f"Ray {i:02d} | "
        f"ideal = "
        f"{lidar_data['ranges'][i]:.6f} | "
        f"noisy = "
        f"{noisy_lidar['ranges'][i]:.6f}"
    )


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n--- VALIDATION ---")


# 1. Encoder keys preserved
encoder_keys_ok = (
    set(encoder_data.keys())
    == set(noisy_encoder.keys())
)


# 2. IMU keys preserved
imu_keys_ok = (
    set(imu_data.keys())
    == set(noisy_imu.keys())
)


# 3. LiDAR count preserved
lidar_count_ok = (
    len(noisy_lidar["ranges"])
    == len(lidar_data["ranges"])
)


# 4. Noise actually changes measurements
encoder_changed = any(
    noisy_encoder[key]
    != encoder_data[key]
    for key in encoder_data
)


imu_changed = any(
    noisy_imu[key]
    != imu_data[key]
    for key in imu_data
)


lidar_changed = any(
    noisy_lidar["ranges"][i]
    != lidar_data["ranges"][i]
    for i in range(
        len(lidar_data["ranges"])
    )
)


# 5. LiDAR ranges remain valid
lidar_ranges_valid = all(
    0.0 <= distance <= 5.0
    for distance
    in noisy_lidar["ranges"]
)


# 6. All values remain finite
all_values_finite = (
    all(
        math.isfinite(value)
        for value in noisy_encoder.values()
    )
    and
    all(
        math.isfinite(value)
        for value in noisy_imu.values()
    )
    and
    all(
        math.isfinite(distance)
        for distance
        in noisy_lidar["ranges"]
    )
)


print(
    f"Encoder structure : "
    f"{'PASS' if encoder_keys_ok else 'FAIL'}"
)

print(
    f"IMU structure     : "
    f"{'PASS' if imu_keys_ok else 'FAIL'}"
)

print(
    f"LiDAR count       : "
    f"{'PASS' if lidar_count_ok else 'FAIL'}"
)

print(
    f"Encoder noise     : "
    f"{'PASS' if encoder_changed else 'FAIL'}"
)

print(
    f"IMU noise         : "
    f"{'PASS' if imu_changed else 'FAIL'}"
)

print(
    f"LiDAR noise       : "
    f"{'PASS' if lidar_changed else 'FAIL'}"
)

print(
    f"LiDAR limits      : "
    f"{'PASS' if lidar_ranges_valid else 'FAIL'}"
)

print(
    f"Finite values     : "
    f"{'PASS' if all_values_finite else 'FAIL'}"
)


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

sensor_noise_pass = (
    encoder_keys_ok
    and imu_keys_ok
    and lidar_count_ok
    and encoder_changed
    and imu_changed
    and lidar_changed
    and lidar_ranges_valid
    and all_values_finite
)


if sensor_noise_pass:

    print("\n========================================")
    print("V3.19.4 SENSOR NOISE TEST: PASS")
    print("========================================")

else:

    print("\n========================================")
    print("V3.19.4 SENSOR NOISE TEST: FAIL")
    print("========================================")
