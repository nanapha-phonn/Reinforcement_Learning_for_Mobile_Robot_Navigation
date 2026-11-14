import random

# --------------------------------------------------
# DEFAULT SENSOR NOISE
# --------------------------------------------------

ENCODER_ANGLE_STD = 0.002
ENCODER_VELOCITY_STD = 0.005

IMU_ANGLE_STD = 0.001
IMU_ANGULAR_VELOCITY_STD = 0.005

LIDAR_RANGE_STD = 0.01


# --------------------------------------------------
# RANDOM NUMBER GENERATOR
# --------------------------------------------------

_rng = random.Random(42)


def set_noise_seed(seed):
    """
    Set the random seed used by the sensor noise model.
    """

    _rng.seed(seed)


# --------------------------------------------------
# GAUSSIAN NOISE
# --------------------------------------------------

def add_gaussian_noise(
    value,
    standard_deviation,
):
    """
    Add zero-mean Gaussian noise to a value.
    """

    noise = _rng.gauss(
        0.0,
        standard_deviation,
    )

    return value + noise


# --------------------------------------------------
# ENCODER NOISE
# --------------------------------------------------

def add_encoder_noise(data):
    """
    Add measurement noise to encoder data.
    """

    return {
        "left_angle": add_gaussian_noise(
            data["left_angle"],
            ENCODER_ANGLE_STD,
        ),

        "right_angle": add_gaussian_noise(
            data["right_angle"],
            ENCODER_ANGLE_STD,
        ),

        "left_velocity": add_gaussian_noise(
            data["left_velocity"],
            ENCODER_VELOCITY_STD,
        ),

        "right_velocity": add_gaussian_noise(
            data["right_velocity"],
            ENCODER_VELOCITY_STD,
        ),

        "left_distance": add_gaussian_noise(
            data["left_distance"],
            ENCODER_ANGLE_STD * 0.08,
        ),

        "right_distance": add_gaussian_noise(
            data["right_distance"],
            ENCODER_ANGLE_STD * 0.08,
        ),
    }


# --------------------------------------------------
# IMU NOISE
# --------------------------------------------------

def add_imu_noise(data):
    """
    Add measurement noise to IMU data.
    """

    return {
        "roll": add_gaussian_noise(
            data["roll"],
            IMU_ANGLE_STD,
        ),

        "pitch": add_gaussian_noise(
            data["pitch"],
            IMU_ANGLE_STD,
        ),

        "yaw": add_gaussian_noise(
            data["yaw"],
            IMU_ANGLE_STD,
        ),

        "angular_vel_x": add_gaussian_noise(
            data["angular_vel_x"],
            IMU_ANGULAR_VELOCITY_STD,
        ),

        "angular_vel_y": add_gaussian_noise(
            data["angular_vel_y"],
            IMU_ANGULAR_VELOCITY_STD,
        ),

        "angular_vel_z": add_gaussian_noise(
            data["angular_vel_z"],
            IMU_ANGULAR_VELOCITY_STD,
        ),
    }


# --------------------------------------------------
# LiDAR NOISE
# --------------------------------------------------

def add_lidar_noise(data):
    """
    Add measurement noise to LiDAR ranges.

    Ranges remain physically valid:
        0 <= range <= max_range
    """

    noisy_ranges = []

    for distance in data["ranges"]:

        noisy_distance = add_gaussian_noise(
            distance,
            LIDAR_RANGE_STD,
        )

        noisy_distance = max(
            0.0,
            noisy_distance,
        )

        noisy_distance = min(
            5.0,
            noisy_distance,
        )

        noisy_ranges.append(
            noisy_distance
        )

    return {
        "ranges": noisy_ranges,
        "angles": data["angles"],
        "hit_points": data["hit_points"],
        "hit_objects": data["hit_objects"],
    }
