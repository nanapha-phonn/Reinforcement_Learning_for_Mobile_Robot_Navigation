import math

import pybullet as p


def quaternion_to_euler(quaternion):
    """
    Convert a PyBullet quaternion to Euler angles.

    Returns:
        roll  [rad]
        pitch [rad]
        yaw   [rad]
    """

    roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)

    return roll, pitch, yaw


def read_imu(robot_id):
    """
    Read ideal IMU measurements from the robot base.

    Measurements:
        - roll          [rad]
        - pitch         [rad]
        - yaw            [rad]
        - angular_vel_x [rad/s]
        - angular_vel_y [rad/s]
        - angular_vel_z [rad/s]

    Returns:
        dict containing the IMU measurements.
    """

    # --------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------

    position, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = quaternion_to_euler(
        orientation
    )


    # --------------------------------------------------
    # ANGULAR VELOCITY
    # --------------------------------------------------

    linear_velocity, angular_velocity = p.getBaseVelocity(
        robot_id
    )

    angular_vel_x = angular_velocity[0]
    angular_vel_y = angular_velocity[1]
    angular_vel_z = angular_velocity[2]


    # --------------------------------------------------
    # RETURN IMU MEASUREMENT
    # --------------------------------------------------

    return {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "angular_vel_x": angular_vel_x,
        "angular_vel_y": angular_vel_y,
        "angular_vel_z": angular_vel_z,
    }