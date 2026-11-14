import pybullet as p


def read_encoder(robot_id, left_joint, right_joint, wheel_radius):
    """
    Read wheel encoder measurements from PyBullet.

    Returns:
        dict containing:
        - left_angle: left wheel joint angle [rad]
        - right_angle: right wheel joint angle [rad]
        - left_velocity: left wheel angular velocity [rad/s]
        - right_velocity: right wheel angular velocity [rad/s]
        - left_distance: left wheel travel [m]
        - right_distance: right wheel travel [m]
    """

    left_state = p.getJointState(robot_id, left_joint)
    right_state = p.getJointState(robot_id, right_joint)

    left_angle = left_state[0]
    left_velocity = left_state[1]

    right_angle = right_state[0]
    right_velocity = right_state[1]

    left_distance = wheel_radius * left_angle
    right_distance = wheel_radius * right_angle

    return {
        "left_angle": left_angle,
        "right_angle": right_angle,
        "left_velocity": left_velocity,
        "right_velocity": right_velocity,
        "left_distance": left_distance,
        "right_distance": right_distance,
    }