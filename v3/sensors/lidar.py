import math

import pybullet as p


def read_lidar(
    robot_id,
    num_rays=36,
    max_range=5.0,
    sensor_height=0.10,
):
    """
    Read an ideal 2D LiDAR scan using PyBullet ray casting.

    Parameters:
        robot_id:
            PyBullet robot body ID.

        num_rays:
            Number of LiDAR rays.
            Default = 36 rays.

        max_range:
            Maximum LiDAR range [m].
            Default = 5.0 m.

        sensor_height:
            LiDAR height above the robot base [m].
            Default = 0.10 m.

    Returns:
        dict containing:
            ranges       : list of distances [m]
            angles       : list of ray angles [rad]
            hit_points   : list of hit positions
            hit_objects  : list of PyBullet body IDs
    """

    # --------------------------------------------------
    # ROBOT POSE
    # --------------------------------------------------

    base_position, base_orientation = (
        p.getBasePositionAndOrientation(robot_id)
    )

    _, _, robot_yaw = p.getEulerFromQuaternion(
        base_orientation
    )


    # --------------------------------------------------
    # SENSOR POSITION
    # --------------------------------------------------

    sensor_origin = [
        base_position[0],
        base_position[1],
        base_position[2] + sensor_height,
    ]


    # --------------------------------------------------
    # CREATE RAYS
    # --------------------------------------------------

    ray_starts = []
    ray_ends = []

    angles = []

    angle_increment = (
        2.0 * math.pi / num_rays
    )

    for i in range(num_rays):

        # Sensor-relative angle
        angle = i * angle_increment

        # Convert to world angle
        world_angle = robot_yaw + angle

        start = [
            sensor_origin[0],
            sensor_origin[1],
            sensor_origin[2],
        ]

        end = [
            sensor_origin[0]
            + max_range * math.cos(world_angle),

            sensor_origin[1]
            + max_range * math.sin(world_angle),

            sensor_origin[2],
        ]

        ray_starts.append(start)
        ray_ends.append(end)
        angles.append(angle)


    # --------------------------------------------------
    # RAY CASTING
    # --------------------------------------------------

    results = p.rayTestBatch(
        ray_starts,
        ray_ends,
    )


    # --------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------

    ranges = []
    hit_points = []
    hit_objects = []

    for result in results:

        hit_object = result[0]
        hit_fraction = result[2]
        hit_position = result[3]

        if hit_object < 0:

            distance = max_range

        else:

            distance = (
                hit_fraction
                * max_range
            )

        ranges.append(distance)

        hit_points.append(
            hit_position
        )

        hit_objects.append(
            hit_object
        )


    # --------------------------------------------------
    # RETURN SENSOR DATA
    # --------------------------------------------------

    return {
        "ranges": ranges,
        "angles": angles,
        "hit_points": hit_points,
        "hit_objects": hit_objects,
    }
