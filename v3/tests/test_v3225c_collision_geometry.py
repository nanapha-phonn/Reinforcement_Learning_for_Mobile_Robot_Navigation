import math
import os
import sys

import pybullet as p
import pybullet_data


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from robot import (
    create_robot,
    WHEEL_RADIUS,
    CASTER_WHEEL_RADIUS,
    CASTER_WHEEL_WIDTH,
    CASTER_X,
)


# ============================================================
# CONSTANTS
# ============================================================

GRAVITY = -9.81

CHASSIS_HEIGHT = 0.10

RESET_BASE_Z = (
    WHEEL_RADIUS
    + CHASSIS_HEIGHT / 2.0
)

CASTER_FORK_LINK = 2
CASTER_WHEEL_LINK = 3

DEG = 180.0 / math.pi

EXPECTED_CASTER_RADIUS = 0.04
EXPECTED_CASTER_WIDTH = 0.025


# ============================================================
# CONNECT
# ============================================================

def connect_physics():

    if p.isConnected():
        p.disconnect()

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        GRAVITY,
    )

    print(
        "PyBullet connection: PASS"
    )

    print(
        "Physics client ID :",
        client_id,
    )


# ============================================================
# VECTOR UTILITIES
# ============================================================

def rotate_vector(
    quaternion,
    vector,
):
    """
    Rotate a local vector into world coordinates.
    """

    return p.rotateVector(
        quaternion,
        vector,
    )


def vector_length(vector):

    return math.sqrt(
        vector[0] ** 2
        + vector[1] ** 2
        + vector[2] ** 2
    )


def normalize(vector):

    length = vector_length(
        vector
    )

    if length < 1e-12:
        return [0.0, 0.0, 0.0]

    return [
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    ]


# ============================================================
# GEOMETRY INFORMATION
# ============================================================

def print_collision_shape_info(
    robot_id,
):

    print()
    print("=" * 100)
    print(
        "COLLISION SHAPE DATA"
    )
    print("=" * 100)

    num_shapes = p.getNumJoints(
        robot_id
    )

    print(
        "Number of links:",
        num_shapes,
    )

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base_shapes = p.getCollisionShapeData(
        robot_id,
        -1,
    )

    print()
    print(
        "--- BASE COLLISION ---"
    )

    for shape in base_shapes:

        print(
            "Shape type        :",
            shape[2],
        )

        print(
            "Dimensions        :",
            shape[3],
        )

        print(
            "Local position    :",
            shape[5],
        )

        print(
            "Local orientation :",
            shape[6],
        )

    # --------------------------------------------------------
    # Links
    # --------------------------------------------------------

    for link_index in range(
        p.getNumJoints(robot_id)
    ):

        shapes = p.getCollisionShapeData(
            robot_id,
            link_index,
        )

        print()
        print(
            f"--- LINK {link_index} COLLISION ---"
        )

        for shape in shapes:

            print(
                "Shape type        :",
                shape[2],
            )

            print(
                "Dimensions        :",
                shape[3],
            )

            print(
                "Local position    :",
                shape[5],
            )

            print(
                "Local orientation :",
                shape[6],
            )


# ============================================================
# LINK STATE
# ============================================================

def get_link_state(
    robot_id,
    link_index,
):

    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True,
    )

    return {
        "world_com_position": state[0],
        "world_com_orientation": state[1],
        "world_link_position": state[4],
        "world_link_orientation": state[5],
    }


# ============================================================
# PRINT LINK STATE
# ============================================================

def print_link_state(
    robot_id,
    link_index,
    name,
):

    data = get_link_state(
        robot_id,
        link_index,
    )

    position = data[
        "world_link_position"
    ]

    orientation = data[
        "world_link_orientation"
    ]

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    print()
    print(
        f"--- {name} ---"
    )

    print(
        "World position:"
    )

    print(
        f"  X = {position[0]:.6f} m"
    )

    print(
        f"  Y = {position[1]:.6f} m"
    )

    print(
        f"  Z = {position[2]:.6f} m"
    )

    print(
        "World Euler:"
    )

    print(
        f"  Roll  = {roll * DEG:.6f} deg"
    )

    print(
        f"  Pitch = {pitch * DEG:.6f} deg"
    )

    print(
        f"  Yaw   = {yaw * DEG:.6f} deg"
    )


# ============================================================
# CASTER WHEEL AXIS
# ============================================================

def inspect_wheel_axis(
    robot_id,
):

    data = get_link_state(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    orientation = data[
        "world_link_orientation"
    ]

    # Wheel joint axis is Y.
    local_axis = [
        0.0,
        1.0,
        0.0,
    ]

    world_axis = rotate_vector(
        orientation,
        local_axis,
    )

    world_axis = normalize(
        world_axis
    )

    print()
    print(
        "--- CASTER WHEEL AXIS ---"
    )

    print(
        "Local wheel axis : [0, 1, 0]"
    )

    print(
        "World wheel axis :"
    )

    print(
        f"  X = {world_axis[0]: .6f}"
    )

    print(
        f"  Y = {world_axis[1]: .6f}"
    )

    print(
        f"  Z = {world_axis[2]: .6f}"
    )

    print(
        "Axis magnitude   :",
        f"{vector_length(world_axis):.6f}",
    )

    return world_axis


# ============================================================
# LOWEST POINT OF CYLINDER
# ============================================================

def estimate_lowest_wheel_point(
    robot_id,
):

    """
    Estimate the lowest point of the caster wheel.

    The wheel is modeled as a cylinder whose local axis
    is Y and whose radius is CASTER_WHEEL_RADIUS.

    For the zero-angle geometry, the radial direction
    with the largest downward world-Z component gives
    the lowest radial point.

    This is a geometric estimate, not a contact query.
    """

    data = get_link_state(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    center = data[
        "world_link_position"
    ]

    orientation = data[
        "world_link_orientation"
    ]

    axis = normalize(
        rotate_vector(
            orientation,
            [0.0, 1.0, 0.0],
        )
    )

    # Local radial basis vectors.
    local_u = [
        1.0,
        0.0,
        0.0,
    ]

    local_v = [
        0.0,
        0.0,
        1.0,
    ]

    u = normalize(
        rotate_vector(
            orientation,
            local_u,
        )
    )

    v = normalize(
        rotate_vector(
            orientation,
            local_v,
        )
    )

    # Search around the circular cross-section.
    samples = 720

    lowest_point = None
    lowest_z = float("inf")
    lowest_angle = None

    for i in range(samples):

        theta = (
            2.0
            * math.pi
            * i
            / samples
        )

        radial = [
            math.cos(theta) * u[0]
            + math.sin(theta) * v[0],

            math.cos(theta) * u[1]
            + math.sin(theta) * v[1],

            math.cos(theta) * u[2]
            + math.sin(theta) * v[2],
        ]

        point = [
            center[0]
            + CASTER_WHEEL_RADIUS
            * radial[0],

            center[1]
            + CASTER_WHEEL_RADIUS
            * radial[1],

            center[2]
            + CASTER_WHEEL_RADIUS
            * radial[2],
        ]

        if point[2] < lowest_z:

            lowest_z = point[2]

            lowest_point = point

            lowest_angle = (
                theta * DEG
            )

    print()
    print(
        "--- ESTIMATED LOWEST CASTER WHEEL POINT ---"
    )

    print(
        f"Wheel center Z : "
        f"{center[2]:.6f} m"
    )

    print(
        f"Wheel radius   : "
        f"{CASTER_WHEEL_RADIUS:.6f} m"
    )

    print(
        f"Lowest point X : "
        f"{lowest_point[0]:.6f} m"
    )

    print(
        f"Lowest point Y : "
        f"{lowest_point[1]:.6f} m"
    )

    print(
        f"Lowest point Z : "
        f"{lowest_point[2]:.6f} m"
    )

    print(
        f"Radial angle   : "
        f"{lowest_angle:.3f} deg"
    )

    print(
        f"Ground gap     : "
        f"{lowest_z:.6f} m"
    )

    if abs(lowest_z) < 0.002:

        print(
            "Ground relationship: PASS"
        )

    elif lowest_z > 0.002:

        print(
            "Ground relationship: "
            "WHEEL ABOVE GROUND"
        )

    else:

        print(
            "Ground relationship: "
            "WHEEL PENETRATES GROUND"
        )

    return lowest_point


# ============================================================
# EXPECTED GEOMETRY
# ============================================================

def print_expected_geometry():

    print()
    print(
        "=" * 100
    )

    print(
        "EXPECTED CASTER GEOMETRY"
    )

    print(
        "=" * 100
    )

    print(
        f"Expected radius : "
        f"{EXPECTED_CASTER_RADIUS:.6f} m"
    )

    print(
        f"Expected width  : "
        f"{EXPECTED_CASTER_WIDTH:.6f} m"
    )

    print(
        f"Configured radius: "
        f"{CASTER_WHEEL_RADIUS:.6f} m"
    )

    print(
        f"Configured width : "
        f"{CASTER_WHEEL_WIDTH:.6f} m"
    )

    radius_error = abs(
        CASTER_WHEEL_RADIUS
        - EXPECTED_CASTER_RADIUS
    )

    width_error = abs(
        CASTER_WHEEL_WIDTH
        - EXPECTED_CASTER_WIDTH
    )

    if radius_error < 1e-9:

        print(
            "Radius: PASS"
        )

    else:

        print(
            "Radius: FAIL"
        )

    if width_error < 1e-9:

        print(
            "Width : PASS"
        )

    else:

        print(
            "Width : FAIL"
        )

    print()
    print(
        "Expected caster wheel local position:"
    )

    print(
        "  X = -0.05 m"
    )

    print(
        "  Y =  0.00 m"
    )

    print(
        "  Z = -0.04 m"
    )

    print()
    print(
        "Expected caster wheel local axis:"
    )

    print(
        "  Y axis"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "V3.22.5c CASTER COLLISION GEOMETRY INSPECTION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Inspect the actual PyBullet caster collision "
        "geometry without applying wheel motion."
    )

    print()

    print(
        "Reverse motion: DISABLED"
    )

    print(
        "PPO            : DISABLED"
    )

    print(
        "RL environment : DISABLED"
    )

    print(
        "Sensor noise   : DISABLED"
    )

    print(
        "Robot modified : NO"
    )

    print()

    connect_physics()

    try:

        # ----------------------------------------------------
        # WORLD
        # ----------------------------------------------------

        p.resetSimulation()

        p.setGravity(
            0.0,
            0.0,
            GRAVITY,
        )

        p.setTimeStep(
            1.0 / 240.0
        )

        plane_id = p.loadURDF(
            "plane.urdf"
        )

        robot_id = create_robot()

        p.resetBasePositionAndOrientation(
            robot_id,
            [
                0.0,
                0.0,
                RESET_BASE_Z,
            ],
            p.getQuaternionFromEuler(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )

        # ----------------------------------------------------
        # SETTLE
        # ----------------------------------------------------

        for _ in range(120):

            p.stepSimulation()

        # ----------------------------------------------------
        # EXPECTED VALUES
        # ----------------------------------------------------

        print_expected_geometry()

        # ----------------------------------------------------
        # COLLISION SHAPES
        # ----------------------------------------------------

        print_collision_shape_info(
            robot_id
        )

        # ----------------------------------------------------
        # LINK STATES
        # ----------------------------------------------------

        print_link_state(
            robot_id,
            CASTER_FORK_LINK,
            "CASTER FORK LINK",
        )

        print_link_state(
            robot_id,
            CASTER_WHEEL_LINK,
            "CASTER WHEEL LINK",
        )

        # ----------------------------------------------------
        # WHEEL AXIS
        # ----------------------------------------------------

        wheel_axis = inspect_wheel_axis(
            robot_id
        )

        # ----------------------------------------------------
        # LOWEST POINT
        # ----------------------------------------------------

        lowest_point = (
            estimate_lowest_wheel_point(
                robot_id
            )
        )

        # ----------------------------------------------------
        # CONTACT
        # ----------------------------------------------------

        contacts = p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=CASTER_WHEEL_LINK,
        )

        total_force = 0.0

        for contact in contacts:

            total_force += contact[9]

        print()
        print(
            "--- INITIAL CASTER CONTACT ---"
        )

        print(
            "Contact count :",
            len(contacts),
        )

        print(
            f"Normal force  : "
            f"{total_force:.6f} N"
        )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        print()
        print(
            "=" * 100
        )

        print(
            "V3.22.5c VALIDATION"
        )

        print(
            "=" * 100
        )

        print()

        shape_data = p.getCollisionShapeData(
            robot_id,
            CASTER_WHEEL_LINK,
        )

        shape_ok = False

        if shape_data:

            shape = shape_data[0]

            shape_type = shape[2]

            dimensions = shape[3]

            print(
                "Caster wheel shape type:",
                shape_type,
            )

            print(
                "Caster wheel dimensions:",
                dimensions,
            )

            # GEOM_CYLINDER = 4
            # For PyBullet cylinder:
            # dimensions[0] = radius
            # dimensions[1] = height

            if shape_type == p.GEOM_CYLINDER:

                radius = dimensions[0]
                height = dimensions[1]

                print(
                    f"Collision radius : "
                    f"{radius:.6f} m"
                )

                print(
                    f"Collision height : "
                    f"{height:.6f} m"
                )

                radius_pass = (
                    abs(
                        radius
                        - EXPECTED_CASTER_RADIUS
                    )
                    < 1e-6
                )

                width_pass = (
                    abs(
                        height
                        - EXPECTED_CASTER_WIDTH
                    )
                    < 1e-6
                )

                print(
                    "Radius check:",
                    "PASS"
                    if radius_pass
                    else "FAIL",
                )

                print(
                    "Width check:",
                    "PASS"
                    if width_pass
                    else "FAIL",
                )

                shape_ok = (
                    radius_pass
                    and width_pass
                )

            else:

                print(
                    "Caster wheel is NOT "
                    "a cylinder."
                )

        print()

        axis_horizontal = (
            abs(wheel_axis[2])
            < 0.01
        )

        print(
            "Wheel axis approximately horizontal:",
            "PASS"
            if axis_horizontal
            else "INVESTIGATE",
        )

        contact_ok = (
            len(contacts) > 0
            and total_force > 0.0
        )

        print(
            "Initial caster support:",
            "PASS"
            if contact_ok
            else "INVESTIGATE",
        )

        print()

        if (
            shape_ok
            and axis_horizontal
            and contact_ok
        ):

            print(
                "V3.22.5c GEOMETRY INSPECTION: PASS"
            )

            print()

            print(
                "Interpretation:"
            )

            print(
                "The caster collision geometry matches "
                "the configured dimensions and has the "
                "expected wheel-axis orientation."
            )

            print()

            print(
                "If V3.22.5 reverse motion still loses "
                "caster contact, the next investigation "
                "should focus on the caster mechanism/"
                "joint configuration rather than basic "
                "collision dimensions."
            )

        else:

            print(
                "V3.22.5c GEOMETRY INSPECTION: INVESTIGATE"
            )

            print()

            print(
                "At least one caster geometry/contact "
                "condition does not match the expected model."
            )

    finally:

        if p.isConnected():

            p.disconnect()

            print()
            print(
                "PyBullet disconnected."
            )

    print()
    print(
        "=" * 100
    )

    print(
        "V3.22.5c COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()