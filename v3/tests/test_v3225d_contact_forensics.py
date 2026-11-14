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
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAVITY = -9.81

TIME_STEP = 1.0 / 240.0

CHASSIS_HEIGHT = 0.10

RESET_BASE_Z = (
    WHEEL_RADIUS
    + CHASSIS_HEIGHT / 2.0
)

CASTER_FORK_LINK = 2
CASTER_WHEEL_LINK = 3

DEG = 180.0 / math.pi

SIMULATION_STEPS = 5


# ============================================================
# CONNECT
# ============================================================

def connect_physics():

    if p.isConnected():
        p.disconnect()

    client_id = p.connect(
        p.DIRECT
    )

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

    p.setTimeStep(
        TIME_STEP
    )

    print(
        "PyBullet connection: PASS"
    )

    print(
        "Physics client ID :",
        client_id,
    )


# ============================================================
# VECTOR HELPERS
# ============================================================

def vector_length(v):

    return math.sqrt(
        v[0] ** 2
        + v[1] ** 2
        + v[2] ** 2
    )


def normalize(v):

    length = vector_length(v)

    if length < 1e-12:

        return [
            0.0,
            0.0,
            0.0,
        ]

    return [
        v[0] / length,
        v[1] / length,
        v[2] / length,
    ]


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
        "position": state[4],
        "orientation": state[5],
    }


# ============================================================
# PRINT CONTACT
# ============================================================

def print_contact(
    contact_index,
    contact,
):

    # PyBullet contact tuple:
    #
    # [0]  contact flag
    # [1]  body unique ID A
    # [2]  body unique ID B
    # [3]  link index A
    # [4]  link index B
    # [5]  position on A
    # [6]  position on B
    # [7]  contact normal on B
    # [8]  contact distance
    # [9]  normal force
    #
    # Depending on PyBullet version additional fields may follow.

    print()
    print(
        f"CONTACT {contact_index}"
    )

    print(
        "  Body A              :",
        contact[1],
    )

    print(
        "  Body B              :",
        contact[2],
    )

    print(
        "  Link A              :",
        contact[3],
    )

    print(
        "  Link B              :",
        contact[4],
    )

    print(
        "  Position on A:"
    )

    print(
        f"    X = {contact[5][0]: .9f} m"
    )

    print(
        f"    Y = {contact[5][1]: .9f} m"
    )

    print(
        f"    Z = {contact[5][2]: .9f} m"
    )

    print(
        "  Position on B:"
    )

    print(
        f"    X = {contact[6][0]: .9f} m"
    )

    print(
        f"    Y = {contact[6][1]: .9f} m"
    )

    print(
        f"    Z = {contact[6][2]: .9f} m"
    )

    print(
        "  Contact normal on B:"
    )

    print(
        f"    X = {contact[7][0]: .9f}"
    )

    print(
        f"    Y = {contact[7][1]: .9f}"
    )

    print(
        f"    Z = {contact[7][2]: .9f}"
    )

    print(
        "  Contact distance    :",
        f"{contact[8]: .9f} m",
    )

    print(
        "  Normal force        :",
        f"{contact[9]: .9f} N",
    )


# ============================================================
# COLLISION SHAPES
# ============================================================

def inspect_collision_shapes(
    robot_id,
):

    print()
    print(
        "=" * 100
    )

    print(
        "CASTER COLLISION SHAPE"
    )

    print(
        "=" * 100
    )

    shapes = p.getCollisionShapeData(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    if not shapes:

        print(
            "NO COLLISION SHAPE FOUND"
        )

        return None

    for index, shape in enumerate(
        shapes
    ):

        print()
        print(
            f"Shape {index}"
        )

        print(
            "  Shape type:",
            shape[2],
        )

        print(
            "  Dimensions:",
            shape[3],
        )

        print(
            "  Local position:",
            shape[5],
        )

        print(
            "  Local orientation:",
            shape[6],
        )

    return shapes


# ============================================================
# LINK STATE INSPECTION
# ============================================================

def inspect_caster_state(
    robot_id,
):

    fork = get_link_state(
        robot_id,
        CASTER_FORK_LINK,
    )

    wheel = get_link_state(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    fork_roll, fork_pitch, fork_yaw = (
        p.getEulerFromQuaternion(
            fork["orientation"]
        )
    )

    wheel_roll, wheel_pitch, wheel_yaw = (
        p.getEulerFromQuaternion(
            wheel["orientation"]
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "CASTER LINK STATE"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "Caster fork:"
    )

    print(
        f"  Position = "
        f"({fork['position'][0]:.9f}, "
        f"{fork['position'][1]:.9f}, "
        f"{fork['position'][2]:.9f})"
    )

    print(
        f"  Euler    = "
        f"({fork_roll * DEG:.6f}, "
        f"{fork_pitch * DEG:.6f}, "
        f"{fork_yaw * DEG:.6f}) deg"
    )

    print()
    print(
        "Caster wheel:"
    )

    print(
        f"  Position = "
        f"({wheel['position'][0]:.9f}, "
        f"{wheel['position'][1]:.9f}, "
        f"{wheel['position'][2]:.9f})"
    )

    print(
        f"  Euler    = "
        f"({wheel_roll * DEG:.6f}, "
        f"{wheel_pitch * DEG:.6f}, "
        f"{wheel_yaw * DEG:.6f}) deg"
    )

    return wheel


# ============================================================
# CONTACT ANALYSIS
# ============================================================

def inspect_contacts(
    robot_id,
    plane_id,
):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "ALL ROBOT ↔ GROUND CONTACTS"
    )

    print(
        "=" * 100
    )

    print(
        "Total contacts:",
        len(contacts),
    )

    caster_contacts = []

    for index, contact in enumerate(
        contacts
    ):

        print_contact(
            index,
            contact,
        )

        if (
            contact[3]
            == CASTER_WHEEL_LINK
        ):

            caster_contacts.append(
                contact
            )

    print()
    print(
        "=" * 100
    )

    print(
        "CASTER WHEEL CONTACTS ONLY"
    )

    print(
        "=" * 100
    )

    print(
        "Caster contact count:",
        len(caster_contacts),
    )

    total_force = 0.0

    for index, contact in enumerate(
        caster_contacts
    ):

        total_force += contact[9]

        print_contact(
            index,
            contact,
        )

    print()
    print(
        "Total caster normal force:",
        f"{total_force:.9f} N",
    )

    return caster_contacts


# ============================================================
# GEOMETRIC SANITY CHECK
# ============================================================

def geometric_sanity_check(
    robot_id,
):

    wheel = get_link_state(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    center_z = wheel[
        "position"
    ][2]

    shapes = p.getCollisionShapeData(
        robot_id,
        CASTER_WHEEL_LINK,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "GEOMETRIC SANITY CHECK"
    )

    print(
        "=" * 100
    )

    if not shapes:

        print(
            "Cannot inspect geometry."
        )

        return

    shape = shapes[0]

    shape_type = shape[2]

    dimensions = shape[3]

    print(
        "Shape type:",
        shape_type,
    )

    print(
        "Dimensions:",
        dimensions,
    )

    print(
        f"Wheel link center Z: "
        f"{center_z:.9f} m"
    )

    if shape_type == p.GEOM_CYLINDER:

        radius = dimensions[0]

        print(
            f"PyBullet cylinder radius: "
            f"{radius:.9f} m"
        )

        expected_bottom = (
            center_z
            - radius
        )

        print(
            f"Estimated cylinder bottom Z: "
            f"{expected_bottom:.9f} m"
        )

        print(
            f"Configured intended radius: "
            f"{CASTER_WHEEL_RADIUS:.9f} m"
        )

        print(
            f"Configured intended width: "
            f"{CASTER_WHEEL_WIDTH:.9f} m"
        )

        print()
        print(
            "NOTE:"
        )

        print(
            "This simple bottom estimate assumes "
            "the cylinder axis is horizontal and "
            "the cylinder orientation is not tilted."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "V3.22.5d CONTACT POINT FORENSICS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Determine exactly why PyBullet reports "
        "caster-ground contact."
    )

    print()

    print(
        "Reverse motion : DISABLED"
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
            TIME_STEP
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
        # SHAPES
        # ----------------------------------------------------

        inspect_collision_shapes(
            robot_id
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        inspect_caster_state(
            robot_id
        )

        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        geometric_sanity_check(
            robot_id
        )

        # ----------------------------------------------------
        # CONTACTS BEFORE EXTRA STEP
        # ----------------------------------------------------

        print()
        print(
            "=" * 100
        )

        print(
            "CONTACTS AFTER SETTLE"
        )

        print(
            "=" * 100
        )

        inspect_contacts(
            robot_id,
            plane_id,
        )

        # ----------------------------------------------------
        # CONTACT DATA AFTER EACH STEP
        # ----------------------------------------------------

        print()
        print(
            "=" * 100
        )

        print(
            "CONTACT EVOLUTION"
        )

        print(
            "=" * 100
        )

        for step in range(
            1,
            SIMULATION_STEPS + 1,
        ):

            p.stepSimulation()

            print()
            print(
                f"--- PHYSICS STEP {step} ---"
            )

            inspect_caster_state(
                robot_id
            )

            contacts = (
                inspect_contacts(
                    robot_id,
                    plane_id,
                )
            )

            caster_count = len(
                contacts
            )

            caster_force = sum(
                contact[9]
                for contact in contacts
            )

            print()
            print(
                f"SUMMARY STEP {step}: "
                f"caster_contacts={caster_count}, "
                f"caster_force={caster_force:.9f} N"
            )

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        print()
        print(
            "=" * 100
        )

        print(
            "V3.22.5d CONCLUSION"
        )

        print(
            "=" * 100
        )

        final_contacts = (
            p.getContactPoints(
                bodyA=robot_id,
                bodyB=plane_id,
                linkIndexA=CASTER_WHEEL_LINK,
            )
        )

        if final_contacts:

            print(
                "Caster-ground contact EXISTS."
            )

            print(
                "Use the reported contact position, "
                "distance, normal, and force to determine "
                "the actual contact geometry."
            )

        else:

            print(
                "Caster-ground contact does NOT exist "
                "after the final step."
            )

        print()

        print(
            "V3.22.5d CONTACT FORENSICS: COMPLETE"
        )

    finally:

        if p.isConnected():

            p.disconnect()

            print()
            print(
                "PyBullet disconnected."
            )


if __name__ == "__main__":
    main()