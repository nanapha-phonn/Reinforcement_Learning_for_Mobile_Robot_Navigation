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
    set_wheel_velocity,
    stop_robot,
    WHEEL_RADIUS,
    TIME_STEP,
)


# ============================================================
# CONFIGURATION
# ============================================================

GRAVITY = -9.81

SIMULATION_STEPS = 150

WARMUP_STEPS = 120

REVERSE_LEFT = -4.0
REVERSE_RIGHT = -4.0

CHASSIS_HEIGHT = 0.10

RESET_BASE_Z = (
    WHEEL_RADIUS
    + CHASSIS_HEIGHT / 2.0
)

CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

CASTER_WHEEL_LINK = 3

DEG = 180.0 / math.pi


# ============================================================
# PYBULLET
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

    p.setTimeStep(TIME_STEP)

    print(
        "PyBullet connection: PASS"
    )

    print(
        "Physics client ID :",
        client_id,
    )


# ============================================================
# WORLD
# ============================================================

def create_world():

    p.resetSimulation()

    p.setGravity(
        0.0,
        0.0,
        GRAVITY,
    )

    p.setTimeStep(TIME_STEP)

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

    stop_robot(robot_id)

    for _ in range(WARMUP_STEPS):
        p.stepSimulation()

    return plane_id, robot_id


# ============================================================
# JOINT STATE
# ============================================================

def get_joint_state(
    robot_id,
    joint_index,
):

    state = p.getJointState(
        robot_id,
        joint_index,
    )

    return {
        "position": state[0],
        "velocity": state[1],
        "reaction_forces": state[2],
        "motor_torque": state[3],
    }


# ============================================================
# BASE STATE
# ============================================================

def get_base_state(robot_id):

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return {
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


# ============================================================
# CASTER LINK STATE
# ============================================================

def get_caster_link_state(robot_id):

    """
    Read the world position and orientation
    of the caster wheel link.
    """

    state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_LINK,
        computeLinkVelocity=1,
        computeForwardKinematics=1,
    )

    # World center-of-mass position/orientation
    world_com_position = state[0]
    world_com_orientation = state[1]

    # World link-frame position/orientation
    world_link_position = state[4]
    world_link_orientation = state[5]

    linear_velocity = state[6]
    angular_velocity = state[7]

    return {
        "com_x": world_com_position[0],
        "com_y": world_com_position[1],
        "com_z": world_com_position[2],

        "link_x": world_link_position[0],
        "link_y": world_link_position[1],
        "link_z": world_link_position[2],

        "com_orientation": (
            world_com_orientation
        ),

        "link_orientation": (
            world_link_orientation
        ),

        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
    }


# ============================================================
# CASTER ORIENTATION
# ============================================================

def quaternion_to_euler(
    quaternion
):

    return p.getEulerFromQuaternion(
        quaternion
    )


# ============================================================
# CONTACT
# ============================================================

def get_caster_contact_data(
    robot_id,
    plane_id,
):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=CASTER_WHEEL_LINK,
    )

    total_normal_force = 0.0

    min_contact_distance = None

    for contact in contacts:

        # contact[8] = contact distance
        # contact[9] = normal force

        distance = contact[8]
        force = contact[9]

        total_normal_force += force

        if (
            min_contact_distance is None
            or distance < min_contact_distance
        ):
            min_contact_distance = distance

    return {
        "contact_count": len(contacts),
        "normal_force": total_normal_force,
        "contact_distance": min_contact_distance,
    }


# ============================================================
# PRINT SAMPLE
# ============================================================

def print_sample(
    step,
    robot_id,
    plane_id,
):

    base = get_base_state(
        robot_id
    )

    caster = get_caster_link_state(
        robot_id
    )

    swivel = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT,
    )

    wheel = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    contact = get_caster_contact_data(
        robot_id,
        plane_id,
    )

    caster_roll, caster_pitch, caster_yaw = (
        quaternion_to_euler(
            caster["link_orientation"]
        )
    )

    print(
        f"Step {step:3d} | "
        f"BaseZ={base['z']:8.5f} | "
        f"Pitch={base['pitch'] * DEG:8.3f}° | "
        f"CasterZ={caster['link_z']:8.5f} | "
        f"Swivel={swivel['position'] * DEG:8.3f}° | "
        f"CasterPitch={caster_pitch * DEG:8.3f}° | "
        f"Contact={contact['contact_count']} | "
        f"Force={contact['normal_force']:8.4f} N"
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 110)
    print(
        "V3.22.5b CASTER GEOMETRY / REVERSE LIFT DIAGNOSTIC"
    )
    print("=" * 110)

    print()
    print(
        "Purpose:"
    )

    print(
        "Measure the caster wheel world pose during "
        "the onset of reverse motion."
    )

    print()
    print(
        "PPO                 : DISABLED"
    )

    print(
        "Reward              : NOT USED"
    )

    print(
        "RL environment      : NOT USED"
    )

    print(
        "Sensor noise        : NOT USED"
    )

    print(
        "Physics modification: NONE"
    )

    print()
    print(
        f"Wheel radius        : "
        f"{WHEEL_RADIUS:.3f} m"
    )

    print(
        f"Reset base Z        : "
        f"{RESET_BASE_Z:.3f} m"
    )

    print(
        f"Reverse command     : "
        f"{REVERSE_LEFT:.3f}, "
        f"{REVERSE_RIGHT:.3f} rad/s"
    )

    connect_physics()

    try:

        plane_id, robot_id = (
            create_world()
        )

        print()
        print(
            "Initial caster state:"
        )

        print_sample(
            0,
            robot_id,
            plane_id,
        )

        # ----------------------------------------------------
        # Apply reverse motion
        # ----------------------------------------------------

        set_wheel_velocity(
            robot_id,
            REVERSE_LEFT,
            REVERSE_RIGHT,
        )

        print()
        print(
            "--- REVERSE MOTION ---"
        )

        # ----------------------------------------------------
        # Selected samples
        # ----------------------------------------------------

        sample_steps = {
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            75,
            100,
            125,
            150,
        }

        caster_lift_start = None
        first_contact_loss = None
        maximum_caster_z = -float("inf")
        minimum_caster_z = float("inf")

        maximum_pitch = 0.0

        for step in range(
            1,
            SIMULATION_STEPS + 1,
        ):

            p.stepSimulation()

            base = get_base_state(
                robot_id
            )

            caster = get_caster_link_state(
                robot_id
            )

            contact = (
                get_caster_contact_data(
                    robot_id,
                    plane_id,
                )
            )

            caster_z = caster[
                "link_z"
            ]

            pitch_abs = abs(
                base["pitch"]
            )

            maximum_caster_z = max(
                maximum_caster_z,
                caster_z,
            )

            minimum_caster_z = min(
                minimum_caster_z,
                caster_z,
            )

            maximum_pitch = max(
                maximum_pitch,
                pitch_abs,
            )

            # First contact loss
            if (
                contact["contact_count"] == 0
                and first_contact_loss is None
            ):
                first_contact_loss = step

            # Detect caster movement upward
            if (
                caster_z > 0.04
                and caster_lift_start is None
            ):
                caster_lift_start = step

            if step in sample_steps:

                print_sample(
                    step,
                    robot_id,
                    plane_id,
                )

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        stop_robot(
            robot_id
        )

        # ----------------------------------------------------
        # Final measurements
        # ----------------------------------------------------

        final_base = get_base_state(
            robot_id
        )

        final_caster = (
            get_caster_link_state(
                robot_id
            )
        )

        final_contact = (
            get_caster_contact_data(
                robot_id,
                plane_id,
            )
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 110)
        print(
            "V3.22.5b RESULTS"
        )
        print("=" * 110)

        print()

        print(
            "--- CASTER POSITION ---"
        )

        print(
            f"Initial caster Z : "
            f"{get_caster_link_state(robot_id)['link_z']:.6f} m"
        )

        print(
            f"Final caster Z   : "
            f"{final_caster['link_z']:.6f} m"
        )

        print(
            f"Minimum caster Z : "
            f"{minimum_caster_z:.6f} m"
        )

        print(
            f"Maximum caster Z : "
            f"{maximum_caster_z:.6f} m"
        )

        print(
            f"Caster Z range   : "
            f"{maximum_caster_z - minimum_caster_z:.6f} m"
        )

        print()
        print(
            "--- CONTACT ---"
        )

        if first_contact_loss is None:

            print(
                "First contact loss : NONE"
            )

        else:

            print(
                f"First contact loss : "
                f"step {first_contact_loss}"
            )

        print(
            f"Final contact     : "
            f"{final_contact['contact_count']}"
        )

        print(
            f"Final normal force: "
            f"{final_contact['normal_force']:.6f} N"
        )

        print()
        print(
            "--- BASE ATTITUDE ---"
        )

        print(
            f"Final base Z      : "
            f"{final_base['z']:.6f} m"
        )

        print(
            f"Final pitch       : "
            f"{final_base['pitch'] * DEG:.6f} deg"
        )

        print(
            f"Maximum |pitch|   : "
            f"{maximum_pitch * DEG:.6f} deg"
        )

        print()
        print(
            "--- DIAGNOSTIC CONCLUSION ---"
        )

        if (
            first_contact_loss is not None
            and maximum_pitch > math.radians(10.0)
        ):

            print(
                "CONFIRMED:"
            )

            print(
                "Caster loses ground contact before/during "
                "the large reverse pitch."
            )

            print(
                "The next task is to determine whether "
                "the caster geometry or joint configuration "
                "causes this lift."
            )

        elif first_contact_loss is None:

            print(
                "Caster did not lose contact."
            )

            print(
                "Investigate wheel/chassis contact mechanics."
            )

        else:

            print(
                "Caster contact loss occurred, but the "
                "large pitch was not reproduced."
            )

    finally:

        if p.isConnected():

            p.disconnect()

            print()
            print(
                "PyBullet disconnected."
            )

    print()
    print("=" * 110)
    print(
        "V3.22.5b COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()