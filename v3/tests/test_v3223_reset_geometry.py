import os
import sys
import math

import numpy as np
import pybullet as p


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from environment import DifferentialDriveEnv


# ============================================================
# V3.22.3 RESET / GEOMETRY DIAGNOSTIC
# ============================================================

def quaternion_to_euler_degrees(quaternion):
    """Convert quaternion to roll, pitch, yaw in degrees."""

    roll, pitch, yaw = p.getEulerFromQuaternion(
        quaternion
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def print_pose(label, position, quaternion):
    """Print position, quaternion, and Euler angles."""

    roll, pitch, yaw = quaternion_to_euler_degrees(
        quaternion
    )

    print()
    print(f"--- {label} ---")

    print(
        f"Position   : "
        f"X={position[0]:.6f} m | "
        f"Y={position[1]:.6f} m | "
        f"Z={position[2]:.6f} m"
    )

    print(
        f"Quaternion : "
        f"X={quaternion[0]:.8f} | "
        f"Y={quaternion[1]:.8f} | "
        f"Z={quaternion[2]:.8f} | "
        f"W={quaternion[3]:.8f}"
    )

    print(
        f"Euler      : "
        f"Roll={roll:.6f}° | "
        f"Pitch={pitch:.6f}° | "
        f"Yaw={yaw:.6f}°"
    )

    return roll, pitch, yaw


def print_body_info(robot_id):
    """Print basic multibody information."""

    print()
    print("--- MULTIBODY INFORMATION ---")

    number_of_joints = p.getNumJoints(
        robot_id
    )

    print(
        f"Number of joints : "
        f"{number_of_joints}"
    )

    for joint_index in range(number_of_joints):

        info = p.getJointInfo(
            robot_id,
            joint_index,
        )

        joint_name = info[1].decode(
            "utf-8",
            errors="replace",
        )

        joint_type = info[2]

        parent_index = info[16]

        joint_axis = info[13]

        print(
            f"Joint {joint_index}: "
            f"name={joint_name} | "
            f"type={joint_type} | "
            f"parent={parent_index} | "
            f"axis={joint_axis}"
        )


def print_contact_points(env):
    """Print all contacts involving the robot."""

    contacts = p.getContactPoints(
        bodyA=env.robot_id
    )

    print()
    print("--- CONTACT INFORMATION ---")

    if len(contacts) == 0:

        print(
            "No contact points detected."
        )

        return contacts

    print(
        f"Total contact points : "
        f"{len(contacts)}"
    )

    for index, contact in enumerate(
        contacts
    ):

        body_a = contact[1]
        body_b = contact[2]

        link_a = contact[3]
        link_b = contact[4]

        distance = contact[8]
        normal_force = contact[9]

        print(
            f"Contact {index}: "
            f"bodyA={body_a} | "
            f"bodyB={body_b} | "
            f"linkA={link_a} | "
            f"linkB={link_b} | "
            f"distance={distance:.6f} m | "
            f"normal_force={normal_force:.6f} N"
        )

    return contacts


def print_link_states(env):
    """Print world position/orientation of every link."""

    print()
    print("--- LINK STATES ---")

    number_of_joints = p.getNumJoints(
        env.robot_id
    )

    for joint_index in range(
        number_of_joints
    ):

        state = p.getLinkState(
            env.robot_id,
            joint_index,
            computeForwardKinematics=True,
        )

        world_position = state[4]
        world_orientation = state[5]

        roll, pitch, yaw = (
            quaternion_to_euler_degrees(
                world_orientation
            )
        )

        print(
            f"Link {joint_index}: "
            f"X={world_position[0]:.6f} | "
            f"Y={world_position[1]:.6f} | "
            f"Z={world_position[2]:.6f} | "
            f"Roll={roll:.3f}° | "
            f"Pitch={pitch:.3f}° | "
            f"Yaw={yaw:.3f}°"
        )


def main():

    print("=" * 90)
    print(
        "V3.22.3 RESET / QUATERNION / "
        "COLLISION GEOMETRY DIAGNOSTIC"
    )
    print("=" * 90)

    print()
    print("Purpose:")
    print(
        "Determine whether the initial pitch is "
        "already encoded in the reset pose and "
        "identify the bodies/links contacting the ground."
    )

    print()
    print("PPO: DISABLED")
    print("Wheel commands: NONE")
    print("Reward: NOT EVALUATED")
    print("Physics parameters: UNCHANGED")
    print("Environment source: UNCHANGED")

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        # ====================================================
        # RESET
        # ====================================================

        env.reset(seed=1)

        # ----------------------------------------------------
        # Read pose immediately after reset
        # ----------------------------------------------------

        position, quaternion = (
            p.getBasePositionAndOrientation(
                env.robot_id
            )
        )

        reset_roll, reset_pitch, reset_yaw = (
            print_pose(
                "IMMEDIATELY AFTER RESET",
                position,
                quaternion,
            )
        )

        # ====================================================
        # MULTIBODY INFORMATION
        # ====================================================

        print_body_info(
            env.robot_id
        )

        # ====================================================
        # LINK INFORMATION
        # ====================================================

        print_link_states(
            env
        )

        # ====================================================
        # CONTACT INFORMATION
        # ====================================================

        contacts_before = print_contact_points(
            env
        )

        # ====================================================
        # ONE PHYSICS STEP
        # ====================================================

        p.stepSimulation()

        position, quaternion = (
            p.getBasePositionAndOrientation(
                env.robot_id
            )
        )

        step_roll, step_pitch, step_yaw = (
            print_pose(
                "AFTER ONE PHYSICS STEP",
                position,
                quaternion,
            )
        )

        contacts_after = print_contact_points(
            env
        )

        # ====================================================
        # DIFFERENCES
        # ====================================================

        print()
        print("=" * 90)
        print("V3.22.3 COMPARISON")
        print("=" * 90)

        print()

        print(
            f"Reset pitch           : "
            f"{reset_pitch:.6f}°"
        )

        print(
            f"After 1 step pitch    : "
            f"{step_pitch:.6f}°"
        )

        print(
            f"Pitch change           : "
            f"{step_pitch - reset_pitch:+.6f}°"
        )

        print()

        print(
            f"Reset Z                : "
            f"{position[2]:.6f} m"
        )

        print(
            f"Contacts before step  : "
            f"{len(contacts_before)}"
        )

        print(
            f"Contacts after step   : "
            f"{len(contacts_after)}"
        )

        # ====================================================
        # DIAGNOSTIC CONCLUSION
        # ====================================================

        print()
        print("=" * 90)
        print("V3.22.3 DIAGNOSTIC CONCLUSION")
        print("=" * 90)

        if abs(reset_pitch) > 1.0:

            print()
            print(
                "INITIAL POSE OFFSET DETECTED"
            )

            print(
                f"The robot already has "
                f"{reset_pitch:.3f}° pitch immediately "
                f"after env.reset()."
            )

            print(
                "Therefore the initial orientation "
                "is not level."
            )

        else:

            print()
            print(
                "INITIAL POSE IS APPROXIMATELY LEVEL"
            )

            print(
                "The pitch develops after physics/contact "
                "simulation."
            )

        if (
            len(contacts_before) == 0
            and len(contacts_after) > 0
        ):

            print()
            print(
                "CONTACT-INDUCED CHANGE DETECTED"
            )

            print(
                "Contact appears only after the first "
                "physics step."
            )

        elif len(contacts_before) > 0:

            print()
            print(
                "CONTACT EXISTS IMMEDIATELY AFTER RESET"
            )

            print(
                "The robot is already touching another "
                "collision body before stepping physics."
            )

        else:

            print()
            print(
                "NO CONTACT DETECTED"
            )

            print(
                "No robot contact points were reported "
                "at the tested checkpoints."
            )

        print()
        print(
            "V3.22.3 RESET / GEOMETRY "
            "DIAGNOSTIC COMPLETE"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()