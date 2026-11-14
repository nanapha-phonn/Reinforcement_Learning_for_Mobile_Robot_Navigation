"""
V3.22.5l — Production Reverse-Motion Regression

Purpose:
    Verify the actual production robot's reverse stability now that
    the PyBullet cylinder-dimension reporting issue has been resolved.

Tests:
    1. Static
    2. Forward +4/+4
    3. Reverse -4/-4
    4. Rotation -4/+4
    5. Curve +4/+2

No PPO.
No RL environment.
No sensor noise.
No robot.py modification.
"""

import math
import time

import pybullet as p

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

import robot


TIME_STEP = 1.0 / 240.0
STEPS = 1000

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1

GROUND_ID = None


def get_euler(robot_id):

    position, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def get_caster_contacts(robot_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=GROUND_ID
    )

    caster_contacts = [
        c
        for c in contacts
        if c[3] == 3
    ]

    return caster_contacts


def get_caster_force(robot_id):

    contacts = get_caster_contacts(
        robot_id
    )

    return sum(
        c[9]
        for c in contacts
    )


def run_case(
    name,
    left_velocity,
    right_velocity,
):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    robot_id = robot.create_robot()

    p.resetBasePositionAndOrientation(
        robot_id,
        [
            0.0,
            0.0,
            robot.WHEEL_RADIUS
            + robot.CHASSIS_HEIGHT / 2.0
        ],
        p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        )
    )

    robot.set_wheel_velocity(
        robot_id,
        left_velocity,
        right_velocity
    )

    pitch_values = []
    roll_values = []
    yaw_values = []
    caster_forces = []
    caster_contacts = []

    for step in range(STEPS):

        p.stepSimulation()

        roll, pitch, yaw = get_euler(
            robot_id
        )

        pitch_values.append(
            abs(pitch)
        )

        roll_values.append(
            abs(roll)
        )

        yaw_values.append(
            yaw
        )

        contacts = get_caster_contacts(
            robot_id
        )

        caster_contacts.append(
            len(contacts)
        )

        caster_forces.append(
            get_caster_force(
                robot_id
            )
        )

        if step in (
            0,
            1,
            10,
            50,
            100,
            500,
            999
        ):

            position, _ = (
                p.getBasePositionAndOrientation(
                    robot_id
                )
            )

            print(
                f"Step {step + 1:4d} | "
                f"X={position[0]: .5f} | "
                f"Z={position[2]: .5f} | "
                f"Roll={roll: .4f}° | "
                f"Pitch={pitch: .4f}° | "
                f"Yaw={yaw: .4f}° | "
                f"Caster={len(contacts)} | "
                f"Force={caster_forces[-1]: .4f} N"
            )

    robot.stop_robot(
        robot_id
    )

    final_roll, final_pitch, final_yaw = get_euler(
        robot_id
    )

    position, _ = p.getBasePositionAndOrientation(
        robot_id
    )

    avg_pitch = (
        sum(pitch_values)
        / len(pitch_values)
    )

    avg_roll = (
        sum(roll_values)
        / len(roll_values)
    )

    avg_caster_force = (
        sum(caster_forces)
        / len(caster_forces)
    )

    contact_fraction = (
        sum(
            c > 0
            for c in caster_contacts
        )
        / len(caster_contacts)
    )

    full_contact_fraction = (
        sum(
            c >= 3
            for c in caster_contacts
        )
        / len(caster_contacts)
    )

    print()
    print("-" * 80)
    print("RESULT")
    print("-" * 80)

    print(
        f"Final X             : {position[0]:.6f} m"
    )

    print(
        f"Final Z             : {position[2]:.6f} m"
    )

    print(
        f"Final roll          : {final_roll:.6f}°"
    )

    print(
        f"Final pitch         : {final_pitch:.6f}°"
    )

    print(
        f"Final yaw           : {final_yaw:.6f}°"
    )

    print(
        f"Max |pitch|         : "
        f"{max(pitch_values):.6f}°"
    )

    print(
        f"Average |pitch|     : "
        f"{avg_pitch:.6f}°"
    )

    print(
        f"Max |roll|          : "
        f"{max(roll_values):.6f}°"
    )

    print(
        f"Average |roll|     : "
        f"{avg_roll:.6f}°"
    )

    print(
        f"Caster contact      : "
        f"{contact_fraction * 100:.2f}%"
    )

    print(
        f"Full caster contact : "
        f"{full_contact_fraction * 100:.2f}%"
    )

    print(
        f"Average caster force: "
        f"{avg_caster_force:.6f} N"
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    attitude_pass = (
        max(pitch_values) < 2.0
        and
        max(roll_values) < 2.0
    )

    caster_pass = (
        contact_fraction >= 0.95
    )

    if attitude_pass:
        print(
            "[PASS] Attitude stability."
        )
    else:
        print(
            "[FAIL] Attitude stability."
        )

    if caster_pass:
        print(
            "[PASS] Caster maintains contact."
        )
    else:
        print(
            "[FAIL] Caster contact stability."
        )

    case_pass = (
        attitude_pass
        and
        caster_pass
    )

    print()

    if case_pass:
        print(
            "[PASS] CASE"
        )
    else:
        print(
            "[FAIL] CASE"
        )

    p.removeBody(
        robot_id
    )

    return case_pass


def main():

    print("=" * 80)
    print("V3.22.5l — PRODUCTION PHYSICS REGRESSION")
    print("=" * 80)

    client = p.connect(
        p.DIRECT
    )

    if client < 0:

        print(
            "[FAIL] PyBullet connection."
        )

        return

    try:

        p.setGravity(
            0,
            0,
            -9.81
        )

        p.setTimeStep(
            TIME_STEP
        )

        # ----------------------------------------------------------
        # Ground
        # ----------------------------------------------------------

        ground_shape = p.createCollisionShape(
            p.GEOM_PLANE
        )

        global GROUND_ID

        GROUND_ID = p.createMultiBody(
            0,
            ground_shape
        )

        # ----------------------------------------------------------
        # Cases
        # ----------------------------------------------------------

        results = {}

        results["STATIC"] = run_case(
            "STATIC",
            0.0,
            0.0
        )

        results["FORWARD"] = run_case(
            "FORWARD +4/+4",
            4.0,
            4.0
        )

        results["REVERSE"] = run_case(
            "REVERSE -4/-4",
            -4.0,
            -4.0
        )

        results["ROTATION"] = run_case(
            "ROTATION -4/+4",
            -4.0,
            4.0
        )

        results["CURVE"] = run_case(
            "CURVE +4/+2",
            4.0,
            2.0
        )

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------

        print()
        print("=" * 80)
        print("V3.22.5l SUMMARY")
        print("=" * 80)

        for name, result in results.items():

            print(
                f"{name:12s} : "
                f"{'PASS' if result else 'FAIL'}"
            )

        all_pass = all(
            results.values()
        )

        print()

        if all_pass:

            print(
                "[PASS] V3.22.5l — production physics regression."
            )

            print()
            print(
                "The robot is ready to proceed to "
                "V3.22.6 generalization."
            )

        else:

            print(
                "[FAIL] V3.22.5l — production physics "
                "still requires investigation."
            )

            print()
            print(
                "Do NOT proceed to PPO retraining."
            )

        print()
        print(
            "robot.py was NOT modified."
        )

        print(
            "PPO was NOT trained."
        )

        print("=" * 80)

    finally:

        p.disconnect()


if __name__ == "__main__":
    main()