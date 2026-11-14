"""
V3.22.5q — DRIVE-WHEEL MOTOR-FORCE ISOLATION TEST

Purpose
-------
Determine whether the reverse instability is caused by
active drive-wheel motor torque / traction.

Test cases:

A: target -4/-4 rad/s, force = 0 N
B: target -4/-4 rad/s, force = 1 N
C: target -4/-4 rad/s, force = 5 N
D: target -4/-4 rad/s, force = 20 N

Production robot.py is NOT modified.

Run:
    cd C:\\python_projects\\rl_robot\\v3
    python tests\\test_v3225q_drive_motor_force_isolation.py
"""

import sys
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


# ============================================================
# PROJECT IMPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot import create_robot, stop_robot


# ============================================================
# CONFIGURATION
# ============================================================

TIME_STEP = 1.0 / 240.0
GRAVITY = -9.81

SETTLE_STEPS = 240
MOTION_STEPS = 1000

TARGET_LEFT_VELOCITY = -4.0
TARGET_RIGHT_VELOCITY = -4.0

MOTOR_FORCES = [
    0.0,
    1.0,
    5.0,
    20.0,
]

EXPECTED_BASE_Z = 0.13
BASE_Z_TOLERANCE = 0.02

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1

CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3


# ============================================================
# GROUND
# ============================================================

def create_ground():
    plane_collision = p.createCollisionShape(
        p.GEOM_PLANE
    )

    plane_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=plane_collision
    )

    return plane_id


# ============================================================
# STATE HELPERS
# ============================================================

def get_base_state(robot_id):

    position, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    linear_velocity, angular_velocity = p.getBaseVelocity(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return {
        "x": position[0],
        "y": position[1],
        "z": position[2],

        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),

        "vx": linear_velocity[0],
        "vy": linear_velocity[1],
        "vz": linear_velocity[2],

        "wx": angular_velocity[0],
        "wy": angular_velocity[1],
        "wz": angular_velocity[2],
    }


def get_joint_state(robot_id, joint_index):

    state = p.getJointState(
        robot_id,
        joint_index
    )

    return {
        "position": state[0],
        "velocity": state[1],
        "reaction_forces": state[2],
        "motor_torque": state[3],
    }


def get_caster_contacts(robot_id, plane_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=3
    )

    total_normal_force = 0.0

    for contact in contacts:
        total_normal_force += contact[9]

    return len(contacts), total_normal_force


def get_caster_z(robot_id):

    link_state = p.getLinkState(
        robot_id,
        3,
        computeForwardKinematics=True
    )

    world_position = link_state[4]

    return world_position[2]


# ============================================================
# MOTOR CONTROL
# ============================================================

def command_drive_wheels(robot_id, motor_force):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=LEFT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=TARGET_LEFT_VELOCITY,
        force=motor_force
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=RIGHT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=TARGET_RIGHT_VELOCITY,
        force=motor_force
    )


def disable_drive_wheels(robot_id):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=LEFT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=RIGHT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0
    )


def disable_caster_motors(robot_id):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=CASTER_SWIVEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=CASTER_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0
    )


def disable_all_motors(robot_id):

    disable_drive_wheels(robot_id)
    disable_caster_motors(robot_id)


# ============================================================
# SINGLE TEST
# ============================================================

def run_test(robot_id, plane_id, motor_force):

    print()
    print("=" * 100)
    print(
        f"CASE: MOTOR FORCE = {motor_force:.1f} N"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Reset base
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        robot_id,
        [0.0, 0.0, EXPECTED_BASE_Z],
        p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        )
    )

    p.resetBaseVelocity(
        robot_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0]
    )

    # --------------------------------------------------------
    # Reset joints
    # --------------------------------------------------------

    for joint_index in range(
        p.getNumJoints(robot_id)
    ):

        p.resetJointState(
            robot_id,
            joint_index,
            targetValue=0.0,
            targetVelocity=0.0
        )

    disable_all_motors(robot_id)

    # --------------------------------------------------------
    # Settle
    # --------------------------------------------------------

    for _ in range(SETTLE_STEPS):

        disable_all_motors(robot_id)

        p.stepSimulation()

    initial = get_base_state(robot_id)

    initial_caster_contacts, initial_caster_force = (
        get_caster_contacts(
            robot_id,
            plane_id
        )
    )

    print()
    print("INITIAL STATE")
    print("-" * 100)

    print(
        f"X={initial['x']: .5f} | "
        f"Z={initial['z']: .5f} | "
        f"Pitch={initial['pitch']: .4f}° | "
        f"Roll={initial['roll']: .4f}°"
    )

    print(
        f"Caster contacts={initial_caster_contacts} | "
        f"Force={initial_caster_force:.4f} N"
    )

    if abs(initial["z"] - EXPECTED_BASE_Z) > BASE_Z_TOLERANCE:

        print()
        print("[FAIL] Invalid initial base height.")

        return None

    print("[PASS] Initial base height is valid.")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    max_pitch = 0.0
    max_roll = 0.0

    min_base_z = float("inf")
    max_base_z = float("-inf")

    min_caster_z = float("inf")
    max_caster_z = float("-inf")

    contact_steps = 0

    total_caster_force = 0.0
    max_caster_force = 0.0

    max_left_velocity_error = 0.0
    max_right_velocity_error = 0.0

    max_left_motor_torque = 0.0
    max_right_motor_torque = 0.0

    max_left_reaction = 0.0
    max_right_reaction = 0.0

    max_swivel = 0.0
    max_swivel_velocity = 0.0

    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    print()
    print("REVERSE DRIVE")
    print("-" * 100)

    for step in range(
        1,
        MOTION_STEPS + 1
    ):

        # Drive wheels ONLY.
        command_drive_wheels(
            robot_id,
            motor_force
        )

        # Caster remains passive.
        disable_caster_motors(
            robot_id
        )

        p.stepSimulation()

        # ----------------------------------------------------
        # Base
        # ----------------------------------------------------

        base = get_base_state(
            robot_id
        )

        max_pitch = max(
            max_pitch,
            abs(base["pitch"])
        )

        max_roll = max(
            max_roll,
            abs(base["roll"])
        )

        min_base_z = min(
            min_base_z,
            base["z"]
        )

        max_base_z = max(
            max_base_z,
            base["z"]
        )

        # ----------------------------------------------------
        # Drive wheels
        # ----------------------------------------------------

        left = get_joint_state(
            robot_id,
            LEFT_WHEEL_JOINT
        )

        right = get_joint_state(
            robot_id,
            RIGHT_WHEEL_JOINT
        )

        left_velocity_error = abs(
            left["velocity"] - TARGET_LEFT_VELOCITY
        )

        right_velocity_error = abs(
            right["velocity"] - TARGET_RIGHT_VELOCITY
        )

        max_left_velocity_error = max(
            max_left_velocity_error,
            left_velocity_error
        )

        max_right_velocity_error = max(
            max_right_velocity_error,
            right_velocity_error
        )

        max_left_motor_torque = max(
            max_left_motor_torque,
            abs(left["motor_torque"])
        )

        max_right_motor_torque = max(
            max_right_motor_torque,
            abs(right["motor_torque"])
        )

        # Reaction force vector.
        left_reaction = math.sqrt(
            sum(
                value * value
                for value in left["reaction_forces"]
            )
        )

        right_reaction = math.sqrt(
            sum(
                value * value
                for value in right["reaction_forces"]
            )
        )

        max_left_reaction = max(
            max_left_reaction,
            left_reaction
        )

        max_right_reaction = max(
            max_right_reaction,
            right_reaction
        )

        # ----------------------------------------------------
        # Caster
        # ----------------------------------------------------

        swivel = get_joint_state(
            robot_id,
            CASTER_SWIVEL_JOINT
        )

        max_swivel = max(
            max_swivel,
            abs(
                math.degrees(
                    swivel["position"]
                )
            )
        )

        max_swivel_velocity = max(
            max_swivel_velocity,
            abs(swivel["velocity"])
        )

        caster_z = get_caster_z(
            robot_id
        )

        min_caster_z = min(
            min_caster_z,
            caster_z
        )

        max_caster_z = max(
            max_caster_z,
            caster_z
        )

        contacts, caster_force = (
            get_caster_contacts(
                robot_id,
                plane_id
            )
        )

        if contacts > 0:
            contact_steps += 1

        total_caster_force += caster_force

        max_caster_force = max(
            max_caster_force,
            caster_force
        )

        # ----------------------------------------------------
        # Progress output
        # ----------------------------------------------------

        if step in (
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000
        ):

            print(
                f"Step {step:4d} | "
                f"X={base['x']: .5f} | "
                f"Z={base['z']: .5f} | "
                f"Pitch={base['pitch']: .4f}° | "
                f"LVel={left['velocity']: .4f} | "
                f"RVel={right['velocity']: .4f} | "
                f"Swivel={math.degrees(swivel['position']): .3f}° | "
                f"CasterZ={caster_z:.5f} | "
                f"Contacts={contacts} | "
                f"Force={caster_force:.4f} N"
            )

    # ========================================================
    # FINAL
    # ========================================================

    final = get_base_state(
        robot_id
    )

    left = get_joint_state(
        robot_id,
        LEFT_WHEEL_JOINT
    )

    right = get_joint_state(
        robot_id,
        RIGHT_WHEEL_JOINT
    )

    swivel = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    final_contacts, final_caster_force = (
        get_caster_contacts(
            robot_id,
            plane_id
        )
    )

    contact_fraction = (
        contact_steps / MOTION_STEPS
    )

    average_caster_force = (
        total_caster_force / MOTION_STEPS
    )

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)

    print(
        f"Motor force              : "
        f"{motor_force:.2f} N"
    )

    print(
        f"Target wheel velocity    : "
        f"{TARGET_LEFT_VELOCITY:.2f} rad/s"
    )

    print(
        f"Final base X             : "
        f"{final['x']:.6f} m"
    )

    print(
        f"Final base Z             : "
        f"{final['z']:.6f} m"
    )

    print(
        f"Final pitch              : "
        f"{final['pitch']:.6f}°"
    )

    print(
        f"Maximum |pitch|          : "
        f"{max_pitch:.6f}°"
    )

    print(
        f"Maximum |roll|           : "
        f"{max_roll:.6f}°"
    )

    print(
        f"Base Z range             : "
        f"{min_base_z:.6f} → "
        f"{max_base_z:.6f} m"
    )

    print()
    print(
        f"Final left wheel vel     : "
        f"{left['velocity']:.6f} rad/s"
    )

    print(
        f"Final right wheel vel    : "
        f"{right['velocity']:.6f} rad/s"
    )

    print(
        f"Max left velocity error  : "
        f"{max_left_velocity_error:.6f} rad/s"
    )

    print(
        f"Max right velocity error : "
        f"{max_right_velocity_error:.6f} rad/s"
    )

    print(
        f"Max left motor torque    : "
        f"{max_left_motor_torque:.6f} N·m"
    )

    print(
        f"Max right motor torque   : "
        f"{max_right_motor_torque:.6f} N·m"
    )

    print(
        f"Max left reaction force  : "
        f"{max_left_reaction:.6f} N"
    )

    print(
        f"Max right reaction force : "
        f"{max_right_reaction:.6f} N"
    )

    print()
    print(
        f"Final caster swivel      : "
        f"{math.degrees(swivel['position']):.6f}°"
    )

    print(
        f"Maximum |caster swivel|  : "
        f"{max_swivel:.6f}°"
    )

    print(
        f"Maximum swivel velocity  : "
        f"{max_swivel_velocity:.6f} rad/s"
    )

    print(
        f"Caster Z range           : "
        f"{min_caster_z:.6f} → "
        f"{max_caster_z:.6f} m"
    )

    print(
        f"Caster contact           : "
        f"{contact_fraction * 100:.2f}%"
    )

    print(
        f"Final caster contacts    : "
        f"{final_contacts}"
    )

    print(
        f"Average caster force     : "
        f"{average_caster_force:.6f} N"
    )

    print(
        f"Maximum caster force     : "
        f"{max_caster_force:.6f} N"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    # Production-like failure.
    if (
        motor_force >= 20.0
        and max_pitch > 10.0
        and contact_fraction < 0.50
    ):

        print(
            "[REFERENCE] Production-force reverse instability "
            "reproduced."
        )

    # Stable case.
    elif (
        max_pitch < 1.0
        and contact_fraction >= 0.99
    ):

        print(
            "[STABLE] Drive-wheel actuation did not produce "
            "the reverse pitch instability at this force."
        )

    # Pitch but caster remains grounded.
    elif (
        max_pitch >= 1.0
        and contact_fraction >= 0.99
    ):

        print(
            "[INVESTIGATE] Significant pitch occurs while "
            "the caster remains in contact."
        )

    # Caster loses contact.
    elif contact_fraction < 0.99:

        print(
            "[FAIL] Caster loses ground contact."
        )

    else:

        print(
            "[INVESTIGATE] Unexpected result."
        )

    return {
        "motor_force": motor_force,
        "final_x": final["x"],
        "final_z": final["z"],
        "final_pitch": final["pitch"],
        "max_pitch": max_pitch,
        "max_roll": max_roll,
        "contact_fraction": contact_fraction,
        "average_caster_force": average_caster_force,
        "max_caster_force": max_caster_force,
        "max_swivel": max_swivel,
        "max_swivel_velocity": max_swivel_velocity,
        "left_velocity": left["velocity"],
        "right_velocity": right["velocity"],
        "left_motor_torque": max_left_motor_torque,
        "right_motor_torque": max_right_motor_torque,
        "left_reaction": max_left_reaction,
        "right_reaction": max_right_reaction,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "V3.22.5q — DRIVE-WHEEL MOTOR-FORCE ISOLATION TEST"
    )
    print("=" * 100)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Production robot.py import:")
    print("PASS")

    print()
    print("Test matrix:")
    print(
        "  Target left velocity  : "
        f"{TARGET_LEFT_VELOCITY} rad/s"
    )

    print(
        "  Target right velocity : "
        f"{TARGET_RIGHT_VELOCITY} rad/s"
    )

    print(
        "  Motor forces          : "
        f"{MOTOR_FORCES}"
    )

    print()
    print("Test conditions:")
    print("  Ground plane          : YES")
    print("  Caster motors         : DISABLED")
    print("  Base motion           : DYNAMIC")
    print("  Drive wheels          : VELOCITY CONTROL")
    print("  Gravity               : -9.81 m/s²")
    print("  Time step             : 1/240 s")
    print("  Motion duration       : 1000 steps")

    physics_client = p.connect(
        p.GUI
    )

    if physics_client < 0:

        print(
            "[FAIL] Could not connect to PyBullet."
        )

        return

    robot_id = None

    try:

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0.0,
            0.0,
            GRAVITY
        )

        p.setTimeStep(
            TIME_STEP
        )

        plane_id = create_ground()

        print()
        print("[GROUND]")
        print(
            f"Plane ID: {plane_id}"
        )

        robot_id = create_robot()

        print()
        print("[ROBOT]")
        print(
            f"Robot ID: {robot_id}"
        )

        results = []

        for motor_force in MOTOR_FORCES:

            result = run_test(
                robot_id,
                plane_id,
                motor_force
            )

            if result is not None:

                results.append(
                    result
                )

            time.sleep(0.5)

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 100)
        print("V3.22.5q SUMMARY")
        print("=" * 100)

        print()

        for result in results:

            print(
                f"Force={result['motor_force']:5.1f} N | "
                f"Pitch={result['max_pitch']:9.4f}° | "
                f"Contact={result['contact_fraction'] * 100:7.2f}% | "
                f"X={result['final_x']:8.4f} m | "
                f"LVel={result['left_velocity']:8.4f} | "
                f"RVel={result['right_velocity']:8.4f} | "
                f"AvgCasterForce={result['average_caster_force']:7.4f} N"
            )

        print()
        print("=" * 100)
        print("V3.22.5q COMPLETE")
        print("=" * 100)

    finally:

        if robot_id is not None:

            stop_robot(
                robot_id
            )

        p.disconnect()


if __name__ == "__main__":
    main()