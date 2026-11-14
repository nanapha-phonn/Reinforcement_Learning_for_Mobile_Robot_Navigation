"""
V3.22.5p — DYNAMIC CASTER BACKWARD-MOTION ISOLATION TEST

Purpose
-------
Isolate caster behavior during REAL dynamic backward motion.

The robot is moved backward using base linear velocity.
Drive-wheel motor commands are NOT used.

This preserves:
    - gravity
    - rigid-body dynamics
    - ground contact
    - friction
    - caster contact
    - caster swivel dynamics
    - caster wheel dynamics

This removes:
    - drive-wheel motor torque
    - drive-wheel traction
    - differential-drive interaction

Production robot.py is NOT modified.

Run:
    cd C:\\python_projects\\rl_robot\\v3
    python tests\\test_v3225p_dynamic_caster_reverse.py
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

# Test velocities.
# -0.32 m/s corresponds approximately to:
# wheel radius 0.08 m × wheel speed 4 rad/s
TEST_VELOCITIES = [
    -0.10,
    -0.25,
    -0.32,
    -0.50,
    -1.00,
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
# HELPERS
# ============================================================

def get_euler(robot_id):
    _, orientation = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def get_base_state(robot_id):
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

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


def disable_wheel_motors(robot_id):
    """
    Remove motor actuation from the two drive wheels.

    force=0 means the wheels are not being actively driven.
    """

    for joint_index in (
        LEFT_WHEEL_JOINT,
        RIGHT_WHEEL_JOINT,
    ):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0
        )


def disable_caster_motors(robot_id):
    """
    Ensure caster swivel and caster wheel are not actively driven.
    """

    for joint_index in (
        CASTER_SWIVEL_JOINT,
        CASTER_WHEEL_JOINT,
    ):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0
        )


def stop_all_motors(robot_id):
    disable_wheel_motors(robot_id)
    disable_caster_motors(robot_id)


# ============================================================
# SINGLE VELOCITY TEST
# ============================================================

def run_velocity_test(robot_id, plane_id, target_velocity):
    print()
    print("=" * 100)
    print(
        f"DYNAMIC BACKWARD VELOCITY TEST: "
        f"{target_velocity:.2f} m/s"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Reset robot position/orientation
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        robot_id,
        [0.0, 0.0, EXPECTED_BASE_Z],
        p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    )

    p.resetBaseVelocity(
        robot_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0]
    )

    # Reset joint states.
    for joint_index in range(p.getNumJoints(robot_id)):
        p.resetJointState(
            robot_id,
            joint_index,
            targetValue=0.0,
            targetVelocity=0.0
        )

    stop_all_motors(robot_id)

    # --------------------------------------------------------
    # Settle under gravity
    # --------------------------------------------------------

    for _ in range(SETTLE_STEPS):
        stop_all_motors(robot_id)

        p.resetBaseVelocity(
            robot_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0]
        )

        p.stepSimulation()

    initial = get_base_state(robot_id)

    initial_swivel = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    initial_wheel = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    initial_contacts, initial_force = get_caster_contacts(
        robot_id,
        plane_id
    )

    print()
    print("INITIAL DYNAMIC STATE")
    print("-" * 100)

    print(
        f"Base X={initial['x']: .5f} | "
        f"Z={initial['z']: .5f} | "
        f"Pitch={initial['pitch']: .4f}° | "
        f"Yaw={initial['yaw']: .4f}° | "
        f"Vx={initial['vx']: .5f} m/s"
    )

    print(
        f"Swivel={math.degrees(initial_swivel['position']): .4f}° | "
        f"SwivelVel={initial_swivel['velocity']: .6f} rad/s"
    )

    print(
        f"WheelAngle={initial_wheel['position']: .6f} rad | "
        f"WheelVel={initial_wheel['velocity']: .6f} rad/s"
    )

    print(
        f"Contacts={initial_contacts} | "
        f"Force={initial_force:.4f} N"
    )

    # --------------------------------------------------------
    # Validate starting state
    # --------------------------------------------------------

    if abs(initial["z"] - EXPECTED_BASE_Z) > BASE_Z_TOLERANCE:
        print()
        print("[FAIL] Invalid initial base height.")
        print(
            f"Expected approximately {EXPECTED_BASE_Z:.3f} m, "
            f"got {initial['z']:.5f} m"
        )
        return None

    print()
    print("[PASS] Initial base height is valid.")

    # --------------------------------------------------------
    # Dynamic backward motion
    # --------------------------------------------------------

    max_pitch = 0.0
    max_roll = 0.0

    max_swivel = 0.0
    max_swivel_velocity = 0.0

    max_wheel_velocity = 0.0

    min_caster_z = float("inf")
    max_caster_z = float("-inf")

    total_caster_force = 0.0
    max_caster_force = 0.0

    contact_steps = 0

    initial_x = initial["x"]

    print()
    print("DYNAMIC BACKWARD MOTION")
    print("-" * 100)

    for step in range(1, MOTION_STEPS + 1):

        # No drive-wheel motor command.
        stop_all_motors(robot_id)

        # Physically command the base velocity.
        #
        # This is NOT teleportation.
        p.resetBaseVelocity(
            robot_id,
            linearVelocity=[
                target_velocity,
                0.0,
                0.0
            ],
            angularVelocity=[
                0.0,
                0.0,
                0.0
            ]
        )

        p.stepSimulation()

        # ----------------------------------------------------
        # Read base state
        # ----------------------------------------------------

        base = get_base_state(robot_id)

        # ----------------------------------------------------
        # Read caster joints
        # ----------------------------------------------------

        swivel = get_joint_state(
            robot_id,
            CASTER_SWIVEL_JOINT
        )

        caster_wheel = get_joint_state(
            robot_id,
            CASTER_WHEEL_JOINT
        )

        # ----------------------------------------------------
        # Read caster world state
        # ----------------------------------------------------

        caster_position, _ = p.getLinkState(
            robot_id,
            3,
            computeForwardKinematics=True
        )[4:6]

        caster_z = caster_position[2]

        # ----------------------------------------------------
        # Contact
        # ----------------------------------------------------

        contacts, caster_force = get_caster_contacts(
            robot_id,
            plane_id
        )

        if contacts > 0:
            contact_steps += 1

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        max_pitch = max(
            max_pitch,
            abs(base["pitch"])
        )

        max_roll = max(
            max_roll,
            abs(base["roll"])
        )

        max_swivel = max(
            max_swivel,
            abs(math.degrees(swivel["position"]))
        )

        max_swivel_velocity = max(
            max_swivel_velocity,
            abs(swivel["velocity"])
        )

        max_wheel_velocity = max(
            max_wheel_velocity,
            abs(caster_wheel["velocity"])
        )

        min_caster_z = min(
            min_caster_z,
            caster_z
        )

        max_caster_z = max(
            max_caster_z,
            caster_z
        )

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
                f"Vx={base['vx']: .5f} | "
                f"Z={base['z']: .5f} | "
                f"Pitch={base['pitch']: .4f}° | "
                f"Swivel={math.degrees(swivel['position']): .4f}° | "
                f"SwivelVel={swivel['velocity']: .6f} | "
                f"WheelVel={caster_wheel['velocity']: .6f} | "
                f"CasterZ={caster_z: .5f} | "
                f"Contacts={contacts} | "
                f"Force={caster_force:.4f} N"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    final = get_base_state(robot_id)

    final_swivel = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    final_wheel = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    final_contacts, final_force = get_caster_contacts(
        robot_id,
        plane_id
    )

    displacement = final["x"] - initial_x

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
        f"Target velocity          : "
        f"{target_velocity:.3f} m/s"
    )

    print(
        f"Initial base X           : "
        f"{initial_x:.6f} m"
    )

    print(
        f"Final base X             : "
        f"{final['x']:.6f} m"
    )

    print(
        f"Actual displacement      : "
        f"{displacement:.6f} m"
    )

    print(
        f"Final base velocity      : "
        f"{final['vx']:.6f} m/s"
    )

    print(
        f"Final roll               : "
        f"{final['roll']:.6f}°"
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

    print()
    print(
        f"Final swivel             : "
        f"{math.degrees(final_swivel['position']):.6f}°"
    )

    print(
        f"Maximum |swivel|         : "
        f"{max_swivel:.6f}°"
    )

    print(
        f"Maximum |swivel velocity|: "
        f"{max_swivel_velocity:.6f} rad/s"
    )

    print()
    print(
        f"Final caster wheel angle : "
        f"{final_wheel['position']:.6f} rad"
    )

    print(
        f"Final caster wheel vel   : "
        f"{final_wheel['velocity']:.6f} rad/s"
    )

    print(
        f"Maximum caster wheel vel : "
        f"{max_wheel_velocity:.6f} rad/s"
    )

    print()
    print(
        f"Caster Z range           : "
        f"{min_caster_z:.6f} → {max_caster_z:.6f} m"
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

    if contact_fraction >= 0.99 and max_pitch < 1.0:
        print(
            "[PASS] Caster remains dynamically supported "
            "during backward base motion."
        )
        print(
            "Drive-wheel motor/traction interaction is "
            "therefore strongly implicated in the original "
            "reverse instability."
        )

    elif contact_fraction >= 0.99 and max_pitch >= 1.0:
        print(
            "[INVESTIGATE] Caster maintains contact, "
            "but the chassis develops significant pitch."
        )
        print(
            "The cause is likely dynamic rigid-body/contact "
            "interaction rather than simple caster geometry."
        )

    elif contact_fraction < 0.99:
        print(
            "[FAIL] Caster loses ground contact during "
            "dynamic backward base motion."
        )
        print(
            "The reverse instability can occur even without "
            "drive-wheel motor torque."
        )

    else:
        print(
            "[INVESTIGATE] Unexpected dynamic caster behavior."
        )

    return {
        "target_velocity": target_velocity,
        "displacement": displacement,
        "final_vx": final["vx"],
        "final_pitch": final["pitch"],
        "max_pitch": max_pitch,
        "max_roll": max_roll,
        "max_swivel": max_swivel,
        "max_swivel_velocity": max_swivel_velocity,
        "max_wheel_velocity": max_wheel_velocity,
        "caster_z_min": min_caster_z,
        "caster_z_max": max_caster_z,
        "contact_fraction": contact_fraction,
        "average_caster_force": average_caster_force,
        "max_caster_force": max_caster_force,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("V3.22.5p — DYNAMIC CASTER BACKWARD-MOTION ISOLATION TEST")
    print("=" * 100)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Production robot.py import:")
    print("PASS")

    print()
    print("Test conditions:")
    print("  Ground plane              : YES")
    print("  Drive-wheel motors        : DISABLED")
    print("  Caster motors             : DISABLED")
    print("  Base motion               : PHYSICAL VELOCITY")
    print("  Base angular velocity     : ZERO")
    print("  Gravity                   : -9.81 m/s²")
    print("  Time step                 : 1/240 s")
    print("  Motion duration           : 1000 steps")
    print("  Test velocities           :", TEST_VELOCITIES)

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        print("[FAIL] Could not connect to PyBullet.")
        return

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
        print(f"Plane ID: {plane_id}")

        robot_id = create_robot()

        print()
        print("[ROBOT]")
        print(f"Robot ID: {robot_id}")

        print()
        print("=" * 100)
        print("RUNNING DYNAMIC VELOCITY TESTS")
        print("=" * 100)

        results = []

        for velocity in TEST_VELOCITIES:

            result = run_velocity_test(
                robot_id,
                plane_id,
                velocity
            )

            if result is not None:
                results.append(result)

            time.sleep(0.5)

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print("=" * 100)
        print("V3.22.5p SUMMARY")
        print("=" * 100)

        print()

        for result in results:

            print(
                f"Vx={result['target_velocity']: .2f} m/s | "
                f"Displacement={result['displacement']: .4f} m | "
                f"MaxPitch={result['max_pitch']: .4f}° | "
                f"Contact={result['contact_fraction'] * 100: .2f}% | "
                f"AvgCasterForce={result['average_caster_force']: .4f} N | "
                f"MaxSwivel={result['max_swivel']: .4f}° | "
                f"MaxWheelVel={result['max_wheel_velocity']: .6f}"
            )

        print()
        print("=" * 100)
        print("V3.22.5p COMPLETE")
        print("=" * 100)

    finally:

        stop_robot(robot_id)

        p.disconnect()


if __name__ == "__main__":
    main()