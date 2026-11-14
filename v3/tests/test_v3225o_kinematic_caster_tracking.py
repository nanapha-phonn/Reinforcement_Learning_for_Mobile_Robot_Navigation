"""
V3.22.5o — Kinematic Caster Tracking Test

PURPOSE
-------
Determine whether the passive caster can physically track backward
motion of the chassis while maintaining ground contact.

This test:
    1. Uses the production robot.py.
    2. Creates a proper ground plane.
    3. Does NOT modify robot.py.
    4. Does NOT use drive-wheel motor commands.
    5. Moves the robot base backward kinematically.
    6. Keeps the chassis orientation level.
    7. Measures caster swivel, wheel rotation, contact, and force.

IMPORTANT
---------
This is a diagnostic test only.

DO NOT:
    - modify robot.py
    - retrain PPO
    - change production geometry
    - change production joint configuration
"""


# ============================================================
# IMPORTS
# ============================================================

import math
import sys
from pathlib import Path

import pybullet as p


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PRODUCTION ROBOT IMPORT
# ============================================================

from robot import create_robot, stop_robot


# ============================================================
# CONFIGURATION
# ============================================================

TIME_STEP = 1.0 / 240.0

GRAVITY = -9.81

TOTAL_STEPS = 1000

TOTAL_BACKWARD_DISTANCE = 1.0

DX = TOTAL_BACKWARD_DISTANCE / TOTAL_STEPS

CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

EXPECTED_BASE_Z = 0.13

INITIAL_Z_TOLERANCE = 0.02


# ============================================================
# HELPER — BASE ATTITUDE
# ============================================================

def get_base_attitude(robot_id):

    _, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


# ============================================================
# HELPER — CASTER STATE
# ============================================================

def get_caster_state(robot_id):

    # --------------------------------------------------------
    # Swivel joint
    # --------------------------------------------------------

    swivel_state = p.getJointState(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    swivel_angle_rad = swivel_state[0]

    swivel_velocity = swivel_state[1]

    swivel_angle_deg = math.degrees(
        swivel_angle_rad
    )

    # --------------------------------------------------------
    # Caster wheel joint
    # --------------------------------------------------------

    wheel_state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    wheel_angle = wheel_state[0]

    wheel_velocity = wheel_state[1]

    # --------------------------------------------------------
    # Caster wheel world position
    # --------------------------------------------------------

    link_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True
    )

    caster_position = link_state[4]

    # --------------------------------------------------------
    # Ground contacts
    # --------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=robot_id,
        linkIndexA=CASTER_WHEEL_JOINT
    )

    # --------------------------------------------------------
    # Total normal force
    # --------------------------------------------------------

    total_force = sum(
        contact[9]
        for contact in contacts
    )

    return {
        "swivel_deg": swivel_angle_deg,
        "swivel_velocity": swivel_velocity,
        "wheel_angle": wheel_angle,
        "wheel_velocity": wheel_velocity,
        "caster_position": caster_position,
        "contacts": len(contacts),
        "force": total_force,
    }


# ============================================================
# HELPER — CREATE GROUND
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
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("V3.22.5o — KINEMATIC CASTER TRACKING TEST")
    print("=" * 100)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Production robot.py import:")
    print("PASS")

    # --------------------------------------------------------
    # CONNECT PYBULLET
    # --------------------------------------------------------

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setGravity(
        0,
        0,
        GRAVITY
    )

    p.setTimeStep(
        TIME_STEP
    )

    # --------------------------------------------------------
    # CREATE GROUND
    # --------------------------------------------------------

    plane_id = create_ground()

    print()
    print("[GROUND]")
    print(f"Plane ID: {plane_id}")

    # --------------------------------------------------------
    # CREATE PRODUCTION ROBOT
    # --------------------------------------------------------

    robot_id = create_robot()

    print()
    print("[CONFIG]")
    print("Production robot.py")
    print("Ground plane created")
    print("No drive-wheel velocity command")
    print("No caster motor command")
    print("Base translated backward kinematically")
    print("Base orientation held level")
    print(
        f"Total backward distance: "
        f"{TOTAL_BACKWARD_DISTANCE:.3f} m"
    )

    # --------------------------------------------------------
    # INITIAL SETTLE
    # --------------------------------------------------------

    for _ in range(240):
        p.stepSimulation()

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    base_position, base_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    roll, pitch, yaw = get_base_attitude(
        robot_id
    )

    caster = get_caster_state(
        robot_id
    )

    print()
    print("=" * 100)
    print("INITIAL STATE")
    print("=" * 100)

    print(
        f"X={base_position[0]: .5f} | "
        f"Z={base_position[2]: .5f} | "
        f"Roll={roll: .4f}° | "
        f"Pitch={pitch: .4f}° | "
        f"Yaw={yaw: .4f}° | "
        f"Swivel={caster['swivel_deg']: .4f}° | "
        f"SwivelVel={caster['swivel_velocity']: .6f} | "
        f"WheelVel={caster['wheel_velocity']: .6f} | "
        f"CasterX={caster['caster_position'][0]: .5f} | "
        f"CasterZ={caster['caster_position'][2]: .5f} | "
        f"Contacts={caster['contacts']} | "
        f"Force={caster['force']:.4f} N"
    )

    # --------------------------------------------------------
    # INITIAL STATE VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("INITIAL STATE VALIDATION")
    print("=" * 100)

    z_error = abs(
        base_position[2] - EXPECTED_BASE_Z
    )

    if z_error > INITIAL_Z_TOLERANCE:

        print(
            f"[ABORT] Invalid initial base height."
        )

        print(
            f"Expected Z ≈ {EXPECTED_BASE_Z:.3f} m"
        )

        print(
            f"Actual Z   = {base_position[2]:.6f} m"
        )

        print(
            "The kinematic caster test will NOT "
            "continue because the robot is not "
            "physically initialized correctly."
        )

        stop_robot(robot_id)

        p.disconnect()

        return

    if caster["contacts"] == 0:

        print(
            "[WARNING] Caster has no ground contact "
            "after settling."
        )

        print(
            "The test will continue only if the "
            "base height is valid."
        )

    else:

        print(
            "[PASS] Initial base height is valid."
        )

    # --------------------------------------------------------
    # INITIAL VALUES
    # --------------------------------------------------------

    initial_base_x = base_position[0]

    initial_orientation = (
        p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        )
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    max_pitch = 0.0

    max_swivel = 0.0

    max_swivel_velocity = 0.0

    min_caster_z = float("inf")

    max_caster_z = float("-inf")

    min_caster_x = float("inf")

    max_caster_x = float("-inf")

    contact_steps = 0

    total_force = 0.0

    max_force = 0.0

    # --------------------------------------------------------
    # BACKWARD KINEMATIC TRANSLATION
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("BACKWARD KINEMATIC MOTION")
    print("=" * 100)

    for step in range(
        1,
        TOTAL_STEPS + 1
    ):

        # ----------------------------------------------------
        # Read current base position.
        # ----------------------------------------------------

        current_position, _ = (
            p.getBasePositionAndOrientation(
                robot_id
            )
        )

        # ----------------------------------------------------
        # Move base backward.
        #
        # Robot forward direction = +X.
        # Therefore backward = decreasing X.
        # ----------------------------------------------------

        new_position = [
            current_position[0] - DX,
            current_position[1],
            current_position[2],
        ]

        # ----------------------------------------------------
        # Keep chassis perfectly level.
        #
        # This isolates caster tracking.
        # ----------------------------------------------------

        p.resetBasePositionAndOrientation(
            robot_id,
            new_position,
            initial_orientation
        )

        # ----------------------------------------------------
        # Advance physics.
        # ----------------------------------------------------

        p.stepSimulation()

        # ----------------------------------------------------
        # Read state.
        # ----------------------------------------------------

        base_position, _ = (
            p.getBasePositionAndOrientation(
                robot_id
            )
        )

        roll, pitch, yaw = (
            get_base_attitude(
                robot_id
            )
        )

        caster = get_caster_state(
            robot_id
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        max_pitch = max(
            max_pitch,
            abs(pitch)
        )

        max_swivel = max(
            max_swivel,
            abs(caster["swivel_deg"])
        )

        max_swivel_velocity = max(
            max_swivel_velocity,
            abs(caster["swivel_velocity"])
        )

        caster_x = caster["caster_position"][0]

        caster_z = caster["caster_position"][2]

        min_caster_x = min(
            min_caster_x,
            caster_x
        )

        max_caster_x = max(
            max_caster_x,
            caster_x
        )

        min_caster_z = min(
            min_caster_z,
            caster_z
        )

        max_caster_z = max(
            max_caster_z,
            caster_z
        )

        if caster["contacts"] > 0:
            contact_steps += 1

        total_force += caster["force"]

        max_force = max(
            max_force,
            caster["force"]
        )

        # ----------------------------------------------------
        # Debug output
        # ----------------------------------------------------

        if step in [
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000,
        ]:

            print(
                f"Step {step:4d} | "
                f"X={base_position[0]: .5f} | "
                f"Pitch={pitch: .4f}° | "
                f"Swivel={caster['swivel_deg']: .4f}° | "
                f"SwivelVel={caster['swivel_velocity']: .6f} | "
                f"WheelAngle={caster['wheel_angle']: .6f} | "
                f"WheelVel={caster['wheel_velocity']: .6f} | "
                f"CasterX={caster_x: .5f} | "
                f"CasterZ={caster_z: .5f} | "
                f"Contacts={caster['contacts']} | "
                f"Force={caster['force']:.4f} N"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    contact_fraction = (
        contact_steps
        / TOTAL_STEPS
        * 100.0
    )

    average_force = (
        total_force
        / TOTAL_STEPS
    )

    final_position, _ = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    final_roll, final_pitch, final_yaw = (
        get_base_attitude(
            robot_id
        )
    )

    final_caster = get_caster_state(
        robot_id
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)

    print(
        f"Initial base X           : "
        f"{initial_base_x:.6f} m"
    )

    print(
        f"Final base X             : "
        f"{final_position[0]:.6f} m"
    )

    print(
        f"Actual displacement      : "
        f"{final_position[0] - initial_base_x:.6f} m"
    )

    print(
        f"Final roll               : "
        f"{final_roll:.6f}°"
    )

    print(
        f"Final pitch              : "
        f"{final_pitch:.6f}°"
    )

    print(
        f"Maximum |pitch|          : "
        f"{max_pitch:.6f}°"
    )

    print()

    print(
        f"Final swivel             : "
        f"{final_caster['swivel_deg']:.6f}°"
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
        f"Final wheel angle        : "
        f"{final_caster['wheel_angle']:.6f} rad"
    )

    print(
        f"Final wheel velocity     : "
        f"{final_caster['wheel_velocity']:.6f} rad/s"
    )

    print()

    print(
        f"Caster X range           : "
        f"{min_caster_x:.6f} → "
        f"{max_caster_x:.6f} m"
    )

    print(
        f"Caster Z range           : "
        f"{min_caster_z:.6f} → "
        f"{max_caster_z:.6f} m"
    )

    print()

    print(
        f"Caster contact           : "
        f"{contact_fraction:.2f}%"
    )

    print(
        f"Final caster contacts    : "
        f"{final_caster['contacts']}"
    )

    print(
        f"Average caster force     : "
        f"{average_force:.6f} N"
    )

    print(
        f"Maximum caster force     : "
        f"{max_force:.6f} N"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    if (
        contact_fraction >= 95.0
        and abs(final_caster["swivel_deg"]) >= 90.0
    ):

        print(
            "[PASS] Caster tracks backward motion."
        )

        print(
            "The caster maintains ground contact "
            "and rotates toward the reverse direction."
        )

    elif contact_fraction < 50.0:

        print(
            "[FAIL] Caster loses ground contact "
            "during kinematic backward translation."
        )

        print(
            "This indicates a caster tracking/contact "
            "problem independent of drive-wheel motor torque."
        )

    else:

        print(
            "[INVESTIGATE] Partial caster tracking."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 100)
    print("V3.22.5o COMPLETE")
    print("=" * 100)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    stop_robot(robot_id)

    p.disconnect()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()