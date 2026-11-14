"""
V3.22.5n — Corrected Swivel Joint Freedom Test

Purpose:
    Determine whether the caster swivel joint is responsible for
    the reverse-motion instability.

IMPORTANT:
    Diagnostic test only.

    robot.py is NOT modified.
    PPO is NOT used.
    RL environment is NOT used.

Cases:
    1. BASELINE
       Current production caster.

    2. FREE_SWIVEL
       Disable motor torque on the caster swivel joint.

    3. FORCED_180
       Actively rotate the caster swivel joint toward 180 degrees.

The experiment isolates caster swivel behavior.
"""

import math
import sys
from pathlib import Path

import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import robot


TIME_STEP = 1.0 / 240.0
STEPS = 1000

CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

GROUND_ID = None


# =====================================================================
# BASIC STATE FUNCTIONS
# =====================================================================

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

    return (
        position,
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def get_joint_state(robot_id, joint_index):

    state = p.getJointState(
        robot_id,
        joint_index
    )

    return (
        state[0],
        state[1]
    )


def get_caster_position(robot_id):

    state = p.getLinkState(
        robot_id,
        3,
        computeLinkVelocity=1
    )

    return state[4]


def get_caster_contacts(robot_id):

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=GROUND_ID
    )

    return [
        contact
        for contact in contacts
        if contact[3] == 3
    ]


def get_caster_force(robot_id):

    contacts = get_caster_contacts(
        robot_id
    )

    return sum(
        contact[9]
        for contact in contacts
    )


# =====================================================================
# RESET
# =====================================================================

def reset_robot(robot_id):

    reset_z = (
        robot.WHEEL_RADIUS
        + robot.CHASSIS_HEIGHT / 2.0
    )

    p.resetBasePositionAndOrientation(
        robot_id,
        [
            0.0,
            0.0,
            reset_z
        ],
        p.getQuaternionFromEuler(
            [
                0.0,
                0.0,
                0.0
            ]
        )
    )

    for joint in range(
        p.getNumJoints(robot_id)
    ):

        p.resetJointState(
            robot_id,
            joint,
            targetValue=0.0,
            targetVelocity=0.0
        )


# =====================================================================
# SWIVEL CONFIGURATIONS
# =====================================================================

def configure_baseline(robot_id):

    print(
        "[CONFIG] BASELINE — no swivel modification"
    )


def configure_free_swivel(robot_id):

    """
    Remove all active motor torque from the swivel joint.

    This does NOT modify robot.py.

    VELOCITY_CONTROL with force=0 means the joint is not actively
    driven by a motor.
    """

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=CASTER_SWIVEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0
    )

    print(
        "[CONFIG] FREE SWIVEL — motor force = 0"
    )


def configure_forced_180(robot_id):

    """
    Diagnostic only.

    Force the swivel toward 180 degrees.

    This is NOT a production solution.
    """

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=CASTER_SWIVEL_JOINT,
        controlMode=p.POSITION_CONTROL,
        targetPosition=math.pi,
        force=5.0,
        positionGain=0.5,
        velocityGain=0.3
    )

    print(
        "[CONFIG] FORCED SWIVEL — target = 180 degrees"
    )


# =====================================================================
# PRINT STATE
# =====================================================================

def print_state(
    robot_id,
    step
):

    position, roll, pitch, yaw = (
        get_base_state(
            robot_id
        )
    )

    swivel, swivel_velocity = (
        get_joint_state(
            robot_id,
            CASTER_SWIVEL_JOINT
        )
    )

    wheel_angle, wheel_velocity = (
        get_joint_state(
            robot_id,
            CASTER_WHEEL_JOINT
        )
    )

    caster_position = (
        get_caster_position(
            robot_id
        )
    )

    contacts = get_caster_contacts(
        robot_id
    )

    force = get_caster_force(
        robot_id
    )

    print(
        f"Step {step:4d} | "
        f"X={position[0]: .5f} | "
        f"Z={position[2]: .5f} | "
        f"Pitch={pitch: .4f}° | "
        f"Swivel={math.degrees(swivel): .4f}° | "
        f"SwivelVel={swivel_velocity: .6f} | "
        f"WheelVel={wheel_velocity: .6f} | "
        f"CasterZ={caster_position[2]: .5f} | "
        f"Contacts={len(contacts)} | "
        f"Force={force: .4f} N"
    )


# =====================================================================
# RUN CASE
# =====================================================================

def run_reverse_case(
    name,
    configuration
):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    robot_id = robot.create_robot()

    reset_robot(
        robot_id
    )

    configuration(
        robot_id
    )

    # ---------------------------------------------------------------
    # Allow physics to settle
    # ---------------------------------------------------------------

    for _ in range(120):

        p.stepSimulation()

    print()
    print("INITIAL STATE")

    print_state(
        robot_id,
        0
    )

    # ---------------------------------------------------------------
    # Reverse drive
    # ---------------------------------------------------------------

    robot.set_wheel_velocity(
        robot_id,
        -4.0,
        -4.0
    )

    pitch_values = []
    swivel_values = []
    swivel_velocity_values = []
    caster_z_values = []
    contact_values = []
    force_values = []

    # ---------------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------------

    for step in range(
        1,
        STEPS + 1
    ):

        p.stepSimulation()

        position, roll, pitch, yaw = (
            get_base_state(
                robot_id
            )
        )

        swivel, swivel_velocity = (
            get_joint_state(
                robot_id,
                CASTER_SWIVEL_JOINT
            )
        )

        caster_position = (
            get_caster_position(
                robot_id
            )
        )

        contacts = get_caster_contacts(
            robot_id
        )

        force = get_caster_force(
            robot_id
        )

        pitch_values.append(
            abs(pitch)
        )

        swivel_values.append(
            math.degrees(swivel)
        )

        swivel_velocity_values.append(
            swivel_velocity
        )

        caster_z_values.append(
            caster_position[2]
        )

        contact_values.append(
            len(contacts)
        )

        force_values.append(
            force
        )

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

            print_state(
                robot_id,
                step
            )

    robot.stop_robot(
        robot_id
    )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    max_pitch = max(
        pitch_values
    )

    contact_fraction = (
        sum(
            c > 0
            for c in contact_values
        )
        / len(contact_values)
    )

    print()
    print("-" * 100)
    print("RESULT")
    print("-" * 100)

    print(
        f"Final pitch              : "
        f"{pitch_values[-1]:.6f}°"
    )

    print(
        f"Max |pitch|              : "
        f"{max_pitch:.6f}°"
    )

    print(
        f"Average |pitch|          : "
        f"{sum(pitch_values) / len(pitch_values):.6f}°"
    )

    print(
        f"Final swivel             : "
        f"{swivel_values[-1]:.6f}°"
    )

    print(
        f"Max |swivel|             : "
        f"{max(abs(x) for x in swivel_values):.6f}°"
    )

    print(
        f"Max |swivel velocity|    : "
        f"{max(abs(x) for x in swivel_velocity_values):.6f}"
    )

    print(
        f"Caster Z range           : "
        f"{min(caster_z_values):.6f} → "
        f"{max(caster_z_values):.6f} m"
    )

    print(
        f"Caster contact            : "
        f"{100.0 * contact_fraction:.2f}%"
    )

    print(
        f"Average caster force      : "
        f"{sum(force_values) / len(force_values):.6f} N"
    )

    print(
        f"Maximum caster force      : "
        f"{max(force_values):.6f} N"
    )

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    if (
        max_pitch < 2.0
        and contact_fraction >= 0.95
    ):

        print(
            "[PASS] Reverse stability restored."
        )

    else:

        print(
            "[INVESTIGATE] Reverse instability remains."
        )

    p.removeBody(
        robot_id
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    global GROUND_ID

    print("=" * 100)
    print("V3.22.5n — SWIVEL JOINT FREEDOM TEST")
    print("=" * 100)

    client = p.connect(
        p.DIRECT
    )

    if client < 0:

        print(
            "[FAIL] PyBullet connection failed."
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

        # -----------------------------------------------------------
        # Ground
        # -----------------------------------------------------------

        ground_shape = p.createCollisionShape(
            p.GEOM_PLANE
        )

        GROUND_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape
        )

        # -----------------------------------------------------------
        # CASE 1
        # -----------------------------------------------------------

        run_reverse_case(
            "CASE 1 — BASELINE",
            configure_baseline
        )

        # -----------------------------------------------------------
        # CASE 2
        # -----------------------------------------------------------

        run_reverse_case(
            "CASE 2 — FREE SWIVEL",
            configure_free_swivel
        )

        # -----------------------------------------------------------
        # CASE 3
        # -----------------------------------------------------------

        run_reverse_case(
            "CASE 3 — FORCED 180° SWIVEL",
            configure_forced_180
        )

        # -----------------------------------------------------------
        # Final
        # -----------------------------------------------------------

        print()
        print("=" * 100)
        print("V3.22.5n COMPLETE")
        print("=" * 100)

        print()
        print("robot.py was NOT modified.")
        print("PPO was NOT trained.")

    finally:

        p.disconnect()


if __name__ == "__main__":
    main()

