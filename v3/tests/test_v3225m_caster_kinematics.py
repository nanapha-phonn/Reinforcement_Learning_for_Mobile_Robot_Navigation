"""
V3.22.5m — Caster Kinematics Diagnostic

Purpose:
    Determine why the passive caster loses ground contact during
    reverse motion.

Measurements:
    - caster swivel joint angle
    - caster swivel joint velocity
    - caster wheel joint angle
    - caster wheel joint velocity
    - caster wheel world position
    - caster wheel world orientation
    - caster contact count
    - caster contact force
    - base pitch
    - base roll
    - base position

Tests:
    STATIC
    FORWARD +4/+4
    REVERSE -4/-4
    ROTATION -4/+4

No modification to robot.py.
No PPO.
No RL environment.
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


def get_base_state(robot_id):

    position, orientation = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        position,
        (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        ),
    )


def get_joint_state(robot_id, joint_index):

    state = p.getJointState(
        robot_id,
        joint_index
    )

    return state[0], state[1]


def get_link_state(robot_id, link_index):

    state = p.getLinkState(
        robot_id,
        link_index,
        computeLinkVelocity=1
    )

    world_position = state[4]
    world_orientation = state[5]

    linear_velocity = state[6]
    angular_velocity = state[7]

    return (
        world_position,
        world_orientation,
        linear_velocity,
        angular_velocity,
    )


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


def caster_force(robot_id):

    contacts = get_caster_contacts(
        robot_id
    )

    return sum(
        contact[9]
        for contact in contacts
    )


def print_state(robot_id, step):

    position, attitude = get_base_state(
        robot_id
    )

    swivel_angle, swivel_velocity = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT
    )

    wheel_angle, wheel_velocity = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT
    )

    wheel_position, wheel_orientation, _, wheel_angular_velocity = (
        get_link_state(
            robot_id,
            3
        )
    )

    contacts = get_caster_contacts(
        robot_id
    )

    print(
        f"Step {step:4d} | "
        f"X={position[0]: .5f} | "
        f"Z={position[2]: .5f} | "
        f"Roll={attitude[0]: .4f}° | "
        f"Pitch={attitude[1]: .4f}° | "
        f"Yaw={attitude[2]: .4f}° | "
        f"Swivel={math.degrees(swivel_angle): .4f}° | "
        f"SwivelVel={swivel_velocity: .4f} | "
        f"WheelAng={wheel_angle: .4f} | "
        f"WheelVel={wheel_velocity: .4f} | "
        f"CasterX={wheel_position[0]: .5f} | "
        f"CasterY={wheel_position[1]: .5f} | "
        f"CasterZ={wheel_position[2]: .5f} | "
        f"Contacts={len(contacts)} | "
        f"Force={caster_force(robot_id): .4f} N"
    )


def run_case(
    name,
    left_velocity,
    right_velocity
):

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    robot_id = robot.create_robot()

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

    # --------------------------------------------------------------
    # Initial state
    # --------------------------------------------------------------

    print()
    print("INITIAL STATE")
    print_state(
        robot_id,
        0
    )

    # --------------------------------------------------------------
    # Drive
    # --------------------------------------------------------------

    robot.set_wheel_velocity(
        robot_id,
        left_velocity,
        right_velocity
    )

    swivel_angles = []
    swivel_velocities = []

    wheel_angles = []
    wheel_velocities = []

    caster_z = []
    caster_x = []

    pitch_values = []
    contact_counts = []
    contact_forces = []

    for step in range(
        1,
        STEPS + 1
    ):

        p.stepSimulation()

        swivel_angle, swivel_velocity = get_joint_state(
            robot_id,
            CASTER_SWIVEL_JOINT
        )

        wheel_angle, wheel_velocity = get_joint_state(
            robot_id,
            CASTER_WHEEL_JOINT
        )

        position, attitude = get_base_state(
            robot_id
        )

        wheel_position, _, _, _ = get_link_state(
            robot_id,
            3
        )

        contacts = get_caster_contacts(
            robot_id
        )

        force = caster_force(
            robot_id
        )

        swivel_angles.append(
            math.degrees(swivel_angle)
        )

        swivel_velocities.append(
            swivel_velocity
        )

        wheel_angles.append(
            wheel_angle
        )

        wheel_velocities.append(
            wheel_velocity
        )

        caster_z.append(
            wheel_position[2]
        )

        caster_x.append(
            wheel_position[0]
        )

        pitch_values.append(
            abs(attitude[1])
        )

        contact_counts.append(
            len(contacts)
        )

        contact_forces.append(
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

    print()
    print("-" * 100)
    print("SUMMARY")
    print("-" * 100)

    print(
        f"Final swivel angle       : "
        f"{swivel_angles[-1]:.6f}°"
    )

    print(
        f"Max |swivel angle|       : "
        f"{max(abs(x) for x in swivel_angles):.6f}°"
    )

    print(
        f"Max |swivel velocity|    : "
        f"{max(abs(x) for x in swivel_velocities):.6f}"
    )

    print(
        f"Final wheel angle        : "
        f"{wheel_angles[-1]:.6f} rad"
    )

    print(
        f"Final wheel velocity     : "
        f"{wheel_velocities[-1]:.6f} rad/s"
    )

    print(
        f"Max |wheel velocity|     : "
        f"{max(abs(x) for x in wheel_velocities):.6f} rad/s"
    )

    print(
        f"Caster X range           : "
        f"{min(caster_x):.6f} → "
        f"{max(caster_x):.6f} m"
    )

    print(
        f"Caster Z range           : "
        f"{min(caster_z):.6f} → "
        f"{max(caster_z):.6f} m"
    )

    print(
        f"Minimum caster Z         : "
        f"{min(caster_z):.6f} m"
    )

    print(
        f"Maximum caster Z         : "
        f"{max(caster_z):.6f} m"
    )

    print(
        f"Max |pitch|              : "
        f"{max(pitch_values):.6f}°"
    )

    print(
        f"Caster contact fraction  : "
        f"{100.0 * sum(c > 0 for c in contact_counts) / len(contact_counts):.2f}%"
    )

    print(
        f"Average caster force     : "
        f"{sum(contact_forces) / len(contact_forces):.6f} N"
    )

    print(
        f"Max caster force         : "
        f"{max(contact_forces):.6f} N"
    )

    p.removeBody(
        robot_id
    )


def main():

    print("=" * 100)
    print("V3.22.5m — CASTER KINEMATICS DIAGNOSTIC")
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

        # ----------------------------------------------------------
        # Ground
        # ----------------------------------------------------------

        ground_shape = p.createCollisionShape(
            p.GEOM_PLANE
        )

        global GROUND_ID

        GROUND_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape
        )

        # ----------------------------------------------------------
        # Tests
        # ----------------------------------------------------------

        run_case(
            "STATIC",
            0.0,
            0.0
        )

        run_case(
            "FORWARD +4/+4",
            4.0,
            4.0
        )

        run_case(
            "REVERSE -4/-4",
            -4.0,
            -4.0
        )

        run_case(
            "ROTATION -4/+4",
            -4.0,
            4.0
        )

        print()
        print("=" * 100)
        print("V3.22.5m COMPLETE")
        print("=" * 100)

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Do NOT modify robot.py from this test."
        )

        print(
            "Do NOT retrain PPO yet."
        )

    finally:

        p.disconnect()


if __name__ == "__main__":
    main()