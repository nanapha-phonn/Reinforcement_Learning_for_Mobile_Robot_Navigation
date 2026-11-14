import math
import os
import sys

import pybullet as p
import pybullet_data


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
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
    WHEEL_BASE,
    TIME_STEP,
)

from sensors.imu import read_imu


# ============================================================
# V3.22.4 CONTROLLED-MOTION ATTITUDE DIAGNOSTIC
#
# Purpose:
# Determine whether commanded wheel motion causes abnormal
# roll/pitch behavior after the V3.22.3 geometry correction.
#
# PPO: DISABLED
# Reward: NOT USED
# RL environment: NOT USED
# Sensor noise: NOT USED
#
# Tests:
#   1. STATIC
#   2. FORWARD +2/+2
#   3. FORWARD +4/+4
#   4. REVERSE -2/-2
#   5. REVERSE -4/-4
#   6. ROTATION -2/+2
#   7. ROTATION -4/+4
#   8. CURVE +4/+2
#   9. CURVE +2/+4
#
# ============================================================


# ============================================================
# TEST SETTINGS
# ============================================================

SIMULATION_STEPS = 1000
WARMUP_STEPS = 120

MOTOR_FORCE = 20.0

TEST_COMMANDS = [
    ("STATIC", 0.0, 0.0),
    ("FORWARD +2/+2", 2.0, 2.0),
    ("FORWARD +4/+4", 4.0, 4.0),
    ("REVERSE -2/-2", -2.0, -2.0),
    ("REVERSE -4/-4", -4.0, -4.0),
    ("ROTATION -2/+2", -2.0, 2.0),
    ("ROTATION -4/+4", -4.0, 4.0),
    ("CURVE +4/+2", 4.0, 2.0),
    ("CURVE +2/+4", 2.0, 4.0),
]


# ============================================================
# PYBULLET CONNECTION
# ============================================================

def connect_physics():
    """
    Connect to PyBullet once for the complete diagnostic.
    """

    if p.isConnected():
        p.disconnect()

    physics_client = p.connect(
        p.DIRECT
    )

    if physics_client < 0:
        raise RuntimeError(
            "Failed to connect to the PyBullet physics server."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        -9.81,
    )

    p.setTimeStep(
        TIME_STEP
    )

    print()
    print("PyBullet connection: PASS")
    print(
        f"Physics client ID : {physics_client}"
    )


def disconnect_physics():
    """
    Disconnect from PyBullet after all tests.
    """

    if p.isConnected():
        p.disconnect()

    print()
    print("PyBullet disconnected.")


# ============================================================
# ANGLE HELPERS
# ============================================================

def deg(rad):
    """
    Convert radians to degrees.
    """

    return math.degrees(rad)


# ============================================================
# DIFFERENTIAL-DRIVE THEORY
# ============================================================

def theoretical_yaw_rate(
    left_velocity,
    right_velocity,
):
    """
    Differential-drive theoretical yaw rate:

        omega_z =
            r * (omega_R - omega_L) / L
    """

    left_linear_velocity = (
        WHEEL_RADIUS * left_velocity
    )

    right_linear_velocity = (
        WHEEL_RADIUS * right_velocity
    )

    return (
        right_linear_velocity
        - left_linear_velocity
    ) / WHEEL_BASE


# ============================================================
# CONTACT INFORMATION
# ============================================================

def get_contact_information(
    robot_id,
    plane_id,
):
    """
    Read robot-ground contacts.

    Link convention:

        Link 0 = left wheel
        Link 1 = right wheel
        Link 2 = caster fork
        Link 3 = caster wheel
    """

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
    )

    left_contacts = 0
    right_contacts = 0
    caster_contacts = 0

    normal_forces = []

    for contact in contacts:

        link_index = contact[3]
        normal_force = contact[9]

        normal_forces.append(
            normal_force
        )

        if link_index == 0:
            left_contacts += 1

        elif link_index == 1:
            right_contacts += 1

        elif link_index == 3:
            caster_contacts += 1

    return {
        "left": left_contacts,
        "right": right_contacts,
        "caster": caster_contacts,
        "normal_forces": normal_forces,
    }


# ============================================================
# CREATE TEST WORLD
# ============================================================

def create_test_world():
    """
    Create a fresh simulation world.
    """

    if not p.isConnected():
        raise RuntimeError(
            "PyBullet is not connected."
        )

    p.resetSimulation()

    p.setGravity(
        0.0,
        0.0,
        -9.81,
    )

    p.setTimeStep(
        TIME_STEP
    )

    plane_id = p.loadURDF(
        "plane.urdf"
    )

    robot_id = create_robot()

    # --------------------------------------------------------
    # CORRECT RESET HEIGHT
    #
    # Wheel radius = 0.08 m
    # Chassis height = 0.10 m
    #
    # Chassis center:
    #
    #     0.08 + 0.10 / 2
    #     = 0.13 m
    # --------------------------------------------------------

    chassis_height = 0.10

    reset_z = (
        WHEEL_RADIUS
        + chassis_height / 2.0
    )

    p.resetBasePositionAndOrientation(
        robot_id,
        [
            0.0,
            0.0,
            reset_z,
        ],
        p.getQuaternionFromEuler(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    # robot.py API:
    # stop_robot(robot_id)

    stop_robot(
        robot_id
    )

    return (
        robot_id,
        plane_id,
    )


# ============================================================
# SETTLE ROBOT
# ============================================================

def settle_robot(
    robot_id,
):
    """
    Allow the robot to establish stable ground contact.
    """

    stop_robot(
        robot_id
    )

    for _ in range(
        WARMUP_STEPS
    ):
        p.stepSimulation()


# ============================================================
# RUN ONE MOTION TEST
# ============================================================

def run_motion_test(
    name,
    left_velocity,
    right_velocity,
):

    robot_id, plane_id = (
        create_test_world()
    )

    # --------------------------------------------------------
    # SETTLE
    # --------------------------------------------------------

    settle_robot(
        robot_id
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    (
        initial_position,
        initial_orientation,
    ) = p.getBasePositionAndOrientation(
        robot_id
    )

    (
        initial_roll,
        initial_pitch,
        initial_yaw,
    ) = p.getEulerFromQuaternion(
        initial_orientation
    )

    # --------------------------------------------------------
    # THEORETICAL YAW RATE
    # --------------------------------------------------------

    theoretical_yaw = (
        theoretical_yaw_rate(
            left_velocity,
            right_velocity,
        )
    )

    # --------------------------------------------------------
    # DATA STORAGE
    # --------------------------------------------------------

    yaw_rates = []

    roll_values = []
    pitch_values = []

    positions_x = []
    positions_y = []
    positions_z = []

    left_contact_counts = []
    right_contact_counts = []
    caster_contact_counts = []

    total_normal_forces = []

    # --------------------------------------------------------
    # APPLY WHEEL COMMAND
    #
    # robot.py API:
    #
    # set_wheel_velocity(
    #     robot_id,
    #     left_velocity,
    #     right_velocity
    # )
    # --------------------------------------------------------

    set_wheel_velocity(
        robot_id,
        left_velocity,
        right_velocity,
    )

    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    for step in range(
        SIMULATION_STEPS
    ):

        p.stepSimulation()

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        imu = read_imu(
            robot_id
        )

        roll = imu["roll"]
        pitch = imu["pitch"]
        yaw_rate = imu["angular_vel_z"]

        # ----------------------------------------------------
        # BASE STATE
        # ----------------------------------------------------

        (
            position,
            orientation,
        ) = p.getBasePositionAndOrientation(
            robot_id
        )

        # ----------------------------------------------------
        # CONTACTS
        # ----------------------------------------------------

        contacts = (
            get_contact_information(
                robot_id,
                plane_id,
            )
        )

        # ----------------------------------------------------
        # STORE SENSOR DATA
        # ----------------------------------------------------

        yaw_rates.append(
            yaw_rate
        )

        roll_values.append(
            roll
        )

        pitch_values.append(
            pitch
        )

        positions_x.append(
            position[0]
        )

        positions_y.append(
            position[1]
        )

        positions_z.append(
            position[2]
        )

        left_contact_counts.append(
            contacts["left"]
        )

        right_contact_counts.append(
            contacts["right"]
        )

        caster_contact_counts.append(
            contacts["caster"]
        )

        total_normal_forces.append(
            sum(
                contacts[
                    "normal_forces"
                ]
            )
        )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    stop_robot(
        robot_id
    )

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    (
        final_position,
        final_orientation,
    ) = p.getBasePositionAndOrientation(
        robot_id
    )

    (
        final_roll,
        final_pitch,
        final_yaw,
    ) = p.getEulerFromQuaternion(
        final_orientation
    )

    # --------------------------------------------------------
    # YAW ANALYSIS
    #
    # Ignore initial acceleration period.
    # --------------------------------------------------------

    analysis_start = min(
        WARMUP_STEPS,
        len(yaw_rates) // 4,
    )

    analyzed_yaw_rates = (
        yaw_rates[
            analysis_start:
        ]
    )

    average_yaw_rate = (
        sum(analyzed_yaw_rates)
        / len(analyzed_yaw_rates)
    )

    final_yaw_rate = (
        yaw_rates[-1]
    )

    # --------------------------------------------------------
    # ATTITUDE STATISTICS
    # --------------------------------------------------------

    maximum_roll = max(
        abs(value)
        for value in roll_values
    )

    maximum_pitch = max(
        abs(value)
        for value in pitch_values
    )

    average_roll = (
        sum(
            abs(value)
            for value in roll_values
        )
        / len(roll_values)
    )

    average_pitch = (
        sum(
            abs(value)
            for value in pitch_values
        )
        / len(pitch_values)
    )

    # --------------------------------------------------------
    # CONTACT STATISTICS
    # --------------------------------------------------------

    average_left_contacts = (
        sum(left_contact_counts)
        / len(left_contact_counts)
    )

    average_right_contacts = (
        sum(right_contact_counts)
        / len(right_contact_counts)
    )

    average_caster_contacts = (
        sum(caster_contact_counts)
        / len(caster_contact_counts)
    )

    average_normal_force = (
        sum(total_normal_forces)
        / len(total_normal_forces)
    )

    # --------------------------------------------------------
    # YAW RATIO
    # --------------------------------------------------------

    if abs(theoretical_yaw) > 1e-9:

        yaw_ratio = (
            average_yaw_rate
            / theoretical_yaw
        )

    else:

        yaw_ratio = None

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print(
        f"TEST: {name}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # COMMAND
    # --------------------------------------------------------

    print()
    print("--- COMMAND ---")

    print(
        f"Left wheel command  : "
        f"{left_velocity:.6f} rad/s"
    )

    print(
        f"Right wheel command : "
        f"{right_velocity:.6f} rad/s"
    )

    # --------------------------------------------------------
    # YAW
    # --------------------------------------------------------

    print()
    print("--- YAW RATE ---")

    print(
        f"Theoretical yaw rate : "
        f"{theoretical_yaw:.6f} rad/s"
    )

    print(
        f"Average yaw rate     : "
        f"{average_yaw_rate:.6f} rad/s"
    )

    print(
        f"Final yaw rate       : "
        f"{final_yaw_rate:.6f} rad/s"
    )

    if yaw_ratio is not None:

        print(
            f"Yaw-rate ratio       : "
            f"{yaw_ratio * 100.0:.3f}%"
        )

    # --------------------------------------------------------
    # ATTITUDE
    # --------------------------------------------------------

    print()
    print("--- ATTITUDE ---")

    print(
        f"Initial roll         : "
        f"{deg(initial_roll):.6f} deg"
    )

    print(
        f"Initial pitch        : "
        f"{deg(initial_pitch):.6f} deg"
    )

    print(
        f"Initial yaw          : "
        f"{deg(initial_yaw):.6f} deg"
    )

    print(
        f"Final roll           : "
        f"{deg(final_roll):.6f} deg"
    )

    print(
        f"Final pitch          : "
        f"{deg(final_pitch):.6f} deg"
    )

    print(
        f"Final yaw            : "
        f"{deg(final_yaw):.6f} deg"
    )

    print(
        f"Maximum |roll|       : "
        f"{deg(maximum_roll):.6f} deg"
    )

    print(
        f"Maximum |pitch|      : "
        f"{deg(maximum_pitch):.6f} deg"
    )

    print(
        f"Average |roll|       : "
        f"{deg(average_roll):.6f} deg"
    )

    print(
        f"Average |pitch|      : "
        f"{deg(average_pitch):.6f} deg"
    )

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    print()
    print("--- FINAL POSITION ---")

    print(
        f"X : {final_position[0]:.6f} m"
    )

    print(
        f"Y : {final_position[1]:.6f} m"
    )

    print(
        f"Z : {final_position[2]:.6f} m"
    )

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    print()
    print("--- CONTACTS ---")

    print(
        f"Average left wheel contacts   : "
        f"{average_left_contacts:.3f}"
    )

    print(
        f"Average right wheel contacts  : "
        f"{average_right_contacts:.3f}"
    )

    print(
        f"Average caster contacts       : "
        f"{average_caster_contacts:.3f}"
    )

    print(
        f"Average total normal force    : "
        f"{average_normal_force:.6f} N"
    )

    # ========================================================
    # DIAGNOSTIC FLAGS
    # ========================================================

    print()
    print("--- DIAGNOSTIC FLAGS ---")

    # --------------------------------------------------------
    # PITCH
    # --------------------------------------------------------

    if maximum_pitch < math.radians(5.0):

        print(
            "Pitch stability       : PASS"
        )

    else:

        print(
            "Pitch stability       : INVESTIGATE"
        )

    # --------------------------------------------------------
    # ROLL
    # --------------------------------------------------------

    if maximum_roll < math.radians(5.0):

        print(
            "Roll stability        : PASS"
        )

    else:

        print(
            "Roll stability        : INVESTIGATE"
        )

    # --------------------------------------------------------
    # YAW
    # --------------------------------------------------------

    if theoretical_yaw == 0.0:

        if abs(average_yaw_rate) < 0.05:

            print(
                "Straight-line yaw     : PASS"
            )

        else:

            print(
                "Straight-line yaw     : INVESTIGATE"
            )

    else:

        if abs(yaw_ratio - 1.0) < 0.05:

            print(
                "Yaw-rate agreement    : PASS"
            )

        else:

            print(
                "Yaw-rate agreement    : INVESTIGATE"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "V3.22.4 CONTROLLED-MOTION "
        "ATTITUDE DIAGNOSTIC"
    )
    print("=" * 90)

    print()
    print("Purpose:")
    print(
        "Determine whether commanded wheel motion "
        "causes abnormal roll/pitch behavior."
    )

    print()
    print("PPO: DISABLED")
    print("Reward: NOT USED")
    print("RL environment: NOT USED")
    print("Sensor noise: NOT USED")
    print("Physics parameters: UNCHANGED")

    print()
    print(
        f"Wheel radius : "
        f"{WHEEL_RADIUS:.3f} m"
    )

    print(
        f"Wheel base   : "
        f"{WHEEL_BASE:.3f} m"
    )

    print(
        f"Simulation steps per test : "
        f"{SIMULATION_STEPS}"
    )

    # --------------------------------------------------------
    # CONNECT ONCE
    # --------------------------------------------------------

    connect_physics()

    try:

        # ----------------------------------------------------
        # RUN ALL 9 TESTS
        # ----------------------------------------------------

        for (
            name,
            left_velocity,
            right_velocity,
        ) in TEST_COMMANDS:

            run_motion_test(
                name,
                left_velocity,
                right_velocity,
            )

    finally:

        # ----------------------------------------------------
        # DISCONNECT ONCE
        # ----------------------------------------------------

        disconnect_physics()

    print()
    print("=" * 90)
    print(
        "V3.22.4 CONTROLLED-MOTION "
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 90)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()