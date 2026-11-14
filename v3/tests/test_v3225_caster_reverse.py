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
    WHEEL_BASE,
    TIME_STEP,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

SIMULATION_STEPS = 1000
WARMUP_STEPS = 120

FORWARD_LEFT = 4.0
FORWARD_RIGHT = 4.0

REVERSE_LEFT = -4.0
REVERSE_RIGHT = -4.0

GRAVITY = -9.81

CHASSIS_HEIGHT = 0.10

# Corrected V3 reset height:
# wheel center = 0.08 m
# chassis center = 0.08 + 0.10/2 = 0.13 m
RESET_BASE_Z = WHEEL_RADIUS + CHASSIS_HEIGHT / 2.0

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1
CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

LEFT_WHEEL_LINK = 0
RIGHT_WHEEL_LINK = 1
CASTER_FORK_LINK = 2
CASTER_WHEEL_LINK = 3

DEG = 180.0 / math.pi


# ============================================================
# PYBULLET CONNECTION
# ============================================================

def connect_physics():
    if p.isConnected():
        p.disconnect()

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, GRAVITY)
    p.setTimeStep(TIME_STEP)

    print("PyBullet connection: PASS")
    print("Physics client ID :", client_id)


# ============================================================
# CREATE WORLD
# ============================================================

def create_test_world():
    p.resetSimulation()

    p.setGravity(0.0, 0.0, GRAVITY)
    p.setTimeStep(TIME_STEP)

    plane_id = p.loadURDF("plane.urdf")

    robot_id = create_robot()

    # Correct V3 reset height.
    p.resetBasePositionAndOrientation(
        robot_id,
        [0.0, 0.0, RESET_BASE_Z],
        p.getQuaternionFromEuler(
            [0.0, 0.0, 0.0]
        ),
    )

    stop_robot(robot_id)

    # Allow initial contacts to settle.
    for _ in range(WARMUP_STEPS):
        p.stepSimulation()

    return plane_id, robot_id


# ============================================================
# CONTACT FUNCTIONS
# ============================================================

def get_link_contacts(
    robot_id,
    plane_id,
    link_index,
):
    """
    Return robot-to-ground contacts for one link.

    IMPORTANT:
    bodyB must be a valid PyBullet body ID.
    Passing bodyB=None causes:
        TypeError: 'NoneType' object cannot be interpreted
        as an integer
    """

    return p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=link_index,
    )


def get_link_contact_count(
    robot_id,
    plane_id,
    link_index,
):
    contacts = get_link_contacts(
        robot_id,
        plane_id,
        link_index,
    )

    return len(contacts)


def get_link_normal_force(
    robot_id,
    plane_id,
    link_index,
):
    contacts = get_link_contacts(
        robot_id,
        plane_id,
        link_index,
    )

    total_force = 0.0

    for contact in contacts:
        # contact[9] = normal force
        total_force += contact[9]

    return total_force


def get_total_normal_force(
    robot_id,
    plane_id,
):
    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
    )

    total_force = 0.0

    for contact in contacts:
        total_force += contact[9]

    return total_force


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
# ROBOT STATE
# ============================================================

def read_robot_state(
    robot_id,
    plane_id,
):
    position, orientation = (
        p.getBasePositionAndOrientation(robot_id)
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(orientation)
    )

    linear_velocity, angular_velocity = (
        p.getBaseVelocity(robot_id)
    )

    caster_swivel = get_joint_state(
        robot_id,
        CASTER_SWIVEL_JOINT,
    )

    caster_wheel = get_joint_state(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    left_wheel = get_joint_state(
        robot_id,
        LEFT_WHEEL_JOINT,
    )

    right_wheel = get_joint_state(
        robot_id,
        RIGHT_WHEEL_JOINT,
    )

    return {
        # Position
        "x": position[0],
        "y": position[1],
        "z": position[2],

        # Attitude
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,

        # Base velocity
        "linear_velocity_x": linear_velocity[0],
        "linear_velocity_y": linear_velocity[1],
        "linear_velocity_z": linear_velocity[2],

        "angular_velocity_x": angular_velocity[0],
        "angular_velocity_y": angular_velocity[1],
        "angular_velocity_z": angular_velocity[2],

        # Caster swivel
        "caster_swivel_angle": (
            caster_swivel["position"]
        ),
        "caster_swivel_velocity": (
            caster_swivel["velocity"]
        ),

        # Caster wheel
        "caster_wheel_angle": (
            caster_wheel["position"]
        ),
        "caster_wheel_velocity": (
            caster_wheel["velocity"]
        ),

        # Drive wheel velocities
        "left_wheel_velocity": (
            left_wheel["velocity"]
        ),
        "right_wheel_velocity": (
            right_wheel["velocity"]
        ),

        # Contacts
        "left_contacts": get_link_contact_count(
            robot_id,
            plane_id,
            LEFT_WHEEL_LINK,
        ),

        "right_contacts": get_link_contact_count(
            robot_id,
            plane_id,
            RIGHT_WHEEL_LINK,
        ),

        "caster_fork_contacts": (
            get_link_contact_count(
                robot_id,
                plane_id,
                CASTER_FORK_LINK,
            )
        ),

        "caster_wheel_contacts": (
            get_link_contact_count(
                robot_id,
                plane_id,
                CASTER_WHEEL_LINK,
            )
        ),

        # Normal forces
        "left_normal_force": (
            get_link_normal_force(
                robot_id,
                plane_id,
                LEFT_WHEEL_LINK,
            )
        ),

        "right_normal_force": (
            get_link_normal_force(
                robot_id,
                plane_id,
                RIGHT_WHEEL_LINK,
            )
        ),

        "caster_normal_force": (
            get_link_normal_force(
                robot_id,
                plane_id,
                CASTER_WHEEL_LINK,
            )
        ),

        "total_normal_force": (
            get_total_normal_force(
                robot_id,
                plane_id,
            )
        ),
    }


# ============================================================
# MOTION TEST
# ============================================================

def run_motion_test(
    test_name,
    left_command,
    right_command,
):
    print()
    print("=" * 100)
    print("TEST:", test_name)
    print("=" * 100)

    print()
    print("--- COMMAND ---")
    print(
        f"Left wheel command  : "
        f"{left_command:.6f} rad/s"
    )
    print(
        f"Right wheel command : "
        f"{right_command:.6f} rad/s"
    )

    plane_id, robot_id = create_test_world()

    initial_state = read_robot_state(
        robot_id,
        plane_id,
    )

    states = []

    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    min_z = float("inf")
    max_z = float("-inf")

    max_abs_swivel = 0.0
    min_swivel = float("inf")
    max_swivel = float("-inf")

    # --------------------------------------------------------
    # Apply wheel command
    # --------------------------------------------------------

    set_wheel_velocity(
        robot_id,
        left_command,
        right_command,
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    sample_steps = {
        0,
        1,
        10,
        50,
        100,
        120,
        200,
        300,
        500,
        750,
        999,
    }

    for step in range(SIMULATION_STEPS):

        p.stepSimulation()

        state = read_robot_state(
            robot_id,
            plane_id,
        )

        states.append(state)

        max_abs_roll = max(
            max_abs_roll,
            abs(state["roll"]),
        )

        max_abs_pitch = max(
            max_abs_pitch,
            abs(state["pitch"]),
        )

        min_z = min(
            min_z,
            state["z"],
        )

        max_z = max(
            max_z,
            state["z"],
        )

        swivel = state[
            "caster_swivel_angle"
        ]

        max_abs_swivel = max(
            max_abs_swivel,
            abs(swivel),
        )

        min_swivel = min(
            min_swivel,
            swivel,
        )

        max_swivel = max(
            max_swivel,
            swivel,
        )

        if step in sample_steps:

            print(
                f"Step {step + 1:4d} | "
                f"X={state['x']:8.4f} | "
                f"Y={state['y']:8.4f} | "
                f"Z={state['z']:8.4f} | "
                f"Pitch={state['pitch'] * DEG:9.3f}° | "
                f"Yaw={state['yaw'] * DEG:9.3f}° | "
                f"Swivel={swivel * DEG:9.3f}° | "
                f"CasterContact="
                f"{state['caster_wheel_contacts']}"
            )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    stop_robot(robot_id)

    for _ in range(120):
        p.stepSimulation()

    final_state = read_robot_state(
        robot_id,
        plane_id,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_abs_pitch = (
        sum(
            abs(state["pitch"])
            for state in states
        )
        / len(states)
    )

    average_abs_roll = (
        sum(
            abs(state["roll"])
            for state in states
        )
        / len(states)
    )

    average_abs_swivel = (
        sum(
            abs(
                state["caster_swivel_angle"]
            )
            for state in states
        )
        / len(states)
    )

    caster_contact_fraction = (
        sum(
            1
            for state in states
            if state["caster_wheel_contacts"] > 0
        )
        / len(states)
    )

    average_caster_force = (
        sum(
            state["caster_normal_force"]
            for state in states
        )
        / len(states)
    )

    average_total_force = (
        sum(
            state["total_normal_force"]
            for state in states
        )
        / len(states)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("--- ATTITUDE ---")

    print(
        f"Initial pitch          : "
        f"{initial_state['pitch'] * DEG:.6f} deg"
    )

    print(
        f"Final pitch            : "
        f"{final_state['pitch'] * DEG:.6f} deg"
    )

    print(
        f"Maximum |pitch|        : "
        f"{max_abs_pitch * DEG:.6f} deg"
    )

    print(
        f"Average |pitch|        : "
        f"{average_abs_pitch * DEG:.6f} deg"
    )

    print(
        f"Initial roll           : "
        f"{initial_state['roll'] * DEG:.6f} deg"
    )

    print(
        f"Final roll             : "
        f"{final_state['roll'] * DEG:.6f} deg"
    )

    print(
        f"Maximum |roll|         : "
        f"{max_abs_roll * DEG:.6f} deg"
    )

    print(
        f"Average |roll|         : "
        f"{average_abs_roll * DEG:.6f} deg"
    )

    print()
    print("--- CASTER SWIVEL ---")

    print(
        f"Initial swivel         : "
        f"{initial_state['caster_swivel_angle'] * DEG:.6f} deg"
    )

    print(
        f"Final swivel           : "
        f"{final_state['caster_swivel_angle'] * DEG:.6f} deg"
    )

    print(
        f"Maximum |swivel|       : "
        f"{max_abs_swivel * DEG:.6f} deg"
    )

    print(
        f"Minimum swivel         : "
        f"{min_swivel * DEG:.6f} deg"
    )

    print(
        f"Maximum swivel         : "
        f"{max_swivel * DEG:.6f} deg"
    )

    print(
        f"Average |swivel|       : "
        f"{average_abs_swivel * DEG:.6f} deg"
    )

    print()
    print("--- CASTER CONTACT ---")

    print(
        f"Initial caster contacts: "
        f"{initial_state['caster_wheel_contacts']}"
    )

    print(
        f"Final caster contacts  : "
        f"{final_state['caster_wheel_contacts']}"
    )

    print(
        f"Caster contact fraction: "
        f"{caster_contact_fraction * 100.0:.3f}%"
    )

    print(
        f"Average caster force   : "
        f"{average_caster_force:.6f} N"
    )

    print()
    print("--- HEIGHT ---")

    print(
        f"Initial Z              : "
        f"{initial_state['z']:.6f} m"
    )

    print(
        f"Final Z                : "
        f"{final_state['z']:.6f} m"
    )

    print(
        f"Minimum Z              : "
        f"{min_z:.6f} m"
    )

    print(
        f"Maximum Z              : "
        f"{max_z:.6f} m"
    )

    print(
        f"Z variation            : "
        f"{max_z - min_z:.6f} m"
    )

    print()
    print("--- CONTACT FORCES ---")

    print(
        f"Final left force       : "
        f"{final_state['left_normal_force']:.6f} N"
    )

    print(
        f"Final right force      : "
        f"{final_state['right_normal_force']:.6f} N"
    )

    print(
        f"Final caster force     : "
        f"{final_state['caster_normal_force']:.6f} N"
    )

    print(
        f"Average total force    : "
        f"{average_total_force:.6f} N"
    )

    print()
    print("--- FINAL POSITION ---")

    print(
        f"X                     : "
        f"{final_state['x']:.6f} m"
    )

    print(
        f"Y                     : "
        f"{final_state['y']:.6f} m"
    )

    print(
        f"Z                     : "
        f"{final_state['z']:.6f} m"
    )

    # --------------------------------------------------------
    # Diagnostic flags
    # --------------------------------------------------------

    print()
    print("--- DIAGNOSTIC FLAGS ---")

    if max_abs_pitch < math.radians(5.0):
        print("Pitch stability        : PASS")
    else:
        print("Pitch stability        : INVESTIGATE")

    if max_abs_roll < math.radians(5.0):
        print("Roll stability         : PASS")
    else:
        print("Roll stability         : INVESTIGATE")

    if caster_contact_fraction >= 0.50:
        print("Caster contact         : PRESENT")
    else:
        print("Caster contact         : LOW / LOST")

    if max_abs_swivel < math.radians(90.0):
        print("Caster swivel range    : NORMAL")
    else:
        print("Caster swivel range    : INVESTIGATE")

    if max_z - min_z < 0.01:
        print("Chassis height         : STABLE")
    else:
        print("Chassis height         : INVESTIGATE")

    # --------------------------------------------------------
    # Reverse diagnosis
    # --------------------------------------------------------

    if left_command < 0.0 and right_command < 0.0:

        print()
        print("--- REVERSE-MOTION DIAGNOSIS ---")

        if max_abs_pitch > math.radians(10.0):

            print(
                "Large reverse pitch detected."
            )

            if caster_contact_fraction < 0.50:

                print(
                    "Caster contact is low during reverse."
                )

                print(
                    "Primary suspect: "
                    "passive caster geometry/contact."
                )

            else:

                print(
                    "Caster remains in contact "
                    "despite large pitch."
                )

                print(
                    "Investigate wheel/chassis "
                    "contact mechanics."
                )

        else:

            print(
                "No abnormal reverse pitch detected."
            )

    return {
        "test_name": test_name,

        "max_pitch_deg": (
            max_abs_pitch * DEG
        ),

        "max_roll_deg": (
            max_abs_roll * DEG
        ),

        "max_caster_swivel_deg": (
            max_abs_swivel * DEG
        ),

        "caster_contact_fraction": (
            caster_contact_fraction
        ),

        "average_caster_force": (
            average_caster_force
        ),

        "z_variation": (
            max_z - min_z
        ),

        "final_pitch_deg": (
            final_state["pitch"] * DEG
        ),

        "final_caster_swivel_deg": (
            final_state[
                "caster_swivel_angle"
            ] * DEG
        ),
    }


# ============================================================
# FORWARD VS REVERSE COMPARISON
# ============================================================

def print_comparison(
    forward_result,
    reverse_result,
):

    print()
    print("=" * 100)
    print("FORWARD vs REVERSE COMPARISON")
    print("=" * 100)

    print()

    print(
        f"{'Metric':<35}"
        f"{'Forward +4/+4':>20}"
        f"{'Reverse -4/-4':>20}"
    )

    print("-" * 80)

    print(
        f"{'Maximum |pitch| (deg)':<35}"
        f"{forward_result['max_pitch_deg']:>20.6f}"
        f"{reverse_result['max_pitch_deg']:>20.6f}"
    )

    print(
        f"{'Maximum |roll| (deg)':<35}"
        f"{forward_result['max_roll_deg']:>20.6f}"
        f"{reverse_result['max_roll_deg']:>20.6f}"
    )

    print(
        f"{'Maximum |caster swivel| (deg)':<35}"
        f"{forward_result['max_caster_swivel_deg']:>20.6f}"
        f"{reverse_result['max_caster_swivel_deg']:>20.6f}"
    )

    print(
        f"{'Caster contact (%)':<35}"
        f"{forward_result['caster_contact_fraction'] * 100:>20.3f}"
        f"{reverse_result['caster_contact_fraction'] * 100:>20.3f}"
    )

    print(
        f"{'Average caster force (N)':<35}"
        f"{forward_result['average_caster_force']:>20.6f}"
        f"{reverse_result['average_caster_force']:>20.6f}"
    )

    print(
        f"{'Z variation (m)':<35}"
        f"{forward_result['z_variation']:>20.6f}"
        f"{reverse_result['z_variation']:>20.6f}"
    )

    print()
    print("--- INTERPRETATION ---")

    reverse_pitch = (
        reverse_result["max_pitch_deg"]
    )

    reverse_contact = (
        reverse_result[
            "caster_contact_fraction"
        ]
    )

    if (
        reverse_pitch > 10.0
        and reverse_contact < 0.50
    ):

        print(
            "RESULT: Large reverse pitch is strongly "
            "associated with caster contact loss."
        )

        print(
            "NEXT: Investigate caster geometry/contact."
        )

    elif reverse_pitch > 10.0:

        print(
            "RESULT: Large reverse pitch is confirmed, "
            "but caster contact remains significant."
        )

        print(
            "NEXT: Investigate wheel/chassis contact "
            "and caster joint mechanics."
        )

    else:

        print(
            "RESULT: Previous abnormal reverse pitch "
            "was not reproduced."
        )

        print(
            "NEXT: Recheck the V3.22.4 reverse result."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("V3.22.5 CASTER / REVERSE-MOTION DIAGNOSTIC")
    print("=" * 100)

    print()
    print("Purpose:")
    print(
        "Determine whether the large reverse-motion "
        "pitch is caused by caster behavior."
    )

    print()
    print("PPO                 : DISABLED")
    print("Reward              : NOT USED")
    print("RL environment      : NOT USED")
    print("Sensor noise        : NOT USED")
    print("Physics modification: NONE")

    print()
    print(
        f"Wheel radius        : "
        f"{WHEEL_RADIUS:.3f} m"
    )

    print(
        f"Wheel base          : "
        f"{WHEEL_BASE:.3f} m"
    )

    print(
        f"Reset base Z        : "
        f"{RESET_BASE_Z:.3f} m"
    )

    print(
        f"Simulation steps    : "
        f"{SIMULATION_STEPS}"
    )

    print()
    print("Joint hierarchy:")
    print(
        "  Joint 0 = LEFT WHEEL"
    )
    print(
        "  Joint 1 = RIGHT WHEEL"
    )
    print(
        "  Joint 2 = CASTER SWIVEL"
    )
    print(
        "  Joint 3 = CASTER WHEEL"
    )

    connect_physics()

    forward_result = None
    reverse_result = None

    try:

        # ====================================================
        # FORWARD BASELINE
        # ====================================================

        forward_result = run_motion_test(
            test_name="FORWARD +4/+4",
            left_command=FORWARD_LEFT,
            right_command=FORWARD_RIGHT,
        )

        # ====================================================
        # REVERSE DIAGNOSTIC
        # ====================================================

        reverse_result = run_motion_test(
            test_name="REVERSE -4/-4",
            left_command=REVERSE_LEFT,
            right_command=REVERSE_RIGHT,
        )

        # ====================================================
        # COMPARISON
        # ====================================================

        print_comparison(
            forward_result,
            reverse_result,
        )

    finally:

        if p.isConnected():
            p.disconnect()
            print()
            print("PyBullet disconnected.")

    print()
    print("=" * 100)
    print(
        "V3.22.5 CASTER / REVERSE DIAGNOSTIC COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()