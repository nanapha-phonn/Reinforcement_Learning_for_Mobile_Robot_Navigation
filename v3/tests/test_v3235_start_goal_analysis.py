"""
V3.23.5 — START/GOAL GENERALIZATION ANALYSIS

Purpose
-------
Analyze the frozen V3.23.1 PPO policy under different
start/goal configurations.

This is a DIAGNOSTIC test.

It does NOT:
    - train PPO
    - modify robot.py
    - modify environment.py
    - modify physics
    - modify sensors
    - modify the reward

The main question is:

    Does the 48-D observation correctly describe the
    robot's relationship to the goal, and does PPO
    produce the expected steering direction?

Important comparisons:

    Goal +Y:
        expected left action < right action

    Goal -Y:
        expected right action < left action

    Goal directly behind:
        policy should rotate rather than drive away

    Different initial yaw:
        policy should use goal bearing rather than
        assuming the robot always starts facing +X.

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3235_start_goal_analysis.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pybullet as p

from stable_baselines3 import PPO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from environment import DifferentialDriveEnv
from robot import CHASSIS_CENTER_Z, stop_robot


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v3231_ppo_baseline.zip"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MAX_WHEEL_VELOCITY = 4.0


# ============================================================
# ANALYSIS CASES
# ============================================================

CASES = [

    {
        "name": "A +X",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 0.0],
        "expected": "FORWARD",
    },

    {
        "name": "B +Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [0.0, 3.0],
        "expected": "TURN_LEFT",
    },

    {
        "name": "C -Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [0.0, -3.0],
        "expected": "TURN_RIGHT",
    },

    {
        "name": "D +X +Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 1.0],
        "expected": "TURN_LEFT",
    },

    {
        "name": "E +X -Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, -1.0],
        "expected": "TURN_RIGHT",
    },

    {
        "name": "F behind",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [-3.0, 0.0],
        "expected": "TURN_AROUND",
    },

    {
        "name": "G yaw +90",
        "start": [0.0, 0.0],
        "yaw_deg": 90.0,
        "goal": [3.0, 0.0],
        "expected": "TURN_RIGHT",
    },

    {
        "name": "H yaw 180",
        "start": [0.0, 0.0],
        "yaw_deg": 180.0,
        "goal": [3.0, 0.0],
        "expected": "TURN_AROUND",
    },

    {
        "name": "I yaw -90",
        "start": [0.0, 0.0],
        "yaw_deg": -90.0,
        "goal": [3.0, 0.0],
        "expected": "TURN_LEFT",
    },

    {
        "name": "J offset",
        "start": [1.0, 1.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 1.0],
        "expected": "FORWARD",
    },
]


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# ============================================================
# GET POSE
# ============================================================

def get_pose(env):

    position, orientation = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return (
        position,
        roll,
        pitch,
        yaw,
    )


# ============================================================
# COMPUTE GOAL RELATION
# ============================================================

def compute_goal_relation(
    env,
    position,
    yaw,
):

    dx = (
        env.goal_position[0]
        -
        position[0]
    )

    dy = (
        env.goal_position[1]
        -
        position[1]
    )

    distance = math.sqrt(
        dx * dx
        +
        dy * dy
    )

    goal_angle = math.atan2(
        dy,
        dx,
    )

    bearing = wrap_angle(
        goal_angle
        -
        yaw
    )

    return (
        distance,
        goal_angle,
        bearing,
    )


# ============================================================
# CONFIGURE CASE
# ============================================================

def configure_case(
    env,
    case,
    seed,
):

    # --------------------------------------------------------
    # Normal environment reset
    # --------------------------------------------------------

    env.reset(
        seed=seed
    )

    # --------------------------------------------------------
    # Desired yaw
    # --------------------------------------------------------

    yaw_rad = math.radians(
        case["yaw_deg"]
    )

    quaternion = p.getQuaternionFromEuler(
        [0.0, 0.0, yaw_rad]
    )

    # --------------------------------------------------------
    # Set robot pose
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [
            case["start"][0],
            case["start"][1],
            CHASSIS_CENTER_Z,
        ],
        quaternion,
    )

    stop_robot(
        env.robot_id
    )

    # --------------------------------------------------------
    # Set goal
    # --------------------------------------------------------

    env.goal_position = np.array(
        case["goal"],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Physics settling
    # --------------------------------------------------------

    for _ in range(120):

        p.stepSimulation()

    # --------------------------------------------------------
    # Update previous distance
    # --------------------------------------------------------

    position, _, _, yaw = (
        get_pose(env)
    )

    distance, _, _ = (
        compute_goal_relation(
            env,
            position,
            yaw,
        )
    )

    env.previous_distance = distance

    env.step_count = 0

    # --------------------------------------------------------
    # Build final observation
    # --------------------------------------------------------

    observation = env._get_observation()

    return observation


# ============================================================
# EXPECTED STEERING
# ============================================================

def expected_direction(
    bearing_deg,
):

    # Small bearing means forward.
    if abs(bearing_deg) < 10.0:

        return "FORWARD"

    # Positive bearing means goal is to the
    # robot's left in the current convention.
    if bearing_deg > 10.0:

        return "TURN_LEFT"

    # Negative bearing means goal is to
    # robot's right.
    if bearing_deg < -10.0:

        return "TURN_RIGHT"

    return "UNKNOWN"


# ============================================================
# ANALYZE POLICY ACTION
# ============================================================

def analyze_action(
    left,
    right,
    bearing_deg,
):

    difference = left - right

    magnitude = (
        abs(left)
        +
        abs(right)
    ) / 2.0

    # --------------------------------------------------------
    # Classify actual steering.
    # --------------------------------------------------------

    if abs(difference) < 0.05:

        actual = "STRAIGHT"

    elif difference < 0.0:

        actual = "TURN_LEFT"

    else:

        actual = "TURN_RIGHT"

    # --------------------------------------------------------
    # Determine whether action agrees with bearing.
    # --------------------------------------------------------

    expected = expected_direction(
        bearing_deg
    )

    if expected == "FORWARD":

        if abs(difference) < 0.05:

            consistency = "MATCH"

        else:

            consistency = "MISMATCH"

    elif expected == "TURN_LEFT":

        if difference < -0.05:

            consistency = "MATCH"

        else:

            consistency = "MISMATCH"

    elif expected == "TURN_RIGHT":

        if difference > 0.05:

            consistency = "MATCH"

        else:

            consistency = "MISMATCH"

    else:

        consistency = "UNKNOWN"

    return (
        actual,
        expected,
        consistency,
        difference,
        magnitude,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("V3.23.5 — START/GOAL GENERALIZATION ANALYSIS")
    print("=" * 90)

    print()
    print("Model:")
    print(
        f"  {MODEL_PATH}"
    )

    print()
    print("Training:")
    print("  NONE")

    print()
    print("Environment:")
    print("  FROZEN")

    print()
    print(
        "Purpose:"
    )

    print(
        "  Determine whether PPO steering agrees with"
    )

    print(
        "  the goal-relative bearing."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = PPO.load(
        str(MODEL_PATH),
        device="auto",
    )

    print()
    print(
        "[PASS] PPO model loaded."
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    results = []

    try:

        for index, case in enumerate(
            CASES,
            start=1,
        ):

            print()
            print("=" * 90)

            print(
                f"CASE {index:02d}/{len(CASES):02d} — "
                f"{case['name']}"
            )

            print("=" * 90)

            print(
                f"Start = "
                f"({case['start'][0]:.2f}, "
                f"{case['start'][1]:.2f})"
            )

            print(
                f"Start yaw = "
                f"{case['yaw_deg']:.1f}°"
            )

            print(
                f"Goal = "
                f"({case['goal'][0]:.2f}, "
                f"{case['goal'][1]:.2f})"
            )

            print(
                f"Expected behavior = "
                f"{case['expected']}"
            )

            # ------------------------------------------------
            # Configure
            # ------------------------------------------------

            observation = configure_case(
                env,
                case,
                SEED + index,
            )

            # ------------------------------------------------
            # Initial pose
            # ------------------------------------------------

            position, roll, pitch, yaw = (
                get_pose(env)
            )

            distance, goal_angle, bearing = (
                compute_goal_relation(
                    env,
                    position,
                    yaw,
                )
            )

            bearing_deg = math.degrees(
                bearing
            )

            goal_angle_deg = math.degrees(
                goal_angle
            )

            yaw_deg = math.degrees(
                yaw
            )

            # ------------------------------------------------
            # Observation extraction
            # ------------------------------------------------

            encoder = observation[0:4]

            imu = observation[4:10]

            lidar = observation[10:46]

            goal_obs = observation[46:48]

            # ------------------------------------------------
            # PPO action
            # ------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float64,
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

            left = float(
                action[0]
            )

            right = float(
                action[1]
            )

            (
                actual,
                expected,
                consistency,
                difference,
                magnitude,
            ) = analyze_action(
                left,
                right,
                bearing_deg,
            )

            # ------------------------------------------------
            # Print physical information
            # ------------------------------------------------

            print()
            print(
                "PHYSICAL STATE"
            )

            print(
                f"Position = "
                f"({position[0]:.6f}, "
                f"{position[1]:.6f})"
            )

            print(
                f"Robot yaw = "
                f"{yaw_deg:.6f}°"
            )

            print(
                f"Goal angle = "
                f"{goal_angle_deg:.6f}°"
            )

            print(
                f"Goal bearing = "
                f"{bearing_deg:.6f}°"
            )

            print(
                f"Goal distance = "
                f"{distance:.6f} m"
            )

            # ------------------------------------------------
            # Print goal observation
            # ------------------------------------------------

            print()
            print(
                "GOAL OBSERVATION"
            )

            print(
                f"Normalized distance = "
                f"{goal_obs[0]:.6f}"
            )

            print(
                f"Normalized bearing = "
                f"{goal_obs[1]:.6f}"
            )

            # ------------------------------------------------
            # Print PPO action
            # ------------------------------------------------

            print()
            print(
                "PPO ACTION"
            )

            print(
                f"Left action = "
                f"{left:.6f}"
            )

            print(
                f"Right action = "
                f"{right:.6f}"
            )

            print(
                f"Left velocity = "
                f"{left * MAX_WHEEL_VELOCITY:.6f} rad/s"
            )

            print(
                f"Right velocity = "
                f"{right * MAX_WHEEL_VELOCITY:.6f} rad/s"
            )

            print(
                f"Left - Right = "
                f"{difference:.6f}"
            )

            print(
                f"Action magnitude = "
                f"{magnitude:.6f}"
            )

            # ------------------------------------------------
            # Print interpretation
            # ------------------------------------------------

            print()
            print(
                "STEERING ANALYSIS"
            )

            print(
                f"Expected from bearing = "
                f"{expected}"
            )

            print(
                f"Actual action = "
                f"{actual}"
            )

            print(
                f"Consistency = "
                f"{consistency}"
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            results.append(
                {
                    "name": case["name"],
                    "bearing_deg": bearing_deg,
                    "distance": distance,
                    "goal_obs_distance": float(
                        goal_obs[0]
                    ),
                    "goal_obs_bearing": float(
                        goal_obs[1]
                    ),
                    "left": left,
                    "right": right,
                    "difference": difference,
                    "expected": expected,
                    "actual": actual,
                    "consistency": consistency,
                }
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        matches = sum(
            result["consistency"] == "MATCH"
            for result in results
        )

        total = len(results)

        percentage = (
            matches
            /
            total
            *
            100.0
        )

        print()
        print("=" * 90)
        print("V3.23.5 START/GOAL ANALYSIS SUMMARY")
        print("=" * 90)

        print()

        print(
            f"Cases analyzed = "
            f"{total}"
        )

        print(
            f"Steering matches = "
            f"{matches}/{total}"
        )

        print(
            f"Steering consistency = "
            f"{percentage:.1f}%"
        )

        print()

        print(
            "CASE | Bearing | "
            "GoalObs | "
            "Left | Right | "
            "Expected | Actual | Match"
        )

        print(
            "-" * 90
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index:4d} | "
                f"{result['bearing_deg']:7.2f}° | "
                f"{result['goal_obs_bearing']:7.3f} | "
                f"{result['left']:5.3f} | "
                f"{result['right']:5.3f} | "
                f"{result['expected']:9s} | "
                f"{result['actual']:9s} | "
                f"{result['consistency']}"
            )

        print()

        # ----------------------------------------------------
        # Goal observation sanity check
        # ----------------------------------------------------

        goal_observation_correct = True

        for result in results:

            expected_normalized = (
                result["bearing_deg"]
                /
                180.0
            )

            actual_normalized = (
                result["goal_obs_bearing"]
            )

            if not np.isclose(
                expected_normalized,
                actual_normalized,
                atol=0.02,
            ):

                goal_observation_correct = False

        if goal_observation_correct:

            print(
                "[PASS] Goal bearing observation "
                "matches physical bearing."
            )

        else:

            print(
                "[FAIL] Goal bearing observation "
                "does not match physical bearing."
            )

        # ----------------------------------------------------
        # Final interpretation
        # ----------------------------------------------------

        print()

        if percentage >= 80.0:

            print(
                "[PASS] PPO generally responds "
                "according to goal bearing."
            )

        elif percentage >= 50.0:

            print(
                "[PARTIAL] PPO responds correctly "
                "in some directions but has a "
                "significant steering bias."
            )

        else:

            print(
                "[FAIL] PPO steering is not "
                "consistent with goal bearing."
            )

        print()
        print("=" * 90)
        print("V3.23.5 COMPLETE")
        print("=" * 90)

        print()
        print(
            "No files were modified."
        )

        print(
            "No PPO training was performed."
        )

        print()
        print(
            "Next decision:"
        )

        print(
            "  Use this analysis to determine whether"
        )

        print(
            "  the next PPO baseline should randomize"
        )

        print(
            "  start positions, start yaw, and goals."
        )

    finally:

        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()