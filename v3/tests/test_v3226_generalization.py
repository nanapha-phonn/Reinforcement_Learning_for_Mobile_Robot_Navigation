"""
V3.22.6 — GENERALIZATION VALIDATION

Purpose
-------
Validate the frozen V3 Gymnasium environment using:

1. Different initial yaw
2. Different goal positions
3. Different initial positions
4. Combined initial position + yaw + goal

This test does NOT modify environment.py.

The environment is:
    environment.py
    DifferentialDriveEnv

Run:
    cd C:\python_projects\rl_robot\v3
    python tests\test_v3226_generalization.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pybullet as p


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT EXISTING ENVIRONMENT
# ============================================================

from environment import DifferentialDriveEnv


# ============================================================
# CONSTANTS
# ============================================================

CHASSIS_Z = 0.13

SETTLE_STEPS = 120

CHECK_STEPS = 240

POSITION_TOLERANCE = 0.01

YAW_TOLERANCE = math.radians(1.0)

MAX_ROLL = 5.0
MAX_PITCH = 5.0


# ============================================================
# TEST CASES
# ============================================================

CASES = [

    # --------------------------------------------------------
    # Initial yaw
    # --------------------------------------------------------

    {
        "name": "Yaw 0",
        "position": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (3.0, 0.0),
    },

    {
        "name": "Yaw +45",
        "position": (0.0, 0.0),
        "yaw": math.radians(45.0),
        "goal": (3.0, 0.0),
    },

    {
        "name": "Yaw -45",
        "position": (0.0, 0.0),
        "yaw": math.radians(-45.0),
        "goal": (3.0, 0.0),
    },

    {
        "name": "Yaw 180",
        "position": (0.0, 0.0),
        "yaw": math.pi,
        "goal": (3.0, 0.0),
    },

    # --------------------------------------------------------
    # Goal variation
    # --------------------------------------------------------

    {
        "name": "Goal +X",
        "position": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (3.0, 0.0),
    },

    {
        "name": "Goal +X +Y",
        "position": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (2.0, 2.0),
    },

    {
        "name": "Goal +X -Y",
        "position": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (2.0, -2.0),
    },

    {
        "name": "Goal +Y",
        "position": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (0.0, 2.0),
    },

    # --------------------------------------------------------
    # Combined variation
    # --------------------------------------------------------

    {
        "name": "Combined 1",
        "position": (-1.0, 1.0),
        "yaw": math.radians(45.0),
        "goal": (2.0, -1.0),
    },

    {
        "name": "Combined 2",
        "position": (-1.0, -1.0),
        "yaw": math.radians(-45.0),
        "goal": (2.0, 1.0),
    },

    {
        "name": "Combined 3",
        "position": (1.0, 1.0),
        "yaw": math.radians(135.0),
        "goal": (-1.0, -1.0),
    },

    {
        "name": "Combined 4",
        "position": (1.0, -1.0),
        "yaw": math.radians(-135.0),
        "goal": (-1.0, 1.0),
    },
]


# ============================================================
# ANGLE HELPER
# ============================================================

def angle_difference(a, b):

    return math.atan2(
        math.sin(a - b),
        math.cos(a - b),
    )


# ============================================================
# READ ROBOT POSE
# ============================================================

def get_robot_pose(env):

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
        position[0],
        position[1],
        position[2],
        roll,
        pitch,
        yaw,
    )


# ============================================================
# CHECK ONE CASE
# ============================================================

def run_case(env, case):

    name = case["name"]

    x_target, y_target = case["position"]

    yaw_target = case["yaw"]

    goal_x, goal_y = case["goal"]

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Start position : "
        f"({x_target:.2f}, {y_target:.2f})"
    )

    print(
        f"Start yaw      : "
        f"{math.degrees(yaw_target):.1f}°"
    )

    print(
        f"Goal           : "
        f"({goal_x:.2f}, {goal_y:.2f})"
    )

    # --------------------------------------------------------
    # NORMAL RESET
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=42
    )

    # --------------------------------------------------------
    # MANUALLY SET INITIAL POSE
    #
    # This is ONLY for validation.
    #
    # environment.py itself remains unchanged.
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [
            x_target,
            y_target,
            CHASSIS_Z,
        ],
        p.getQuaternionFromEuler(
            [
                0.0,
                0.0,
                yaw_target,
            ]
        ),
    )

    # Reset base velocity.

    p.resetBaseVelocity(
        env.robot_id,
        linearVelocity=[
            0.0,
            0.0,
            0.0,
        ],
        angularVelocity=[
            0.0,
            0.0,
            0.0,
        ],
    )

    # --------------------------------------------------------
    # SET GOAL
    # --------------------------------------------------------

    env.goal_position = np.array(
        [
            goal_x,
            goal_y,
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # UPDATE PREVIOUS DISTANCE
    # --------------------------------------------------------

    position, _ = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    env.previous_distance = (
        env._distance_to_goal(
            position
        )
    )

    # --------------------------------------------------------
    # SETTLE
    # --------------------------------------------------------

    for _ in range(
        SETTLE_STEPS
    ):

        p.stepSimulation()

    # --------------------------------------------------------
    # READ INITIAL STATE
    # --------------------------------------------------------

    (
        x,
        y,
        z,
        roll,
        pitch,
        yaw,
    ) = get_robot_pose(
        env
    )

    position_error = math.sqrt(
        (x - x_target) ** 2 +
        (y - y_target) ** 2
    )

    yaw_error = abs(
        angle_difference(
            yaw,
            yaw_target
        )
    )

    print()
    print("Initial state:")
    print(
        f"  Position = "
        f"({x:.5f}, {y:.5f}, {z:.5f})"
    )

    print(
        f"  Roll     = "
        f"{math.degrees(roll):.4f}°"
    )

    print(
        f"  Pitch    = "
        f"{math.degrees(pitch):.4f}°"
    )

    print(
        f"  Yaw      = "
        f"{math.degrees(yaw):.4f}°"
    )

    print(
        f"  Position error = "
        f"{position_error:.6f} m"
    )

    print(
        f"  Yaw error      = "
        f"{math.degrees(yaw_error):.6f}°"
    )

    # --------------------------------------------------------
    # INITIAL VALIDATION
    # --------------------------------------------------------

    if position_error > POSITION_TOLERANCE:

        print(
            "[FAIL] Initial position incorrect."
        )

        return False

    if yaw_error > YAW_TOLERANCE:

        print(
            "[FAIL] Initial yaw incorrect."
        )

        return False

    if (
        abs(math.degrees(roll))
        > MAX_ROLL
    ):

        print(
            "[FAIL] Excessive initial roll."
        )

        return False

    if (
        abs(math.degrees(pitch))
        > MAX_PITCH
    ):

        print(
            "[FAIL] Excessive initial pitch."
        )

        return False

    # --------------------------------------------------------
    # OBSERVATION CHECK
    # --------------------------------------------------------

    observation = env._get_observation()

    if not np.all(
        np.isfinite(observation)
    ):

        print(
            "[FAIL] Observation contains NaN/Inf."
        )

        return False

    if observation.shape != (48,):

        print(
            f"[FAIL] Observation shape = "
            f"{observation.shape}"
        )

        return False

    if np.any(
        observation < -1.00001
    ) or np.any(
        observation > 1.00001
    ):

        print(
            "[FAIL] Observation outside [-1,1]."
        )

        return False

    print(
        "[PASS] Observation valid."
    )

    # --------------------------------------------------------
    # ZERO ACTION STABILITY
    # --------------------------------------------------------

    max_roll_seen = abs(
        math.degrees(roll)
    )

    max_pitch_seen = abs(
        math.degrees(pitch)
    )

    for _ in range(
        CHECK_STEPS
    ):

        observation, reward, terminated, truncated, info = (
            env.step(
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            )
        )

        if not np.all(
            np.isfinite(observation)
        ):

            print(
                "[FAIL] Observation became invalid."
            )

            return False

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_robot_pose(
            env
        )

        max_roll_seen = max(
            max_roll_seen,
            abs(
                math.degrees(roll)
            )
        )

        max_pitch_seen = max(
            max_pitch_seen,
            abs(
                math.degrees(pitch)
            )
        )

        if terminated:

            print(
                "[FAIL] Episode terminated "
                "during zero-action stability test."
            )

            return False

    print(
        f"Max |roll|  = "
        f"{max_roll_seen:.6f}°"
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch_seen:.6f}°"
    )

    if max_roll_seen > MAX_ROLL:

        print(
            "[FAIL] Excessive roll."
        )

        return False

    if max_pitch_seen > MAX_PITCH:

        print(
            "[FAIL] Excessive pitch."
        )

        return False

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "[PASS] CASE COMPLETE"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "V3.22.6 — GENERALIZATION VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Environment: environment.py"
    )

    print(
        "Class: DifferentialDriveEnv"
    )

    print(
        "PPO training: DISABLED"
    )

    print(
        "Production files modified: NONE"
    )

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    results = []

    try:

        for case in CASES:

            try:

                passed = run_case(
                    env,
                    case
                )

            except Exception as exc:

                print()
                print(
                    "[FAIL] Unexpected error:"
                )

                print(
                    repr(exc)
                )

                passed = False

            results.append(
                (
                    case["name"],
                    passed
                )
            )

    finally:

        env.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "V3.22.6 SUMMARY"
    )
    print("=" * 80)

    passed_count = 0
    failed_count = 0

    print()

    for name, passed in results:

        if passed:

            print(
                f"[PASS] {name}"
            )

            passed_count += 1

        else:

            print(
                f"[FAIL] {name}"
            )

            failed_count += 1

    print()
    print(
        f"Passed: {passed_count}/{len(results)}"
    )

    print(
        f"Failed: {failed_count}/{len(results)}"
    )

    print()

    if failed_count == 0:

        print(
            "[PASS] V3.22.6 GENERALIZATION COMPLETE"
        )

        print()
        print(
            "The frozen environment supports:"
        )

        print(
            "  - different initial yaw"
        )

        print(
            "  - different initial positions"
        )

        print(
            "  - different goal positions"
        )

        print(
            "  - combined variation"
        )

        print()
        print(
            "Ready for V3.22.7 robustness."
        )

    else:

        print(
            "[FAIL] V3.22.6 requires investigation."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":

    main()