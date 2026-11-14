"""
V3.23.4 — PPO GENERALIZATION EVALUATION

Purpose
-------
Test whether the frozen V3.23.1 PPO policy can reach goals
under different start positions, start orientations, and
goal locations.

IMPORTANT:
    - No PPO training
    - No environment.py modification
    - No robot.py modification
    - No reward modification
    - No physics modification

The environment is temporarily configured AFTER reset for
each evaluation case.

Tests:
    1. Straight +X
    2. +X +Y
    3. +X -Y
    4. +Y
    5. -Y
    6. Diagonal +X +Y
    7. Diagonal +X -Y
    8. Offset start
    9. Different start yaw
    10. Different start + goal

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3234_generalization.py
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
# ENVIRONMENT
# ============================================================

from environment import DifferentialDriveEnv
from robot import CHASSIS_CENTER_Z, stop_robot


# ============================================================
# PATHS
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

MAX_EPISODE_STEPS = 1000

GOAL_THRESHOLD = 0.20

MAX_WHEEL_VELOCITY = 4.0


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "name": "01 +X",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 0.0],
    },

    {
        "name": "02 +X +Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 1.0],
    },

    {
        "name": "03 +X -Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [3.0, -1.0],
    },

    {
        "name": "04 +Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [0.0, 3.0],
    },

    {
        "name": "05 -Y",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [0.0, -3.0],
    },

    {
        "name": "06 diagonal + +",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [2.0, 2.0],
    },

    {
        "name": "07 diagonal + -",
        "start": [0.0, 0.0],
        "yaw_deg": 0.0,
        "goal": [2.0, -2.0],
    },

    {
        "name": "08 offset start",
        "start": [1.0, 1.0],
        "yaw_deg": 0.0,
        "goal": [3.0, 1.0],
    },

    {
        "name": "09 different yaw",
        "start": [0.0, 0.0],
        "yaw_deg": 90.0,
        "goal": [3.0, 0.0],
    },

    {
        "name": "10 different start + goal",
        "start": [-1.0, 1.0],
        "yaw_deg": 180.0,
        "goal": [2.0, -1.0],
    },
]


# ============================================================
# ANGLE HELPERS
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# ============================================================
# SET CUSTOM CASE
# ============================================================

def configure_case(
    env,
    start,
    yaw_deg,
    goal,
    seed,
):

    # --------------------------------------------------------
    # Reset normally first.
    # --------------------------------------------------------

    observation, info = env.reset(
        seed=seed
    )

    # --------------------------------------------------------
    # Convert yaw to quaternion.
    # --------------------------------------------------------

    yaw_rad = math.radians(
        yaw_deg
    )

    quaternion = p.getQuaternionFromEuler(
        [0.0, 0.0, yaw_rad]
    )

    # --------------------------------------------------------
    # Set robot pose.
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [
            float(start[0]),
            float(start[1]),
            CHASSIS_CENTER_Z,
        ],
        quaternion,
    )

    # --------------------------------------------------------
    # Stop wheel motors.
    # --------------------------------------------------------

    stop_robot(
        env.robot_id
    )

    # --------------------------------------------------------
    # Set goal.
    # --------------------------------------------------------

    env.goal_position = np.array(
        [
            float(goal[0]),
            float(goal[1]),
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Allow physics to settle.
    # --------------------------------------------------------

    for _ in range(120):

        p.stepSimulation()

    # --------------------------------------------------------
    # Update previous distance.
    # --------------------------------------------------------

    position, _ = p.getBasePositionAndOrientation(
        env.robot_id
    )

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

    env.previous_distance = math.sqrt(
        dx * dx
        +
        dy * dy
    )

    env.step_count = 0

    # --------------------------------------------------------
    # Generate observation from the configured state.
    # --------------------------------------------------------

    observation = env._get_observation()

    return observation


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
# GOAL METRICS
# ============================================================

def get_goal_metrics(
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
        math.degrees(bearing),
    )


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    model,
    env,
    case_index,
    case,
):

    observation = configure_case(
        env=env,
        start=case["start"],
        yaw_deg=case["yaw_deg"],
        goal=case["goal"],
        seed=SEED + case_index,
    )

    initial_position, _, _, initial_yaw = (
        get_pose(env)
    )

    initial_distance, initial_bearing = (
        get_goal_metrics(
            env,
            initial_position,
            initial_yaw,
        )
    )

    total_reward = 0.0

    path_length = 0.0

    previous_xy = np.array(
        [
            initial_position[0],
            initial_position[1],
        ],
        dtype=np.float64,
    )

    terminated = False
    truncated = False

    step_count = 0

    actions = []

    while (
        not terminated
        and not truncated
        and step_count < MAX_EPISODE_STEPS
    ):

        # ----------------------------------------------------
        # PPO prediction
        # ----------------------------------------------------

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

        actions.append(
            action.copy()
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        total_reward += float(
            reward
        )

        # ----------------------------------------------------
        # Position after step
        # ----------------------------------------------------

        position, roll, pitch, yaw = (
            get_pose(env)
        )

        current_xy = np.array(
            [
                position[0],
                position[1],
            ],
            dtype=np.float64,
        )

        segment_distance = np.linalg.norm(
            current_xy
            -
            previous_xy
        )

        path_length += float(
            segment_distance
        )

        previous_xy = current_xy

        step_count += 1

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_position, final_roll, final_pitch, final_yaw = (
        get_pose(env)
    )

    final_distance, final_bearing = (
        get_goal_metrics(
            env,
            final_position,
            final_yaw,
        )
    )

    net_displacement = math.sqrt(
        (
            final_position[0]
            -
            initial_position[0]
        ) ** 2
        +
        (
            final_position[1]
            -
            initial_position[1]
        ) ** 2
    )

    if path_length > 1e-12:

        path_efficiency = (
            min(
                net_displacement
                /
                path_length,
                1.0,
            )
        )

    else:

        path_efficiency = 0.0

    # --------------------------------------------------------
    # Action statistics
    # --------------------------------------------------------

    action_array = np.asarray(
        actions,
        dtype=np.float64,
    )

    if len(action_array) > 0:

        mean_left = float(
            np.mean(
                action_array[:, 0]
            )
        )

        mean_right = float(
            np.mean(
                action_array[:, 1]
            )
        )

        mean_difference = float(
            np.mean(
                np.abs(
                    action_array[:, 0]
                    -
                    action_array[:, 1]
                )
            )
        )

    else:

        mean_left = 0.0
        mean_right = 0.0
        mean_difference = 0.0

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    success = (
        final_distance
        <
        GOAL_THRESHOLD
    )

    collision = bool(
        info.get(
            "collision",
            False,
        )
    )

    return {
        "name": case["name"],
        "start_x": case["start"][0],
        "start_y": case["start"][1],
        "start_yaw": case["yaw_deg"],
        "goal_x": case["goal"][0],
        "goal_y": case["goal"][1],
        "steps": step_count,
        "reward": total_reward,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "final_bearing": final_bearing,
        "final_roll": math.degrees(
            final_roll
        ),
        "final_pitch": math.degrees(
            final_pitch
        ),
        "final_yaw": math.degrees(
            final_yaw
        ),
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "mean_difference": mean_difference,
        "success": success,
        "collision": collision,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("V3.23.4 — PPO GENERALIZATION EVALUATION")
    print("=" * 80)

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
        f"Number of test cases: "
        f"{len(TEST_CASES)}"
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
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    results = []

    try:

        # ----------------------------------------------------
        # Run all cases
        # ----------------------------------------------------

        for index, case in enumerate(
            TEST_CASES,
            start=1,
        ):

            print()
            print("=" * 80)

            print(
                f"TEST {index:02d}/"
                f"{len(TEST_CASES):02d} — "
                f"{case['name']}"
            )

            print("=" * 80)

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

            result = run_case(
                model=model,
                env=env,
                case_index=index,
                case=case,
            )

            results.append(
                result
            )

            status = (
                "PASS"
                if result["success"]
                else "FAIL"
            )

            print()
            print(
                f"[{status}] "
                f"Steps = "
                f"{result['steps']}"
            )

            print(
                f"Reward = "
                f"{result['reward']:.6f}"
            )

            print(
                f"Final distance = "
                f"{result['final_distance']:.6f} m"
            )

            print(
                f"Final bearing = "
                f"{result['final_bearing']:.3f}°"
            )

            print(
                f"Final yaw = "
                f"{result['final_yaw']:.3f}°"
            )

            print(
                f"Path efficiency = "
                f"{result['path_efficiency'] * 100:.3f}%"
            )

            print(
                f"Mean left action = "
                f"{result['mean_left']:.3f}"
            )

            print(
                f"Mean right action = "
                f"{result['mean_right']:.3f}"
            )

            print(
                f"Mean |L-R| = "
                f"{result['mean_difference']:.3f}"
            )

            print(
                f"Collision = "
                f"{result['collision']}"
            )

        # ----------------------------------------------------
        # Overall summary
        # ----------------------------------------------------

        successes = sum(
            result["success"]
            for result in results
        )

        collisions = sum(
            result["collision"]
            for result in results
        )

        success_rate = (
            successes
            /
            len(results)
            *
            100.0
        )

        average_final_distance = float(
            np.mean(
                [
                    result["final_distance"]
                    for result in results
                ]
            )
        )

        average_steps = float(
            np.mean(
                [
                    result["steps"]
                    for result in results
                ]
            )
        )

        average_bearing = float(
            np.mean(
                [
                    abs(
                        result["final_bearing"]
                    )
                    for result in results
                ]
            )
        )

        average_efficiency = float(
            np.mean(
                [
                    result["path_efficiency"]
                    for result in results
                ]
            )
        )

        print()
        print("=" * 80)
        print("V3.23.4 GENERALIZATION SUMMARY")
        print("=" * 80)

        print()

        print(
            f"Cases tested          = "
            f"{len(results)}"
        )

        print(
            f"Successful cases      = "
            f"{successes}/{len(results)}"
        )

        print(
            f"Success rate          = "
            f"{success_rate:.1f}%"
        )

        print(
            f"Collisions            = "
            f"{collisions}"
        )

        print(
            f"Average final distance = "
            f"{average_final_distance:.6f} m"
        )

        print(
            f"Average episode steps  = "
            f"{average_steps:.2f}"
        )

        print(
            f"Average |bearing|      = "
            f"{average_bearing:.3f}°"
        )

        print(
            f"Average path efficiency = "
            f"{average_efficiency * 100:.3f}%"
        )

        print()

        # ----------------------------------------------------
        # Case table
        # ----------------------------------------------------

        print(
            "CASE RESULTS"
        )

        print()

        print(
            "Case | Success | Steps | "
            "Final Dist | Bearing | Collision"
        )

        print(
            "-" * 75
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index:4d} | "
                f"{str(result['success']):7s} | "
                f"{result['steps']:5d} | "
                f"{result['final_distance']:10.4f} | "
                f"{result['final_bearing']:7.2f}° | "
                f"{str(result['collision'])}"
            )

        print()

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if successes == len(results):

            print(
                "[PASS] PPO generalized to all "
                "tested start/goal configurations."
            )

        elif successes > 0:

            print(
                "[PARTIAL] PPO generalized to "
                f"{successes}/{len(results)} cases."
            )

        else:

            print(
                "[FAIL] PPO did not generalize "
                "to the tested configurations."
            )

        print()

        if collisions == 0:

            print(
                "[PASS] No collisions detected."
            )

        else:

            print(
                "[WARNING] Collisions detected."
            )

        print()
        print("=" * 80)
        print("V3.23.4 COMPLETE")
        print("=" * 80)

        print()
        print(
            "Next:"
        )

        print(
            "  V3.23.5 — Start/goal generalization analysis"
        )

    finally:

        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()