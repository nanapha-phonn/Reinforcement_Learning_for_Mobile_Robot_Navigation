"""
V3.23.3 — SINGLE-EPISODE PPO EVALUATION

Purpose
-------
Inspect one complete trajectory produced by the trained
V3.23.1 PPO baseline.

This test does NOT:
    - train PPO
    - modify robot.py
    - modify environment.py
    - modify the reward
    - modify physics

It records:

    Step
    X position
    Y position
    Yaw
    Distance to goal
    Goal bearing
    Left action
    Right action
    Left wheel velocity
    Right wheel velocity
    Reward

The purpose is to understand what the trained controller
actually does during one successful episode.

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3233_single_episode.py
"""

import sys
import math
from pathlib import Path

import numpy as np

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


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v3231_ppo_baseline.zip"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "v3233_single_episode.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MAX_EPISODE_STEPS = 1000

PRINT_EVERY = 10

MAX_WHEEL_VELOCITY = 4.0

GOAL_THRESHOLD = 0.20


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment():

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    return env


# ============================================================
# ROBOT POSE
# ============================================================

def get_robot_pose(env):

    import pybullet as p

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

    goal_x = env.goal_position[0]

    goal_y = env.goal_position[1]

    dx = goal_x - position[0]

    dy = goal_y - position[1]

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
        goal_angle - yaw
    )

    return (
        distance,
        math.degrees(bearing),
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    import csv

    if not records:

        return

    fieldnames = list(
        records[0].keys()
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    print()
    print(
        f"Trajectory saved to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 80)
    print("V3.23.3 — SINGLE-EPISODE PPO EVALUATION")
    print("=" * 80)

    print()
    print(
        "Model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print()
    print(
        "Environment: FROZEN"
    )

    print(
        "Training: NONE"
    )

    print(
        "Seed:",
        SEED,
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

    env = create_environment()

    records = []

    total_reward = 0.0

    try:

        observation, info = env.reset(
            seed=SEED
        )

        initial_position, _, _, initial_yaw = (
            get_robot_pose(env)
        )

        initial_distance, initial_bearing = (
            get_goal_metrics(
                env,
                initial_position,
                initial_yaw,
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            "INITIAL STATE"
        )

        print(
            "=" * 80
        )

        print(
            f"Position:"
        )

        print(
            f"  X = {initial_position[0]:.6f} m"
        )

        print(
            f"  Y = {initial_position[1]:.6f} m"
        )

        print(
            f"Yaw = "
            f"{math.degrees(initial_yaw):.6f}°"
        )

        print(
            f"Distance = "
            f"{initial_distance:.6f} m"
        )

        print(
            f"Bearing = "
            f"{initial_bearing:.6f}°"
        )

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------

        terminated = False
        truncated = False

        step = 0

        print()
        print(
            "=" * 80
        )

        print(
            "TRAJECTORY"
        )

        print(
            "=" * 80
        )

        print()
        print(
            "step | "
            "x      | "
            "y      | "
            "yaw    | "
            "dist   | "
            "bearing | "
            "left   | "
            "right  | "
            "reward"
        )

        print(
            "-" * 95
        )

        while (
            not terminated
            and not truncated
            and step < MAX_EPISODE_STEPS
        ):

            # -----------------------------------------------
            # Current state BEFORE action
            # -----------------------------------------------

            position, roll, pitch, yaw = (
                get_robot_pose(env)
            )

            distance, bearing = (
                get_goal_metrics(
                    env,
                    position,
                    yaw,
                )
            )

            # -----------------------------------------------
            # PPO action
            # -----------------------------------------------

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

            left_action = float(
                action[0]
            )

            right_action = float(
                action[1]
            )

            left_velocity = (
                left_action
                *
                MAX_WHEEL_VELOCITY
            )

            right_velocity = (
                right_action
                *
                MAX_WHEEL_VELOCITY
            )

            # -----------------------------------------------
            # Environment step
            # -----------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                step_info,
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            # -----------------------------------------------
            # Record
            # -----------------------------------------------

            record = {
                "step": step + 1,
                "x_m": position[0],
                "y_m": position[1],
                "yaw_deg": math.degrees(yaw),
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "distance_m": distance,
                "bearing_deg": bearing,
                "left_action": left_action,
                "right_action": right_action,
                "left_velocity_rad_s": left_velocity,
                "right_velocity_rad_s": right_velocity,
                "reward": float(reward),
            }

            records.append(
                record
            )

            # -----------------------------------------------
            # Console output
            # -----------------------------------------------

            if (
                step % PRINT_EVERY == 0
                or terminated
                or truncated
            ):

                print(
                    f"{step + 1:4d} | "
                    f"{position[0]: .4f} | "
                    f"{position[1]: .4f} | "
                    f"{math.degrees(yaw): .3f} | "
                    f"{distance: .4f} | "
                    f"{bearing: .3f} | "
                    f"{left_action: .3f} | "
                    f"{right_action: .3f} | "
                    f"{float(reward): .4f}"
                )

            observation = next_observation

            step += 1

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        final_position, final_roll, final_pitch, final_yaw = (
            get_robot_pose(env)
        )

        final_distance, final_bearing = (
            get_goal_metrics(
                env,
                final_position,
                final_yaw,
            )
        )

        # ----------------------------------------------------
        # Path metrics
        # ----------------------------------------------------

        if len(records) > 1:

            points = np.array(
                [
                    [
                        record["x_m"],
                        record["y_m"],
                    ]
                    for record in records
                ],
                dtype=np.float64,
            )

            segment_lengths = np.linalg.norm(
                np.diff(
                    points,
                    axis=0,
                ),
                axis=1,
            )

            path_length = float(
                np.sum(
                    segment_lengths
                )
            )

        else:

            path_length = 0.0

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
                net_displacement
                /
                path_length
            )

        else:

            path_efficiency = 0.0

        # ----------------------------------------------------
        # Action statistics
        # ----------------------------------------------------

        left_actions = np.array(
            [
                record["left_action"]
                for record in records
            ],
            dtype=np.float64,
        )

        right_actions = np.array(
            [
                record["right_action"]
                for record in records
            ],
            dtype=np.float64,
        )

        if len(records) > 0:

            mean_left = float(
                np.mean(left_actions)
            )

            mean_right = float(
                np.mean(right_actions)
            )

            std_left = float(
                np.std(left_actions)
            )

            std_right = float(
                np.std(right_actions)
            )

            mean_difference = float(
                np.mean(
                    np.abs(
                        left_actions
                        -
                        right_actions
                    )
                )
            )

        else:

            mean_left = 0.0
            mean_right = 0.0
            std_left = 0.0
            std_right = 0.0
            mean_difference = 0.0

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_csv(
            records
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("V3.23.3 SINGLE-EPISODE SUMMARY")
        print("=" * 80)

        print()

        print(
            f"Steps              = "
            f"{step}"
        )

        print(
            f"Total reward       = "
            f"{total_reward:.6f}"
        )

        print(
            f"Initial distance   = "
            f"{initial_distance:.6f} m"
        )

        print(
            f"Final distance     = "
            f"{final_distance:.6f} m"
        )

        print(
            f"Final bearing      = "
            f"{final_bearing:.6f}°"
        )

        print(
            f"Net displacement   = "
            f"{net_displacement:.6f} m"
        )

        print(
            f"Path length        = "
            f"{path_length:.6f} m"
        )

        print(
            f"Path efficiency    = "
            f"{path_efficiency * 100:.6f}%"
        )

        print()

        print(
            f"Final roll         = "
            f"{math.degrees(final_roll):.6f}°"
        )

        print(
            f"Final pitch        = "
            f"{math.degrees(final_pitch):.6f}°"
        )

        print(
            f"Final yaw          = "
            f"{math.degrees(final_yaw):.6f}°"
        )

        print()

        print(
            "LEFT ACTION"
        )

        print(
            f"  Mean = "
            f"{mean_left:.6f}"
        )

        print(
            f"  Std  = "
            f"{std_left:.6f}"
        )

        print()

        print(
            "RIGHT ACTION"
        )

        print(
            f"  Mean = "
            f"{mean_right:.6f}"
        )

        print(
            f"  Std  = "
            f"{std_right:.6f}"
        )

        print()

        print(
            f"Mean |left-right| = "
            f"{mean_difference:.6f}"
        )

        print()

        if final_distance < GOAL_THRESHOLD:

            print(
                "[PASS] PPO reached the goal."
            )

        else:

            print(
                "[FAIL] PPO did not reach the goal."
            )

        if (
            path_efficiency
            >= 0.99
        ):

            print(
                "[PASS] Nearly straight trajectory."
            )

        else:

            print(
                "[INFO] Trajectory is not perfectly straight."
            )

        if (
            abs(
                math.degrees(final_roll)
            )
            < 1.0
            and
            abs(
                math.degrees(final_pitch)
            )
            < 1.0
        ):

            print(
                "[PASS] Robot remains physically stable."
            )

        else:

            print(
                "[WARNING] Robot attitude changed "
                "by more than 1 degree."
            )

        print()
        print("=" * 80)
        print("V3.23.3 COMPLETE")
        print("=" * 80)

        print()
        print(
            "Trajectory CSV:"
        )

        print(
            f"  {OUTPUT_PATH}"
        )

        print()
        print(
            "Next:"
        )

        print(
            "  V3.23.4 — Multi-episode evaluation"
        )

    finally:

        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()