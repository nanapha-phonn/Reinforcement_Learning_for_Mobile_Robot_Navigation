"""
V3.23.6 — GENERALIZED PPO TRAINING

Purpose
-------
Train PPO on randomized start positions, start orientations,
and goal positions.

V3.22 physics/environment remains FROZEN.

This script changes ONLY the training distribution.

It does NOT modify:
    robot.py
    environment.py
    robot physics
    sensor definitions
    observation structure
    reward function

Training distribution
---------------------
Each episode receives:

    random start position
    random start yaw
    random goal position

The goal is to train a policy that can navigate from
different initial states instead of memorizing one fixed
trajectory.

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3236_generalized_ppo.py
"""

import sys
import math
import random
from pathlib import Path

import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


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
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

TOTAL_TIMESTEPS = 200_704

MODEL_NAME = "v3236_generalized_ppo"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / f"{MODEL_NAME}.zip"
)

LOG_PATH = (
    PROJECT_ROOT
    / "logs"
    / MODEL_NAME
)


# ============================================================
# RANDOMIZED NAVIGATION SPACE
# ============================================================

# Start position range.

START_X_MIN = -1.0
START_X_MAX = 1.0

START_Y_MIN = -1.0
START_Y_MAX = 1.0


# Goal position range.

GOAL_X_MIN = -3.0
GOAL_X_MAX = 3.0

GOAL_Y_MIN = -3.0
GOAL_Y_MAX = 3.0


# Minimum start-goal distance.

MIN_GOAL_DISTANCE = 1.5


# Maximum start-goal distance.

MAX_GOAL_DISTANCE = 5.0


# ============================================================
# SETTLING
# ============================================================

SETTLE_STEPS = 120


# ============================================================
# GENERALIZED ENVIRONMENT
# ============================================================


class GeneralizedNavigationEnv(DifferentialDriveEnv):
    """
    DifferentialDriveEnv with randomized initial states.

    The underlying physics, sensors, observations, and reward
    remain unchanged.
    """

    def __init__(self, **kwargs):

        super().__init__(
            **kwargs
        )

        self.episode_rng = random.Random(
            SEED
        )

    # --------------------------------------------------------
    # Sample navigation configuration
    # --------------------------------------------------------

    def _sample_configuration(self):

        for _ in range(1000):

            start_x = self.episode_rng.uniform(
                START_X_MIN,
                START_X_MAX,
            )

            start_y = self.episode_rng.uniform(
                START_Y_MIN,
                START_Y_MAX,
            )

            goal_x = self.episode_rng.uniform(
                GOAL_X_MIN,
                GOAL_X_MAX,
            )

            goal_y = self.episode_rng.uniform(
                GOAL_Y_MIN,
                GOAL_Y_MAX,
            )

            distance = math.sqrt(
                (goal_x - start_x) ** 2
                +
                (goal_y - start_y) ** 2
            )

            if (
                distance >= MIN_GOAL_DISTANCE
                and
                distance <= MAX_GOAL_DISTANCE
            ):

                yaw = self.episode_rng.uniform(
                    -math.pi,
                    math.pi,
                )

                return (
                    start_x,
                    start_y,
                    yaw,
                    goal_x,
                    goal_y,
                )

        raise RuntimeError(
            "Could not generate a valid "
            "start/goal configuration."
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
        seed=None,
        options=None,
    ):

        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        # ----------------------------------------------------
        # Use Gymnasium seed when provided.
        # ----------------------------------------------------

        if seed is not None:

            self.episode_rng.seed(
                seed
            )

        # ----------------------------------------------------
        # Generate randomized configuration.
        # ----------------------------------------------------

        (
            start_x,
            start_y,
            yaw,
            goal_x,
            goal_y,
        ) = self._sample_configuration()

        # ----------------------------------------------------
        # Convert yaw to quaternion.
        # ----------------------------------------------------

        quaternion = (
            p.getQuaternionFromEuler(
                [
                    0.0,
                    0.0,
                    yaw,
                ]
            )
        )

        # ----------------------------------------------------
        # Set robot pose.
        # ----------------------------------------------------

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [
                start_x,
                start_y,
                CHASSIS_CENTER_Z,
            ],
            quaternion,
        )

        stop_robot(
            self.robot_id
        )

        # ----------------------------------------------------
        # Set randomized goal.
        # ----------------------------------------------------

        self.goal_position = np.array(
            [
                goal_x,
                goal_y,
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Settle robot.
        # ----------------------------------------------------

        for _ in range(
            SETTLE_STEPS
        ):

            p.stepSimulation()

        # ----------------------------------------------------
        # Read final settled pose.
        # ----------------------------------------------------

        position, orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        # ----------------------------------------------------
        # Recalculate initial distance.
        # ----------------------------------------------------

        distance = math.sqrt(
            (
                self.goal_position[0]
                -
                position[0]
            ) ** 2
            +
            (
                self.goal_position[1]
                -
                position[1]
            ) ** 2
        )

        self.previous_distance = (
            distance
        )

        self.step_count = 0

        # ----------------------------------------------------
        # Observation.
        # ----------------------------------------------------

        observation = (
            self._get_observation()
        )

        # ----------------------------------------------------
        # Extra diagnostic information.
        # ----------------------------------------------------

        info = dict(info)

        info.update(
            {
                "start_x": float(
                    position[0]
                ),
                "start_y": float(
                    position[1]
                ),
                "start_yaw": float(
                    yaw
                ),
                "goal_x": float(
                    goal_x
                ),
                "goal_y": float(
                    goal_y
                ),
                "goal_distance": float(
                    distance
                ),
            }
        )

        return observation, info


# ============================================================
# TRAINING CALLBACK
# ============================================================


class TrainingProgressCallback(
    BaseCallback
):
    """
    Print useful progress information
    without changing training behavior.
    """

    def __init__(
        self,
        print_freq=10_000,
    ):

        super().__init__()

        self.print_freq = (
            print_freq
        )

    def _on_step(self):

        if (
            self.num_timesteps
            %
            self.print_freq
            == 0
        ):

            print()
            print(
                "=" * 70
            )

            print(
                "V3.23.6 TRAINING PROGRESS"
            )

            print(
                f"Timesteps = "
                f"{self.num_timesteps:,}"
            )

            print(
                "=" * 70
            )

        return True


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "=" * 80
    )

    print(
        "V3.23.6 — GENERALIZED PPO TRAINING"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Physics/environment:"
    )

    print(
        "  FROZEN"
    )

    print()

    print(
        "Randomized:"
    )

    print(
        "  Start X/Y"
    )

    print(
        "  Start yaw"
    )

    print(
        "  Goal X/Y"
    )

    print()

    print(
        f"Total timesteps = "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Seed = {SEED}"
    )

    print()

    print(
        "Output model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = GeneralizedNavigationEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    env.reset(
        seed=SEED
    )

    # --------------------------------------------------------
    # Check observation/action spaces.
    # --------------------------------------------------------

    print(
        "Observation space:"
    )

    print(
        f"  {env.observation_space}"
    )

    print()

    print(
        "Action space:"
    )

    print(
        f"  {env.action_space}"
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.0,

        vf_coef=0.5,

        max_grad_norm=0.5,

        seed=SEED,

        verbose=1,

        tensorboard_log=str(
            LOG_PATH
        ),
    )

    print()

    print(
        "[PASS] PPO model created."
    )

    print()

    print(
        "Starting generalized training..."
    )

    print()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    callback = TrainingProgressCallback(
        print_freq=10_000
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        tb_log_name=MODEL_NAME,
        progress_bar=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(
        str(
            MODEL_PATH.with_suffix("")
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "V3.23.6 TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Model saved:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print()

    print(
        "TensorBoard logs:"
    )

    print(
        f"  {LOG_PATH}"
    )

    print()

    print(
        "Next:"
    )

    print(
        "  V3.23.6 evaluation"
    )

    print(
        "  Evaluate randomized start/goal cases."
    )

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()