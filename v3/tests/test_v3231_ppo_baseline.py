"""
V3.23.1 — PPO BASELINE

Purpose
-------
Train the first PPO baseline using the frozen V3.22 environment.

Environment:
    DifferentialDriveEnv

Observation:
    48 values

Action:
    2 continuous wheel commands

Algorithm:
    PPO

Policy:
    MlpPolicy

Important:
    - Do NOT modify robot.py
    - Do NOT modify environment.py
    - Do NOT modify the reward function
    - Do NOT tune hyperparameters yet
    - This is the baseline experiment

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3231_ppo_baseline.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


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

MODEL_DIR = PROJECT_ROOT / "models"

LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_PATH = MODEL_DIR / "v3231_ppo_baseline"

TB_LOG_DIR = LOG_DIR / "v3231_ppo_baseline"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TOTAL_TIMESTEPS = 100_352

N_ENVS = 1

DEVICE = "auto"

EVAL_EPISODES = 10

MAX_EPISODE_STEPS = 1000


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================

def make_env(seed):

    def _init():

        env = DifferentialDriveEnv(
            use_noise=True,
            render_mode="direct",
            max_episode_steps=MAX_EPISODE_STEPS,
        )

        env = Monitor(env)

        env.reset(seed=seed)

        return env

    return _init


# ============================================================
# CREATE TRAINING ENVIRONMENT
# ============================================================

def create_training_env():

    set_random_seed(SEED)

    env = DummyVecEnv(
        [
            make_env(SEED)
        ]
    )

    return env


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

def validate_environment(env):

    print()
    print("=" * 80)
    print("ENVIRONMENT VALIDATION")
    print("=" * 80)

    observation = env.reset()

    print(
        f"Observation shape = "
        f"{observation.shape}"
    )

    print(
        f"Observation dtype = "
        f"{observation.dtype}"
    )

    print(
        f"Observation min = "
        f"{np.min(observation):.6f}"
    )

    print(
        f"Observation max = "
        f"{np.max(observation):.6f}"
    )

    if observation.shape != (
        N_ENVS,
        48,
    ):

        raise RuntimeError(
            "Unexpected observation shape."
        )

    if not np.all(
        np.isfinite(observation)
    ):

        raise RuntimeError(
            "Observation contains NaN/Inf."
        )

    if np.any(
        observation < -1.00001
    ):

        raise RuntimeError(
            "Observation below -1."
        )

    if np.any(
        observation > 1.00001
    ):

        raise RuntimeError(
            "Observation above +1."
        )

    print(
        "[PASS] Environment observation."
    )


# ============================================================
# CREATE PPO
# ============================================================

def create_model(env):

    print()
    print("=" * 80)
    print("CREATING PPO BASELINE")
    print("=" * 80)

    print(
        "Algorithm: PPO"
    )

    print(
        "Policy: MlpPolicy"
    )

    print(
        f"Observation dimension: "
        f"{env.observation_space.shape}"
    )

    print(
        f"Action dimension: "
        f"{env.action_space.shape}"
    )

    print(
        f"Device: {DEVICE}"
    )

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
        verbose=1,
        seed=SEED,
        tensorboard_log=str(
            TB_LOG_DIR
        ),
        device=DEVICE,
    )

    print()
    print(
        "[PASS] PPO model created."
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train(model):

    print()
    print("=" * 80)
    print("V3.23.1 PPO BASELINE TRAINING")
    print("=" * 80)

    print()
    print(
        f"Total timesteps: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Seed: {SEED}"
    )

    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    print()
    print(
        "[PASS] PPO training completed."
    )


# ============================================================
# SAVE
# ============================================================

def save_model(model):

    print()
    print("=" * 80)
    print("SAVING PPO BASELINE")
    print("=" * 80)

    model.save(
        str(MODEL_PATH)
    )

    print(
        f"Model saved to:"
    )

    print(
        f"  {MODEL_PATH}.zip"
    )

    if not MODEL_PATH.with_suffix(
        ".zip"
    ).exists():

        print(
            "[WARNING] Model file was "
            "not found after save."
        )

        return False

    print(
        "[PASS] Model saved."
    )

    return True


# ============================================================
# EVALUATION ENVIRONMENT
# ============================================================

def create_evaluation_env():

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    return env


# ============================================================
# ANGLE HELPER
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# ============================================================
# EVALUATE ONE EPISODE
# ============================================================

def evaluate_episode(
    model,
    env,
    seed,
):

    observation, info = env.reset(
        seed=seed
    )

    episode_reward = 0.0

    episode_length = 0

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        episode_reward += float(
            reward
        )

        episode_length += 1

        if episode_length >= MAX_EPISODE_STEPS:

            break

    # --------------------------------------------------------
    # Final robot state
    # --------------------------------------------------------

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

    goal_x = env.goal_position[0]

    goal_y = env.goal_position[1]

    final_distance = math.sqrt(
        (position[0] - goal_x) ** 2
        +
        (position[1] - goal_y) ** 2
    )

    goal_bearing = math.atan2(
        goal_y - position[1],
        goal_x - position[0],
    )

    relative_bearing = wrap_angle(
        goal_bearing - yaw
    )

    success = (
        final_distance < 0.20
    )

    collision = env._check_collision()

    return {
        "reward": episode_reward,
        "length": episode_length,
        "final_distance": final_distance,
        "bearing_deg": math.degrees(
            relative_bearing
        ),
        "roll_deg": math.degrees(
            roll
        ),
        "pitch_deg": math.degrees(
            pitch
        ),
        "yaw_deg": math.degrees(
            yaw
        ),
        "success": success,
        "collision": collision,
        "terminated": terminated,
        "truncated": truncated,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model):

    print()
    print("=" * 80)
    print("V3.23.1 PPO BASELINE EVALUATION")
    print("=" * 80)

    env = create_evaluation_env()

    results = []

    try:

        for episode in range(
            EVAL_EPISODES
        ):

            seed = (
                SEED
                + episode
            )

            result = evaluate_episode(
                model,
                env,
                seed,
            )

            results.append(
                result
            )

            print()
            print(
                f"Episode "
                f"{episode + 1:02d}/"
                f"{EVAL_EPISODES}"
            )

            print(
                f"  Reward        = "
                f"{result['reward']:.6f}"
            )

            print(
                f"  Length        = "
                f"{result['length']}"
            )

            print(
                f"  Final distance = "
                f"{result['final_distance']:.6f} m"
            )

            print(
                f"  Bearing       = "
                f"{result['bearing_deg']:.6f}°"
            )

            print(
                f"  Roll          = "
                f"{result['roll_deg']:.6f}°"
            )

            print(
                f"  Pitch         = "
                f"{result['pitch_deg']:.6f}°"
            )

            print(
                f"  Yaw           = "
                f"{result['yaw_deg']:.6f}°"
            )

            print(
                f"  Success       = "
                f"{result['success']}"
            )

            print(
                f"  Collision     = "
                f"{result['collision']}"
            )

    finally:

        env.close()

    return results


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 80)
    print("V3.23.1 PPO BASELINE SUMMARY")
    print("=" * 80)

    rewards = np.array(
        [
            r["reward"]
            for r in results
        ],
        dtype=np.float64,
    )

    lengths = np.array(
        [
            r["length"]
            for r in results
        ],
        dtype=np.float64,
    )

    distances = np.array(
        [
            r["final_distance"]
            for r in results
        ],
        dtype=np.float64,
    )

    bearings = np.array(
        [
            abs(r["bearing_deg"])
            for r in results
        ],
        dtype=np.float64,
    )

    rolls = np.array(
        [
            abs(r["roll_deg"])
            for r in results
        ],
        dtype=np.float64,
    )

    pitches = np.array(
        [
            abs(r["pitch_deg"])
            for r in results
        ],
        dtype=np.float64,
    )

    successes = sum(
        r["success"]
        for r in results
    )

    collisions = sum(
        r["collision"]
        for r in results
    )

    print()

    print(
        f"Average reward       = "
        f"{np.mean(rewards):.6f}"
    )

    print(
        f"Average episode len  = "
        f"{np.mean(lengths):.2f}"
    )

    print(
        f"Average final dist   = "
        f"{np.mean(distances):.6f} m"
    )

    print(
        f"Average |bearing|    = "
        f"{np.mean(bearings):.6f}°"
    )

    print(
        f"Average |roll|       = "
        f"{np.mean(rolls):.6f}°"
    )

    print(
        f"Average |pitch|      = "
        f"{np.mean(pitches):.6f}°"
    )

    print(
        f"Successes            = "
        f"{successes}/{len(results)}"
    )

    print(
        f"Collisions           = "
        f"{collisions}/{len(results)}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a BASELINE result."
    )

    print(
        "Do not tune PPO based on this result yet."
    )

    print(
        "V3.23.2 will analyze the training behavior."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("V3.23.1 — PPO BASELINE")
    print("=" * 80)

    print()
    print(
        "V3.22 environment: FROZEN"
    )

    print(
        "Physics modifications: NONE"
    )

    print(
        "Reward modifications: NONE"
    )

    print(
        "PPO tuning: NONE"
    )

    print(
        f"Training timesteps: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Evaluation episodes: "
        f"{EVAL_EPISODES}"
    )

    print()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(SEED)

    torch.manual_seed(
        SEED
    )

    set_random_seed(
        SEED
    )

    # --------------------------------------------------------
    # Training environment
    # --------------------------------------------------------

    training_env = create_training_env()

    try:

        validate_environment(
            training_env
        )

        model = create_model(
            training_env
        )

        train(
            model
        )

        saved = save_model(
            model
        )

        if not saved:

            print(
                "[FAIL] Could not verify "
                "saved PPO model."
            )

            return

    finally:

        training_env.close()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = evaluate_model(
        model
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("V3.23.1 COMPLETE")
    print("=" * 80)

    print()
    print(
        "PPO baseline model:"
    )

    print(
        f"  {MODEL_PATH}.zip"
    )

    print()
    print(
        "Next:"
    )

    print(
        "  V3.23.2 — PPO training diagnostics"
    )

    print()
    print(
        "Do NOT modify the environment yet."
    )

    print("=" * 80)


if __name__ == "__main__":

    main()