import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import DifferentialDriveEnv


# ============================================================
# V3.21.5 CONFIGURATION
# ============================================================

MODEL_DIR = "models"
LOG_DIR = "logs"

MODEL_NAME = "ppo_v3215_goal_navigation"


# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    return DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== V3.21.5 PPO TRAINING ===")

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = make_env()

    print("\n--- ENVIRONMENT ---")
    print("Action space      :", env.action_space)
    print("Observation space :", env.observation_space)
    print("Observation shape :", env.observation_space.shape)

    # --------------------------------------------------------
    # Verify V3.21.4 goal-conditioned observation
    # --------------------------------------------------------

    expected_observation_shape = (48,)

    if env.observation_space.shape != expected_observation_shape:
        env.close()
        raise RuntimeError(
            f"Expected observation shape {expected_observation_shape}, "
            f"but got {env.observation_space.shape}"
        )

    print("48-D observation check: PASS")

    # --------------------------------------------------------
    # Gymnasium validation
    # --------------------------------------------------------

    print("\n--- GYMNASIUM CHECK ---")

    check_env(env, warn=True)

    print("Gymnasium check_env: PASS")

    # --------------------------------------------------------
    # PPO MODEL
    #
    # IMPORTANT:
    # Hyperparameters are unchanged from V3.21 baseline.
    # --------------------------------------------------------

    print("\n--- PPO MODEL ---")

    model = PPO(
        policy="MlpPolicy",
        env=env,

        # Same hyperparameters as V3.21 baseline
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
    )

    print("PPO model created successfully.")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n--- TRAINING ---")
    print("Timesteps         : 100,000")
    print("Observation       : 48-D")
    print("Goal conditioning : ENABLED")
    print("Sensor noise      : ENABLED")

    model.learn(
        total_timesteps=100_000,
        progress_bar=True,
    )

    # --------------------------------------------------------
    # SAVE NEW MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "ppo_v3216_heading_navigation",
    )

    model.save(model_path)

    print("\n--- TRAINING COMPLETE ---")
    print("Model saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Close environment
    # --------------------------------------------------------

    env.close()

    print("\nV3.21.5 PPO TRAINING: COMPLETE")


if __name__ == "__main__":
    main()