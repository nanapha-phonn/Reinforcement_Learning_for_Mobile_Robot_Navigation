from stable_baselines3 import PPO
from robot_env import RobotReachEnv


env = RobotReachEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

model.learn(
    total_timesteps=100000
)

model.save(
    "ppo_diff_drive_v21"
)

env.close()

print("V2.1 training complete.")