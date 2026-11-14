from robot_env import RobotReachEnv
from stable_baselines3 import PPO

env = RobotReachEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64])
)

model.learn(total_timesteps=100000)

model.save("ppo_diff_drive_v24")

env.close()

print()
print("V2.4 training completed.")
print("Model saved as: ppo_diff_drive_v24")