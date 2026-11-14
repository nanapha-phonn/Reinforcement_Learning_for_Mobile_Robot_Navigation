from robot_env import RobotReachEnv
from stable_baselines3 import PPO

env = RobotReachEnv()

model = PPO.load("ppo_robot")

for episode in range(5):

    observation, info = env.reset(seed=episode)

    total_reward = 0

    for step in range(200):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(
        f"Episode {episode + 1}: "
        f"steps={step + 1}, "
        f"reward={total_reward:.2f}, "
        f"distance={env.previous_distance:.3f}, "
        f"success={terminated}"
    )

env.close()