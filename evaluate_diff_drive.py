from robot_env import RobotReachEnv
from stable_baselines3 import PPO


env = RobotReachEnv()

model = PPO.load("ppo_diff_drive")

successes = 0

for episode in range(10):

    observation, info = env.reset(seed=episode)

    total_reward = 0.0

    for step in range(300):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    if terminated:
        successes += 1

    print(
        f"Episode {episode + 1}: "
        f"steps={step + 1}, "
        f"reward={total_reward:.2f}, "
        f"distance={env.previous_distance:.3f}, "
        f"success={terminated}"
    )

print()
print(f"Success rate: {successes}/10")

env.close()