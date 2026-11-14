from robot_env import RobotReachEnv
from stable_baselines3 import PPO


env = RobotReachEnv()

model = PPO.load("ppo_diff_drive_v21")

successes = 0
total_steps = 0
total_reward = 0.0

for episode in range(10):

    observation, info = env.reset(seed=episode)

    episode_reward = 0.0

    for step in range(300):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

        if terminated or truncated:
            break

    if terminated:
        successes += 1

    total_steps += step + 1
    total_reward += episode_reward

    print(
        f"Episode {episode + 1}: "
        f"steps={step + 1}, "
        f"reward={episode_reward:.2f}, "
        f"distance={env.previous_distance:.3f}, "
        f"success={terminated}"
    )

print()
print(f"Success rate: {successes}/10")
print(f"Average steps: {total_steps / 10:.1f}")
print(f"Average reward: {total_reward / 10:.2f}")

env.close()