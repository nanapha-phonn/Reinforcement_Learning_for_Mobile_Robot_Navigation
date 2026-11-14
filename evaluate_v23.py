from robot_env import RobotReachEnv
from stable_baselines3 import PPO
import numpy as np

env = RobotReachEnv()

model = PPO.load(
    "ppo_diff_drive_v23",
    env=env
)

successes = 0
episode_rewards = []
episode_steps = []
final_distances = []

for episode in range(100):

    obs, info = env.reset()

    total_reward = 0.0
    steps = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    final_distance = float(obs[0])

    success = final_distance < 0.25

    if success:
        successes += 1

    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    final_distances.append(final_distance)

    print(
        f"Episode {episode + 1:3d}: "
        f"steps={steps:3d}, "
        f"reward={total_reward:7.2f}, "
        f"distance={final_distance:.3f}, "
        f"success={success}"
    )

success_rate = (successes / 100) * 100

print()
print("=" * 60)
print("V2.3 FINAL EVALUATION")
print("=" * 60)

print(f"Success: {successes}/100")
print(f"Success rate: {success_rate:.1f}%")
print(f"Average steps: {np.mean(episode_steps):.1f}")
print(f"Average reward: {np.mean(episode_rewards):.2f}")
print(f"Average final distance: {np.mean(final_distances):.3f}")
print(f"Maximum final distance: {np.max(final_distances):.3f}")

env.close()