from robot_env import RobotReachEnv
from stable_baselines3 import PPO
import numpy as np


env = RobotReachEnv()

model = PPO.load("ppo_diff_drive_v25")

num_episodes = 100

successes = 0
steps_list = []
rewards_list = []
final_distances = []


for episode in range(num_episodes):

    obs, info = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    steps = 0

    while not terminated and not truncated:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    final_distance = obs[0]

    success = final_distance < env.success_radius

    if success:
        successes += 1

    steps_list.append(steps)
    rewards_list.append(total_reward)
    final_distances.append(final_distance)

    print(
        f"Episode {episode + 1:3d} | "
        f"Steps: {steps:3d} | "
        f"Reward: {total_reward:7.2f} | "
        f"Final distance: {final_distance:.3f} | "
        f"Success: {success}"
    )


success_rate = successes / num_episodes * 100


print()
print("=" * 50)
print("V2.5 EVALUATION")
print("=" * 50)

print(f"Success:          {successes}/{num_episodes}")
print(f"Success rate:     {success_rate:.1f}%")
print(f"Average steps:    {np.mean(steps_list):.1f}")
print(f"Average reward:   {np.mean(rewards_list):.2f}")
print(f"Average distance: {np.mean(final_distances):.3f}")
print(f"Max distance:     {np.max(final_distances):.3f}")
print(f"Min distance:     {np.min(final_distances):.3f}")


env.close()