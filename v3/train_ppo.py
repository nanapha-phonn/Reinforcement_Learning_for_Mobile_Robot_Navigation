import os
import math
import numpy as np

from stable_baselines3 import PPO
from environment import DifferentialDriveEnv


MODEL_PATH = "models/ppo_v3217_heading_alignment_navigation.zip"

NUM_EPISODES = 10
SUCCESS_DISTANCE = 0.20


def make_env():
    return DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )


def get_ground_truth_state(env):
    position, orientation = env.p.getBasePositionAndOrientation(
        env.robot_id
    )

    x = position[0]
    y = position[1]

    _, _, yaw = env.p.getEulerFromQuaternion(orientation)

    goal_x, goal_y = env.goal_position

    dx = goal_x - x
    dy = goal_y - y

    distance = math.sqrt(dx * dx + dy * dy)

    goal_angle = math.atan2(dy, dx)

    bearing = goal_angle - yaw
    bearing = math.atan2(
        math.sin(bearing),
        math.cos(bearing),
    )

    return x, y, yaw, distance, bearing


def main():

    print("=" * 60)
    print("V3.21.7 PPO EVALUATION")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("\nERROR: Model file not found:")
        print(MODEL_PATH)
        return

    print("\nModel:")
    print(MODEL_PATH)

    env = make_env()

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="auto",
    )

    print("Model loaded successfully.")

    print("\n--- ENVIRONMENT ---")
    print("Action space      :", env.action_space)
    print("Observation space :", env.observation_space)
    print("Observation shape :", env.observation_space.shape)

    print("\n--- EVALUATION ---")
    print("Episodes          :", NUM_EPISODES)
    print("Success threshold :", SUCCESS_DISTANCE, "m")

    results = []

    for episode in range(NUM_EPISODES):

        observation, info = env.reset()

        episode_reward = 0.0
        minimum_distance = float("inf")

        terminated = False
        truncated = False

        final_distance = None
        final_bearing = None
        final_yaw = None
        final_x = None
        final_y = None

        for step in range(1, env.max_episode_steps + 1):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += reward

            (
                x,
                y,
                yaw,
                distance,
                bearing,
            ) = get_ground_truth_state(env)

            minimum_distance = min(
                minimum_distance,
                distance,
            )

            final_x = x
            final_y = y
            final_yaw = yaw
            final_distance = distance
            final_bearing = bearing

            if terminated or truncated:
                break

        success = (
            terminated
            and final_distance <= SUCCESS_DISTANCE
        )

        results.append(
            {
                "episode": episode + 1,
                "success": success,
                "reward": episode_reward,
                "steps": step,
                "final_distance": final_distance,
                "minimum_distance": minimum_distance,
                "final_bearing": math.degrees(final_bearing),
                "final_yaw": math.degrees(final_yaw),
                "x": final_x,
                "y": final_y,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        status = "SUCCESS" if success else "FAIL"

        print(
            f"Episode {episode + 1:02d} | "
            f"{status:7s} | "
            f"Reward={episode_reward:8.3f} | "
            f"Steps={step:4d} | "
            f"Final D={final_distance:7.3f} m | "
            f"Min D={minimum_distance:7.3f} m | "
            f"Bearing={math.degrees(final_bearing):8.2f}°"
        )

    env.close()

    successes = sum(
        result["success"]
        for result in results
    )

    success_rate = successes / NUM_EPISODES

    average_reward = np.mean(
        [result["reward"] for result in results]
    )

    average_steps = np.mean(
        [result["steps"] for result in results]
    )

    average_final_distance = np.mean(
        [result["final_distance"] for result in results]
    )

    average_minimum_distance = np.mean(
        [result["minimum_distance"] for result in results]
    )

    average_final_bearing = np.mean(
        [abs(result["final_bearing"]) for result in results]
    )

    collision_count = sum(
        result["terminated"]
        and not result["success"]
        for result in results
    )

    print("\n" + "=" * 60)
    print("V3.21.7 FINAL EVALUATION")
    print("=" * 60)

    print(f"Success rate              : {successes}/{NUM_EPISODES} "
          f"({success_rate * 100:.1f}%)")

    print(f"Average reward            : {average_reward:.3f}")

    print(f"Average steps             : {average_steps:.1f}")

    print(f"Average final distance    : "
          f"{average_final_distance:.3f} m")

    print(f"Average minimum distance  : "
          f"{average_minimum_distance:.3f} m")

    print(f"Average |final bearing|   : "
          f"{average_final_bearing:.2f}°")

    print(f"Collision/other terminations: "
          f"{collision_count}")

    print("=" * 60)

    print("\nV3.21.7 PPO EVALUATION COMPLETE")


if __name__ == "__main__":
    main()