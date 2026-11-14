import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math
import os
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from environment import DifferentialDriveEnv

MODEL_PATH = PROJECT_ROOT / "models" / "v3236_generalized_ppo.zip"

NUM_CASES = 20
MAX_STEPS = 1000


TEST_CASES = [
    # start_x, start_y, start_yaw_deg, goal_x, goal_y
    (0.0, 0.0,   0.0,   3.0,  0.0),
    (0.0, 0.0,   0.0,   0.0,  3.0),
    (0.0, 0.0,   0.0,   0.0, -3.0),
    (0.0, 0.0,   0.0,   3.0,  1.0),
    (0.0, 0.0,   0.0,   3.0, -1.0),
    (0.0, 0.0,   0.0,   2.0,  2.0),
    (0.0, 0.0,   0.0,   2.0, -2.0),

    (1.0, 1.0,   0.0,   3.0,  1.0),
    (-1.0, 1.0, 0.0,   2.0, -1.0),

    (0.0, 0.0,  90.0,  3.0,  0.0),
    (0.0, 0.0, -90.0,  3.0,  0.0),
    (0.0, 0.0, 180.0,  3.0,  0.0),

    (0.0, 0.0,  90.0,  0.0,  3.0),
    (0.0, 0.0, -90.0,  0.0, -3.0),

    (1.0, -1.0, 0.0,  3.0,  1.0),
    (-1.0, -1.0, 180.0, 2.0,  2.0),

    (0.5, 0.5, 45.0,   3.0, -1.0),
    (-0.5, 0.5, -45.0, 2.0, -2.0),
    (0.5, -0.5, 135.0, 2.0, 2.0),
    (-0.5, -0.5, -135.0, 2.0, 1.0),
]


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def set_test_configuration(env, start_x, start_y, start_yaw_deg, goal_x, goal_y):
    yaw = math.radians(start_yaw_deg)

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [start_x, start_y, env.chassis_center_z],
        p.getQuaternionFromEuler([0.0, 0.0, yaw]),
    )

    p.resetBaseVelocity(
        env.robot_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
    )

    env.goal_position = np.array(
        [goal_x, goal_y],
        dtype=np.float32,
    )

    env.step_count = 0

    for _ in range(120):
        p.stepSimulation()

    position, _ = p.getBasePositionAndOrientation(env.robot_id)

    distance = math.sqrt(
        (goal_x - position[0]) ** 2
        + (goal_y - position[1]) ** 2
    )

    env.previous_distance = distance


def get_metrics(env):
    position, orientation = p.getBasePositionAndOrientation(env.robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

    dx = env.goal_position[0] - position[0]
    dy = env.goal_position[1] - position[1]

    distance = math.sqrt(dx * dx + dy * dy)

    world_goal_angle = math.atan2(dy, dx)
    bearing = wrap_angle(world_goal_angle - yaw)

    return {
        "x": position[0],
        "y": position[1],
        "distance": distance,
        "bearing_deg": math.degrees(bearing),
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(wrap_angle(yaw)),
    }


def main():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("=" * 70)
    print("V3.23.7 GENERALIZED PPO EVALUATION")
    print("=" * 70)

    print()
    print("Loading model:")
    print(MODEL_PATH)

    model = PPO.load(MODEL_PATH)

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
    )

    results = []

    for case_index, case in enumerate(TEST_CASES[:NUM_CASES], start=1):

        (
            start_x,
            start_y,
            start_yaw_deg,
            goal_x,
            goal_y,
        ) = case

        print()
        print("-" * 70)
        print(f"CASE {case_index:02d}")
        print("-" * 70)

        print(
            f"Start = ({start_x:.2f}, {start_y:.2f}) "
            f"yaw={start_yaw_deg:.1f}°"
        )

        print(
            f"Goal  = ({goal_x:.2f}, {goal_y:.2f})"
        )

        observation, info = env.reset(seed=1000 + case_index)

        set_test_configuration(
            env,
            start_x,
            start_y,
            start_yaw_deg,
            goal_x,
            goal_y,
        )

        observation = env._get_observation()

        total_reward = 0.0
        collision = False
        success = False
        path_length = 0.0

        previous_x, previous_y = start_x, start_y

        for step in range(1, MAX_STEPS + 1):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            metrics = get_metrics(env)

            dx = metrics["x"] - previous_x
            dy = metrics["y"] - previous_y

            path_length += math.sqrt(dx * dx + dy * dy)

            previous_x = metrics["x"]
            previous_y = metrics["y"]

            if metrics["distance"] < 0.20:
                success = True

            if env._check_collision():
                collision = True

            if terminated or truncated:
                break

        metrics = get_metrics(env)

        success = metrics["distance"] < 0.20

        print()
        print(f"Steps          = {step}")
        print(f"Reward         = {total_reward:.6f}")
        print(f"Final distance = {metrics['distance']:.6f} m")
        print(f"Final bearing  = {metrics['bearing_deg']:.3f}°")
        print(f"Final yaw      = {metrics['yaw_deg']:.3f}°")
        print(f"Roll           = {metrics['roll_deg']:.6f}°")
        print(f"Pitch          = {metrics['pitch_deg']:.6f}°")
        print(f"Path length    = {path_length:.6f} m")
        print(f"Success        = {success}")
        print(f"Collision      = {collision}")

        if success:
            print("[PASS] Goal reached.")
        else:
            print("[FAIL] Goal not reached.")

        results.append(
            {
                "success": success,
                "collision": collision,
                "reward": total_reward,
                "steps": step,
                "distance": metrics["distance"],
                "bearing": abs(metrics["bearing_deg"]),
                "path_length": path_length,
            }
        )

    env.close()

    successes = sum(r["success"] for r in results)
    collisions = sum(r["collision"] for r in results)

    success_rate = 100.0 * successes / len(results)

    avg_distance = np.mean(
        [r["distance"] for r in results]
    )

    avg_bearing = np.mean(
        [r["bearing"] for r in results]
    )

    avg_steps = np.mean(
        [r["steps"] for r in results]
    )

    avg_reward = np.mean(
        [r["reward"] for r in results]
    )

    print()
    print("=" * 70)
    print("V3.23.7 GENERALIZED PPO EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Cases tested          = {len(results)}")
    print(f"Successes             = {successes}/{len(results)}")
    print(f"Success rate          = {success_rate:.1f}%")
    print(f"Collisions            = {collisions}")
    print(f"Average final distance = {avg_distance:.6f} m")
    print(f"Average |bearing|      = {avg_bearing:.3f}°")
    print(f"Average episode steps  = {avg_steps:.2f}")
    print(f"Average reward         = {avg_reward:.6f}")

    print()

    if success_rate >= 80.0:
        print("[PASS] Strong generalized navigation performance.")
    elif success_rate >= 60.0:
        print("[PARTIAL] Generalized navigation is working but has failures.")
    else:
        print("[FAIL] Generalization is still weak.")

    print()
    print("V3.23.7 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()