import sys
from pathlib import Path

import numpy as np
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import DifferentialDriveEnv


# =========================================================
# TEST UTILITY
# =========================================================

def check(condition, message):

    if condition:
        print(f"{message}: PASS")
        return True

    print(f"{message}: FAIL")
    return False


def get_pose(env):

    position, orientation = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    return np.array(position), orientation


# =========================================================
# 1. OBSERVATION INTEGRITY
# =========================================================

def test_observation_integrity():

    print()
    print("========================================")
    print("1. OBSERVATION INTEGRITY TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        observation, info = env.reset(seed=42)

        shape_ok = check(
            observation.shape == (46,),
            "Observation shape",
        )

        dtype_ok = check(
            observation.dtype == np.float32,
            "Observation dtype",
        )

        finite_ok = check(
            np.all(np.isfinite(observation)),
            "Initial observation finite",
        )

        space_ok = check(
            env.observation_space.contains(
                observation
            ),
            "Observation space",
        )

        observations_finite = True

        for _ in range(100):

            action = env.action_space.sample()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if not np.all(
                np.isfinite(observation)
            ):
                observations_finite = False
                break

            if terminated or truncated:
                env.reset()

        long_run_ok = check(
            observations_finite,
            "Long-run observations finite",
        )

        return (
            shape_ok
            and dtype_ok
            and finite_ok
            and space_ok
            and long_run_ok
        )

    finally:

        env.close()


# =========================================================
# 2. SENSOR RESPONSE
# =========================================================

def test_sensor_response():

    print()
    print("========================================")
    print("2. SENSOR RESPONSE TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        initial_observation, _ = (
            env.reset(seed=42)
        )

        # -------------------------------------------------
        # Encoder indices:
        #
        # 0 = left angle
        # 1 = right angle
        # 2 = left velocity
        # 3 = right velocity
        # -------------------------------------------------

        initial_left_angle = (
            initial_observation[0]
        )

        initial_right_angle = (
            initial_observation[1]
        )

        # -------------------------------------------------
        # Move forward.
        # -------------------------------------------------

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        observation = initial_observation

        for _ in range(5):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if terminated or truncated:
                break

        left_angle_changed = (
            abs(
                observation[0] -
                initial_left_angle
            ) > 0.01
        )

        right_angle_changed = (
            abs(
                observation[1] -
                initial_right_angle
            ) > 0.01
        )

        encoder_ok = check(
            left_angle_changed
            and right_angle_changed,
            "Encoder responds to wheel motion",
        )

        # -------------------------------------------------
        # Rotate.
        # -------------------------------------------------

        env.reset(seed=42)

        initial_observation = (
            env._get_observation()
        )

        initial_yaw = (
            initial_observation[6]
        )

        rotation_action = np.array(
            [-1.0, 1.0],
            dtype=np.float32,
        )

        observation = initial_observation

        for _ in range(5):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                rotation_action
            )

            if terminated or truncated:
                break

        yaw_changed = (
            abs(
                observation[6] -
                initial_yaw
            ) > 0.01
        )

        imu_ok = check(
            yaw_changed,
            "IMU responds to rotation",
        )

        # -------------------------------------------------
        # LiDAR structure.
        #
        # Indices 10:46 = 36 LiDAR ranges.
        # -------------------------------------------------

        lidar = observation[10:46]

        lidar_ok = check(
            len(lidar) == 36
            and np.all(
                np.isfinite(lidar)
            )
            and np.all(
                lidar >= 0.0
            )
            and np.all(
                lidar <= 5.0
            ),
            "LiDAR response and limits",
        )

        return (
            encoder_ok
            and imu_ok
            and lidar_ok
        )

    finally:

        env.close()


# =========================================================
# 3. IDEAL VS NOISY SENSOR TEST
# =========================================================

def test_noise():

    print()
    print("========================================")
    print("3. IDEAL VS NOISY SENSOR TEST")
    print("========================================")
    print()

    ideal_env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    noisy_env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        ideal_observation, _ = (
            ideal_env.reset(seed=42)
        )

        noisy_observation, _ = (
            noisy_env.reset(seed=42)
        )

        difference = np.abs(
            ideal_observation -
            noisy_observation
        )

        noise_present = (
            np.any(difference > 1e-8)
        )

        noise_ok = check(
            noise_present,
            "Noise changes sensor observation",
        )

        finite_ok = check(
            np.all(
                np.isfinite(
                    noisy_observation
                )
            ),
            "Noisy observation finite",
        )

        return (
            noise_ok
            and finite_ok
        )

    finally:

        ideal_env.close()
        noisy_env.close()


# =========================================================
# 4. PROGRESS REWARD
# =========================================================

def test_progress_reward():

    print()
    print("========================================")
    print("4. PROGRESS REWARD TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        rewards = []

        for _ in range(10):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                np.array(
                    [1.0, 1.0],
                    dtype=np.float32,
                )
            )

            rewards.append(reward)

            if terminated or truncated:
                break

        positive_progress = any(
            info["progress"] > 0.0
            for _ in [0]
        )

        progress_ok = check(
            positive_progress,
            "Forward movement produces progress",
        )

        reward_ok = check(
            max(rewards) > 0.0,
            "Forward movement produces positive reward",
        )

        return (
            progress_ok
            and reward_ok
        )

    finally:

        env.close()


# =========================================================
# 5. AWAY-FROM-GOAL REWARD
# =========================================================

def test_away_reward():

    print()
    print("========================================")
    print("5. AWAY-FROM-GOAL REWARD TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        initial_distance = (
            env.previous_distance
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            np.array(
                [-1.0, -1.0],
                dtype=np.float32,
            )
        )

        distance_increased = (
            info["distance_to_goal"] >
            initial_distance
        )

        progress_negative = (
            info["progress"] < 0.0
        )

        distance_ok = check(
            distance_increased,
            "Moving backward increases goal distance",
        )

        reward_direction_ok = check(
            progress_negative,
            "Moving away produces negative progress",
        )

        return (
            distance_ok
            and reward_direction_ok
        )

    finally:

        env.close()


# =========================================================
# 6. COLLISION REWARD
# =========================================================

def test_collision_reward():

    print()
    print("========================================")
    print("6. COLLISION REWARD TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        obstacle_collision_shape = (
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    0.20,
                    0.20,
                    0.30,
                ],
            )
        )

        obstacle_id = (
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=(
                    obstacle_collision_shape
                ),
                basePosition=[
                    0.70,
                    0.0,
                    0.30,
                ],
            )
        )

        collision_detected = False
        collision_reward = None

        for _ in range(30):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                np.array(
                    [1.0, 1.0],
                    dtype=np.float32,
                )
            )

            if info["collision"]:

                collision_detected = True
                collision_reward = reward
                break

        collision_ok = check(
            collision_detected,
            "Collision detected",
        )

        penalty_ok = check(
            collision_reward is not None
            and collision_reward < -5.0,
            "Collision produces strong penalty",
        )

        return (
            collision_ok
            and penalty_ok
        )

    finally:

        env.close()


# =========================================================
# 7. GOAL REWARD
# =========================================================

def test_goal_reward():

    print()
    print("========================================")
    print("7. GOAL REWARD TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        env.goal_position = np.array(
            [0.10, 0.0],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            np.array(
                [1.0, 1.0],
                dtype=np.float32,
            )
        )

        goal_ok = check(
            info["goal_reached"] is True,
            "Goal reached",
        )

        reward_ok = check(
            reward > 5.0,
            "Goal produces positive terminal reward",
        )

        termination_ok = check(
            terminated is True,
            "Goal terminates episode",
        )

        return (
            goal_ok
            and reward_ok
            and termination_ok
        )

    finally:

        env.close()


# =========================================================
# 8. TIME-LIMIT TRUNCATION
# =========================================================

def test_time_limit():

    print()
    print("========================================")
    print("8. TIME-LIMIT TRUNCATION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=5,
    )

    try:

        env.reset(seed=42)

        terminated = False
        truncated = False

        for _ in range(5):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            )

            if terminated or truncated:
                break

        truncated_ok = check(
            truncated is True,
            "Time limit produces truncation",
        )

        terminated_ok = check(
            terminated is False,
            "Time limit does not cause termination",
        )

        return (
            truncated_ok
            and terminated_ok
        )

    finally:

        env.close()


# =========================================================
# 9. LONG-RUN STABILITY
# =========================================================

def test_long_run():

    print()
    print("========================================")
    print("9. LONG-RUN STABILITY TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        observation, _ = (
            env.reset(seed=42)
        )

        finite_ok = True
        reward_finite_ok = True

        completed_steps = 0

        for _ in range(500):

            action = env.action_space.sample()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            completed_steps += 1

            if not np.all(
                np.isfinite(observation)
            ):
                finite_ok = False
                break

            if not np.isfinite(reward):
                reward_finite_ok = False
                break

            if terminated or truncated:

                observation, _ = (
                    env.reset()
                )

        observation_ok = check(
            finite_ok,
            "500-step observations finite",
        )

        reward_ok = check(
            reward_finite_ok,
            "500-step rewards finite",
        )

        steps_ok = check(
            completed_steps >= 100,
            "Long-run simulation completed",
        )

        return (
            observation_ok
            and reward_ok
            and steps_ok
        )

    finally:

        env.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("========================================")
    print("V3.20.3 OBSERVATION + REWARD VALIDATION")
    print("========================================")

    results = []

    results.append(
        test_observation_integrity()
    )

    results.append(
        test_sensor_response()
    )

    results.append(
        test_noise()
    )

    results.append(
        test_progress_reward()
    )

    results.append(
        test_away_reward()
    )

    results.append(
        test_collision_reward()
    )

    results.append(
        test_goal_reward()
    )

    results.append(
        test_time_limit()
    )

    results.append(
        test_long_run()
    )

    print()
    print("========================================")
    print("FINAL VALIDATION")
    print("========================================")
    print()

    test_names = [
        "Observation integrity",
        "Sensor response",
        "Ideal vs noisy sensors",
        "Progress reward",
        "Away-from-goal reward",
        "Collision reward",
        "Goal reward",
        "Time-limit truncation",
        "Long-run stability",
    ]

    for name, result in zip(
        test_names,
        results,
    ):

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{name:<30} : "
            f"{status}"
        )

    print()

    if all(results):

        print(
            "V3.20.3 OBSERVATION + REWARD "
            "VALIDATION: PASS"
        )

    else:

        print(
            "V3.20.3 OBSERVATION + REWARD "
            "VALIDATION: FAIL"
        )


if __name__ == "__main__":
    main()