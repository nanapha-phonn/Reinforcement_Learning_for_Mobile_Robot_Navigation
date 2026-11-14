import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import DifferentialDriveEnv


def validate_observation(env, observation, label):
    print(f"\n--- {label} ---")

    print("Shape :", observation.shape)
    print("Dtype :", observation.dtype)
    print("Min   :", np.min(observation))
    print("Max   :", np.max(observation))

    assert observation.shape == (46,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    assert env.observation_space.contains(observation)

    print("Shape: PASS")
    print("Dtype: PASS")
    print("Finite values: PASS")
    print("Range [-1, 1]: PASS")
    print("Observation space: PASS")


def main():
    print("=== V3.20.4 OBSERVATION NORMALIZATION TEST ===")

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
    )

    try:
        observation, info = env.reset(seed=42)

        validate_observation(
            env,
            observation,
            "INITIAL OBSERVATION",
        )

        print("\n--- SENSOR GROUPS ---")

        encoder = observation[0:4]
        imu = observation[4:10]
        lidar = observation[10:46]

        print("Encoder values :", encoder)
        print("IMU values     :", imu)
        print("LiDAR min      :", np.min(lidar))
        print("LiDAR max      :", np.max(lidar))

        assert np.all(encoder >= -1.0)
        assert np.all(encoder <= 1.0)

        assert np.all(imu >= -1.0)
        assert np.all(imu <= 1.0)

        assert np.all(lidar >= 0.0)
        assert np.all(lidar <= 1.0)

        print("Encoder range: PASS")
        print("IMU range: PASS")
        print("LiDAR range [0, 1]: PASS")

        print("\n--- ACTION RESPONSE ---")

        observation_before = observation.copy()

        observation_after, reward, terminated, truncated, info = env.step(
            np.array([1.0, 1.0], dtype=np.float32)
        )

        difference = np.max(
            np.abs(observation_after - observation_before)
        )

        print("Maximum observation change:", difference)

        assert difference > 0.0

        validate_observation(
            env,
            observation_after,
            "AFTER FORWARD ACTION",
        )

        print("Sensor response: PASS")

        print("\n--- NOISY OBSERVATION ---")

        env.close()

        noisy_env = DifferentialDriveEnv(
            use_noise=True,
            render_mode="direct",
        )

        try:
            noisy_observation, _ = noisy_env.reset(seed=42)

            validate_observation(
                noisy_env,
                noisy_observation,
                "NOISY OBSERVATION",
            )

            print("Noisy observation: PASS")

        finally:
            noisy_env.close()

        print("\n--- LONG-RUN STABILITY ---")

        stability_env = DifferentialDriveEnv(
            use_noise=True,
            render_mode="direct",
        )

        try:
            observation, _ = stability_env.reset(seed=42)

            for step in range(500):
                action = np.array(
                    [
                        0.5,
                        0.5,
                    ],
                    dtype=np.float32,
                )

                observation, reward, terminated, truncated, info = (
                    stability_env.step(action)
                )

                assert observation.shape == (46,)
                assert observation.dtype == np.float32
                assert np.all(np.isfinite(observation))
                assert np.all(observation >= -1.0)
                assert np.all(observation <= 1.0)

                if terminated or truncated:
                    observation, _ = stability_env.reset()

            print("500-step stability: PASS")

        finally:
            stability_env.close()

        print("\nV3.20.4 OBSERVATION NORMALIZATION TEST: PASS")

    finally:
        if env.is_connected():
            env.close()


if __name__ == "__main__":
    main()