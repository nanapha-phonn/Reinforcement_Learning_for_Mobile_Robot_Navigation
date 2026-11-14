import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from environment import DifferentialDriveEnv


def main():

    print("=== V3.21.4 GOAL OBSERVATION TEST ===")

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        # -------------------------------------------------
        # Reset environment
        # -------------------------------------------------

        observation, info = env.reset(seed=42)

        # -------------------------------------------------
        # Basic structure
        # -------------------------------------------------

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "Observation dtype:",
            observation.dtype
        )

        assert observation.shape == (48,)
        assert observation.dtype == np.float32

        print("Shape: PASS")
        print("Dtype: PASS")

        # -------------------------------------------------
        # Finite values
        # -------------------------------------------------

        assert np.all(
            np.isfinite(observation)
        )

        print("Finite values: PASS")

        # -------------------------------------------------
        # Observation bounds
        # -------------------------------------------------

        assert np.all(
            observation >= -1.0
        )

        assert np.all(
            observation <= 1.0
        )

        print("Range [-1, 1]: PASS")

        # -------------------------------------------------
        # Gymnasium observation space
        # -------------------------------------------------

        assert env.observation_space.shape == (48,)

        assert env.observation_space.contains(
            observation
        )

        print("Observation space: PASS")

        # -------------------------------------------------
        # Goal observation
        #
        # Index 46 = goal distance
        # Index 47 = goal bearing
        # -------------------------------------------------

        goal_distance = observation[46]
        goal_bearing = observation[47]

        print(
            f"Goal distance : {goal_distance:.6f}"
        )

        print(
            f"Goal bearing  : {goal_bearing:.6f}"
        )

        assert 0.0 <= goal_distance <= 1.0

        assert -1.0 <= goal_bearing <= 1.0

        print("Goal distance range: PASS")
        print("Goal bearing range: PASS")

        # -------------------------------------------------
        # Move toward the goal
        # -------------------------------------------------

        initial_distance = goal_distance

        observation, reward, terminated, truncated, info = (
            env.step(
                np.array(
                    [1.0, 1.0],
                    dtype=np.float32,
                )
            )
        )

        new_distance = observation[46]

        print(
            f"Initial goal distance : "
            f"{initial_distance:.6f}"
        )

        print(
            f"New goal distance     : "
            f"{new_distance:.6f}"
        )

        assert new_distance < initial_distance

        print("Goal distance response: PASS")

        # -------------------------------------------------
        # Long-run stability
        # -------------------------------------------------

        for _ in range(500):

            observation, reward, terminated, truncated, info = (
                env.step(
                    np.array(
                        [0.0, 0.0],
                        dtype=np.float32,
                    )
                )
            )

            assert observation.shape == (48,)

            assert observation.dtype == np.float32

            assert np.all(
                np.isfinite(observation)
            )

            assert np.all(
                observation >= -1.0
            )

            assert np.all(
                observation <= 1.0
            )

            if terminated or truncated:
                break

        print("500-step stability: PASS")

        print(
            "\nV3.21.4 GOAL OBSERVATION TEST: PASS"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()