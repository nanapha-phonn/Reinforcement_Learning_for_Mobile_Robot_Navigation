import math
import os
import sys

import numpy as np

# Add the project root to Python's import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from environment import DifferentialDriveEnv

def test_environment_creation():
    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    print("Environment creation : PASS")

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (48,)

    env.close()


def test_reset():
    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    observation, info = env.reset(seed=42)

    print(
        "Observation shape    :",
        observation.shape,
    )

    print(
        "Observation dtype    :",
        observation.dtype,
    )

    assert observation.shape == (48,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)

    print("Reset                : PASS")
    print("Observation range    : PASS")

    env.close()


def test_reward_components():
    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    observation, info = env.reset(seed=42)

    action = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print()
    print("Reward:", reward)

    print(
        "Distance to goal     :",
        info["distance_to_goal"],
    )

    print(
        "Progress             :",
        info["progress"],
    )

    print(
        "Goal bearing         :",
        info["goal_bearing"],
    )

    print(
        "Heading weight       :",
        info["heading_weight"],
    )

    print(
        "Heading alignment    :",
        info["heading_alignment"],
    )

    print(
        "Heading reward       :",
        info["heading_reward"],
    )

    print(
        "Action penalty       :",
        info["action_penalty"],
    )

    print(
        "Collision reward     :",
        info["collision_reward"],
    )

    print(
        "Goal reward          :",
        info["goal_reward"],
    )

    print(
        "Collision            :",
        info["collision"],
    )

    print(
        "Goal reached         :",
        info["goal_reached"],
    )

    # Required reward information must exist.
    required_keys = [
        "step",
        "distance_to_goal",
        "progress",
        "goal_bearing",
        "heading_weight",
        "heading_alignment",
        "heading_reward",
        "action_penalty",
        "collision_reward",
        "goal_reward",
        "collision",
        "goal_reached",
    ]

    for key in required_keys:
        assert key in info, (
            f"Missing reward information: {key}"
        )

    print("Reward components    : PASS")

    env.close()


def test_heading_reward_math():
    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    # ---------------------------------------------------------
    # Test the mathematical behavior independently.
    # ---------------------------------------------------------

    # Farther than 1 m:
    # heading reward must be zero.
    distance = 2.0
    bearing = 0.0

    weight = np.clip(
        1.0 -
        distance / env.HEADING_DISTANCE,
        0.0,
        1.0,
    )

    reward = (
        env.HEADING_WEIGHT *
        weight *
        math.cos(bearing)
    )

    assert abs(reward) < 1e-12

    # Exactly 1 m:
    # heading reward must still be zero.
    distance = 1.0

    weight = np.clip(
        1.0 -
        distance / env.HEADING_DISTANCE,
        0.0,
        1.0,
    )

    reward = (
        env.HEADING_WEIGHT *
        weight *
        math.cos(0.0)
    )

    assert abs(reward) < 1e-12

    # 0.5 m and perfect alignment:
    # weight = 0.5
    # reward = 0.02 * 0.5 = 0.01
    distance = 0.5

    weight = np.clip(
        1.0 -
        distance / env.HEADING_DISTANCE,
        0.0,
        1.0,
    )

    reward = (
        env.HEADING_WEIGHT *
        weight *
        math.cos(0.0)
    )

    assert abs(
        reward - 0.01
    ) < 1e-9

    # 0.5 m and 90 degrees:
    # cos(90 deg) = 0
    distance = 0.5
    bearing = math.radians(90.0)

    weight = np.clip(
        1.0 -
        distance / env.HEADING_DISTANCE,
        0.0,
        1.0,
    )

    reward = (
        env.HEADING_WEIGHT *
        weight *
        math.cos(bearing)
    )

    assert abs(reward) < 1e-12

    # 0.5 m and 180 degrees:
    # cos(180 deg) = -1
    # reward = -0.01
    distance = 0.5
    bearing = math.radians(180.0)

    weight = np.clip(
        1.0 -
        distance / env.HEADING_DISTANCE,
        0.0,
        1.0,
    )

    reward = (
        env.HEADING_WEIGHT *
        weight *
        math.cos(bearing)
    )

    assert abs(
        reward + 0.01
    ) < 1e-9

    print("Heading reward math  : PASS")

    env.close()


def test_action_penalty():
    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    # For action [1, 1]:
    #
    # penalty =
    # 0.001 * (1^2 + 1^2)
    #
    # = 0.002

    action = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    expected_penalty = 0.002

    actual_penalty = (
        env.ACTION_PENALTY_WEIGHT *
        float(
            np.sum(
                np.square(action)
            )
        )
    )

    assert abs(
        actual_penalty -
        expected_penalty
    ) < 1e-9

    print("Action penalty       : PASS")

    env.close()


def test_multiple_steps():
    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    observation, info = env.reset(
        seed=42
    )

    for step in range(100):

        action = np.array(
            [0.5, 0.5],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

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

        assert np.isfinite(reward)

        if terminated or truncated:
            break

    print(
        "100-step stability   : PASS"
    )

    env.close()


def main():

    print()
    print(
        "========================================"
    )
    print(
        "V3.21.7 ENVIRONMENT REWARD TEST"
    )
    print(
        "========================================"
    )
    print()

    test_environment_creation()

    test_reset()

    test_reward_components()

    test_heading_reward_math()

    test_action_penalty()

    test_multiple_steps()

    print()
    print(
        "========================================"
    )
    print(
        "V3.21.7 ENVIRONMENT REWARD TEST: PASS"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()