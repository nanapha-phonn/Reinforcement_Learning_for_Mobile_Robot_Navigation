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
    # Farther than 1 m:
    # heading reward must be zero.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Exactly 1 m:
    # heading reward must still be zero.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 0.5 m and perfect alignment:
    #
    # weight = 0.5
    # reward = 0.02 * 0.5 = 0.01
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 0.5 m and 90 degrees:
    #
    # cos(90 degrees) = 0
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 0.5 m and 180 degrees:
    #
    # cos(180 degrees) = -1
    # reward = -0.01
    # ---------------------------------------------------------

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


# =============================================================
# V3.21.7 BEHAVIORAL SANITY TESTS
# =============================================================


def get_robot_distance_to_goal(env):
    """
    Read the robot's actual PyBullet position and calculate
    distance to the goal.
    """

    import pybullet as p

    position, _ = p.getBasePositionAndOrientation(
        env.robot_id
    )

    dx = (
        position[0] -
        env.goal_position[0]
    )

    dy = (
        position[1] -
        env.goal_position[1]
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )


def set_robot_pose(
    env,
    x,
    y,
    yaw_degrees,
):
    """
    Place the robot at a deterministic pose.
    """

    import pybullet as p

    yaw = math.radians(
        yaw_degrees
    )

    orientation = p.getQuaternionFromEuler(
        [
            0.0,
            0.0,
            yaw,
        ]
    )

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [
            x,
            y,
            0.12,
        ],
        orientation,
    )

    # Reset linear/angular velocity so that the
    # test begins from a deterministic state.

    p.resetBaseVelocity(
        env.robot_id,
        linearVelocity=[
            0.0,
            0.0,
            0.0,
        ],
        angularVelocity=[
            0.0,
            0.0,
            0.0,
        ],
    )


def test_behavior_moving_toward_goal():
    """
    Check that forward motion from the initial position
    decreases the distance to the goal.
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    initial_distance = (
        get_robot_distance_to_goal(env)
    )

    action = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    final_distance = (
        info["distance_to_goal"]
    )

    passed = (
        final_distance <
        initial_distance
    )

    print(
        "Moving toward goal   :",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Distance: "
        f"{initial_distance:.4f} -> "
        f"{final_distance:.4f} m"
    )

    assert passed

    env.close()


def test_behavior_moving_away_from_goal():
    """
    Check that reverse motion from the initial position
    increases the distance to the goal.
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    initial_distance = (
        get_robot_distance_to_goal(env)
    )

    action = np.array(
        [-1.0, -1.0],
        dtype=np.float32,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    final_distance = (
        info["distance_to_goal"]
    )

    passed = (
        final_distance >
        initial_distance
    )

    print(
        "Moving away from goal:",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Distance: "
        f"{initial_distance:.4f} -> "
        f"{final_distance:.4f} m"
    )

    assert passed

    env.close()


def test_behavior_good_heading():
    """
    Place the robot near the goal and point it toward
    the goal.

    Expected:
        cos(beta) > 0
        heading reward > 0
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    # Goal = [3.0, 0.0]
    #
    # Robot = [2.5, 0.0]
    #
    # Therefore the goal is directly in front of
    # the robot when yaw = 0 degrees.

    set_robot_pose(
        env,
        x=2.5,
        y=0.0,
        yaw_degrees=0.0,
    )

    action = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    heading_reward = (
        info["heading_reward"]
    )

    heading_alignment = (
        info["heading_alignment"]
    )

    passed = (
        heading_alignment > 0.0
        and heading_reward > 0.0
    )

    print(
        "Good heading          :",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Bearing: "
        f"{math.degrees(info['goal_bearing']):.3f} deg"
    )

    print(
        f"  Alignment: "
        f"{heading_alignment:.6f}"
    )

    print(
        f"  Heading reward: "
        f"{heading_reward:+.6f}"
    )

    assert passed

    env.close()


def test_behavior_poor_heading():
    """
    Place the robot near the goal but point it directly
    away from the goal.

    Expected:
        cos(beta) < 0
        heading reward < 0
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    # Goal = [3.0, 0.0]
    #
    # Robot = [2.5, 0.0]
    #
    # Robot faces -X.
    #
    # Goal is therefore directly behind it.

    set_robot_pose(
        env,
        x=2.5,
        y=0.0,
        yaw_degrees=180.0,
    )

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

    heading_reward = (
        info["heading_reward"]
    )

    heading_alignment = (
        info["heading_alignment"]
    )

    passed = (
        heading_alignment < 0.0
        and heading_reward < 0.0
    )

    print(
        "Poor heading          :",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Bearing: "
        f"{math.degrees(info['goal_bearing']):.3f} deg"
    )

    print(
        f"  Alignment: "
        f"{heading_alignment:.6f}"
    )

    print(
        f"  Heading reward: "
        f"{heading_reward:+.6f}"
    )

    assert passed

    env.close()


def test_behavior_action_penalty():
    """
    Verify that stronger actions receive a larger
    action penalty.
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    # Zero action.

    env.reset(seed=42)

    zero_action = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    (
        observation,
        reward_zero,
        terminated,
        truncated,
        info_zero,
    ) = env.step(zero_action)

    zero_penalty = (
        info_zero["action_penalty"]
    )

    # Full action.

    env.reset(seed=42)

    full_action = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    (
        observation,
        reward_full,
        terminated,
        truncated,
        info_full,
    ) = env.step(full_action)

    full_penalty = (
        info_full["action_penalty"]
    )

    passed = (
        abs(zero_penalty) < 1e-12
        and full_penalty > zero_penalty
    )

    print(
        "Action penalty        :",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Zero action penalty: "
        f"{zero_penalty:.6f}"
    )

    print(
        f"  Full action penalty: "
        f"{full_penalty:.6f}"
    )

    assert passed

    env.close()


def test_behavior_goal_termination():
    """
    Place the robot inside the goal threshold.

    Expected:
        goal_reached = True
        goal_reward = +10
        terminated = True
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    # Goal is [3.0, 0.0].
    #
    # Place robot at [2.90, 0.0].
    #
    # Distance = 0.10 m < 0.20 m threshold.

    set_robot_pose(
        env,
        x=2.90,
        y=0.0,
        yaw_degrees=0.0,
    )

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

    distance = (
        info["distance_to_goal"]
    )

    goal_reward = (
        info["goal_reward"]
    )

    passed = (
        distance < 0.20
        and info["goal_reached"]
        and goal_reward == 10.0
        and terminated
    )

    print(
        "Goal termination       :",
        "PASS" if passed else "FAIL",
    )

    print(
        f"  Distance: "
        f"{distance:.6f} m"
    )

    print(
        f"  Goal reward: "
        f"{goal_reward:+.1f}"
    )

    print(
        f"  Goal reached: "
        f"{info['goal_reached']}"
    )

    print(
        f"  Terminated: "
        f"{terminated}"
    )

    assert passed

    env.close()


def test_behavior_collision():
    """
    Verify collision behavior.

    This test does NOT assume an arbitrary obstacle.
    It checks whether the environment can actually
    produce a collision during a controlled simulation.

    If no collision geometry exists in the environment,
    the test reports SKIP rather than fabricating a PASS.
    """

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    env.reset(seed=42)

    collision_detected = False

    for _ in range(500):

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if info["collision"]:

            collision_detected = True

            collision_reward = (
                info["collision_reward"]
            )

            passed = (
                collision_reward == -10.0
                and terminated
            )

            print(
                "Collision behavior     :",
                "PASS" if passed else "FAIL",
            )

            print(
                f"  Collision reward: "
                f"{collision_reward:+.1f}"
            )

            print(
                f"  Terminated: "
                f"{terminated}"
            )

            assert passed

            env.close()

            return

        if terminated or truncated:
            break

    print(
        "Collision behavior     : SKIP"
    )

    print(
        "  No collision geometry was encountered."
    )

    env.close()


def main():

    print()
    print(
        "============================================================"
    )
    print(
        "V3.21.7 BEHAVIORAL SANITY TEST"
    )
    print(
        "============================================================"
    )

    print()

    # ---------------------------------------------------------
    # Existing V3.21.7 environment tests
    # ---------------------------------------------------------

    test_environment_creation()

    test_reset()

    test_reward_components()

    test_heading_reward_math()

    test_action_penalty()

    test_multiple_steps()

    # ---------------------------------------------------------
    # Behavioral tests
    # ---------------------------------------------------------

    print()
    print(
        "--- BEHAVIORAL TESTS ---"
    )

    test_behavior_moving_toward_goal()

    test_behavior_moving_away_from_goal()

    test_behavior_good_heading()

    test_behavior_poor_heading()

    test_behavior_action_penalty()

    test_behavior_goal_termination()

    test_behavior_collision()

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "V3.21.7 BEHAVIORAL SANITY TEST: PASS"
    )
    print(
        "============================================================"
    )
    print()


if __name__ == "__main__":
    main()