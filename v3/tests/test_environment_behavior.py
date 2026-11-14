import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import DifferentialDriveEnv


# =========================================================
# TEST UTILITIES
# =========================================================

def check(condition, message):

    if condition:
        print(f"{message}: PASS")
        return True

    print(f"{message}: FAIL")
    return False


def get_robot_pose(env):

    position, orientation = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return (
        np.array(position),
        np.array([roll, pitch, yaw]),
    )


# =========================================================
# 1. FORWARD MOTION
# =========================================================

def test_forward_motion():

    print()
    print("========================================")
    print("1. FORWARD MOTION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        initial_position, _ = (
            get_robot_pose(env)
        )

        print(
            f"Initial X : "
            f"{initial_position[0]:.6f} m"
        )

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        for _ in range(20):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if terminated or truncated:
                break

        final_position, final_orientation = (
            get_robot_pose(env)
        )

        displacement_x = (
            final_position[0] -
            initial_position[0]
        )

        displacement_y = (
            final_position[1] -
            initial_position[1]
        )

        print(
            f"Final X   : "
            f"{final_position[0]:.6f} m"
        )

        print(
            f"Final Y   : "
            f"{final_position[1]:.6f} m"
        )

        print(
            f"ΔX        : "
            f"{displacement_x:.6f} m"
        )

        print(
            f"ΔY        : "
            f"{displacement_y:.6f} m"
        )

        print(
            f"Collision : "
            f"{info['collision']}"
        )

        print(
            f"Terminated: "
            f"{terminated}"
        )

        forward_x_ok = check(
            displacement_x > 0.05,
            "Forward X displacement",
        )

        lateral_ok = check(
            abs(displacement_y) < 0.10,
            "Forward lateral stability",
        )

        collision_ok = check(
            info["collision"] is False,
            "Forward collision",
        )

        return (
            forward_x_ok
            and lateral_ok
            and collision_ok
        )

    finally:

        env.close()


# =========================================================
# 2. REVERSE MOTION
# =========================================================

def test_reverse_motion():

    print()
    print("========================================")
    print("2. REVERSE MOTION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        initial_position, _ = (
            get_robot_pose(env)
        )

        action = np.array(
            [-1.0, -1.0],
            dtype=np.float32,
        )

        for _ in range(20):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if terminated or truncated:
                break

        final_position, _ = (
            get_robot_pose(env)
        )

        displacement_x = (
            final_position[0] -
            initial_position[0]
        )

        print(
            f"Initial X : "
            f"{initial_position[0]:.6f} m"
        )

        print(
            f"Final X   : "
            f"{final_position[0]:.6f} m"
        )

        print(
            f"ΔX        : "
            f"{displacement_x:.6f} m"
        )

        print(
            f"Collision : "
            f"{info['collision']}"
        )

        reverse_ok = check(
            displacement_x < -0.05,
            "Reverse X displacement",
        )

        collision_ok = check(
            info["collision"] is False,
            "Reverse collision",
        )

        return (
            reverse_ok
            and collision_ok
        )

    finally:

        env.close()


# =========================================================
# 3. IN-PLACE ROTATION
# =========================================================

def test_rotation():

    print()
    print("========================================")
    print("3. ROTATION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        initial_position, initial_orientation = (
            get_robot_pose(env)
        )

        initial_yaw = initial_orientation[2]

        action = np.array(
            [-1.0, 1.0],
            dtype=np.float32,
        )

        for _ in range(20):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if terminated or truncated:
                break

        final_position, final_orientation = (
            get_robot_pose(env)
        )

        final_yaw = final_orientation[2]

        position_change = np.linalg.norm(
            final_position[:2] -
            initial_position[:2]
        )

        yaw_change = final_yaw - initial_yaw

        print(
            f"Initial yaw : "
            f"{np.degrees(initial_yaw):.6f} deg"
        )

        print(
            f"Final yaw   : "
            f"{np.degrees(final_yaw):.6f} deg"
        )

        print(
            f"Yaw change  : "
            f"{np.degrees(yaw_change):.6f} deg"
        )

        print(
            f"Position Δ  : "
            f"{position_change:.6f} m"
        )

        print(
            f"Collision    : "
            f"{info['collision']}"
        )

        rotation_ok = check(
            abs(yaw_change) > np.radians(20.0),
            "Yaw rotation",
        )

        position_ok = check(
            position_change < 0.20,
            "Rotation position stability",
        )

        collision_ok = check(
            info["collision"] is False,
            "Rotation collision",
        )

        return (
            rotation_ok
            and position_ok
            and collision_ok
        )

    finally:

        env.close()


# =========================================================
# 4. GOAL PROGRESS
# =========================================================

def test_goal_progress():

    print()
    print("========================================")
    print("4. GOAL PROGRESS TEST")
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

        print(
            f"Initial distance : "
            f"{initial_distance:.6f} m"
        )

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        distances = []

        for _ in range(10):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            distances.append(
                info["distance_to_goal"]
            )

            print(
                f"Step {info['step']:02d} | "
                f"Distance = "
                f"{info['distance_to_goal']:.6f} m | "
                f"Progress = "
                f"{info['progress']:.6f} | "
                f"Reward = "
                f"{reward:.6f}"
            )

            if terminated or truncated:
                break

        final_distance = distances[-1]

        print()
        print(
            f"Final distance   : "
            f"{final_distance:.6f} m"
        )

        progress_ok = check(
            final_distance < initial_distance,
            "Goal distance decreased",
        )

        reward_ok = check(
            info["progress"] > 0.0,
            "Positive progress",
        )

        collision_ok = check(
            info["collision"] is False,
            "Goal progress collision",
        )

        return (
            progress_ok
            and reward_ok
            and collision_ok
        )

    finally:

        env.close()


# =========================================================
# 5. GOAL TERMINATION
# =========================================================

def test_goal_termination():

    print()
    print("========================================")
    print("5. GOAL TERMINATION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        # Place the goal very close to the robot.
        env.goal_position = np.array(
            [0.10, 0.0],
            dtype=np.float32,
        )

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        goal_reached = False
        terminated = False

        for _ in range(10):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            print(
                f"Step {info['step']:02d} | "
                f"Distance = "
                f"{info['distance_to_goal']:.6f} m | "
                f"Goal = "
                f"{info['goal_reached']}"
            )

            if info["goal_reached"]:

                goal_reached = True
                break

        goal_ok = check(
            goal_reached,
            "Goal reached",
        )

        termination_ok = check(
            terminated,
            "Goal termination",
        )

        collision_ok = check(
            info["collision"] is False,
            "Goal termination without collision",
        )

        return (
            goal_ok
            and termination_ok
            and collision_ok
        )

    finally:

        env.close()


# =========================================================
# 6. COLLISION TERMINATION
# =========================================================

def test_collision():

    print()
    print("========================================")
    print("6. COLLISION TERMINATION TEST")
    print("========================================")
    print()

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        env.reset(seed=42)

        # -------------------------------------------------
        # Create a temporary obstacle.
        # -------------------------------------------------

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

        obstacle_visual_shape = (
            p.createVisualShape(
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
                baseVisualShapeIndex=(
                    obstacle_visual_shape
                ),
                basePosition=[
                    0.70,
                    0.0,
                    0.30,
                ],
            )
        )

        print(
            f"Obstacle ID : "
            f"{obstacle_id}"
        )

        action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        collision_detected = False
        terminated = False

        for _ in range(30):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            print(
                f"Step {info['step']:02d} | "
                f"Distance = "
                f"{info['distance_to_goal']:.6f} m | "
                f"Collision = "
                f"{info['collision']} | "
                f"Terminated = "
                f"{terminated}"
            )

            if info["collision"]:

                collision_detected = True
                break

        collision_ok = check(
            collision_detected,
            "Obstacle collision detected",
        )

        termination_ok = check(
            terminated,
            "Collision termination",
        )

        return (
            collision_ok
            and termination_ok
        )

    finally:

        env.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("========================================")
    print("V3.20 ENVIRONMENT BEHAVIOR VALIDATION")
    print("========================================")

    results = []

    results.append(
        test_forward_motion()
    )

    results.append(
        test_reverse_motion()
    )

    results.append(
        test_rotation()
    )

    results.append(
        test_goal_progress()
    )

    results.append(
        test_goal_termination()
    )

    results.append(
        test_collision()
    )

    print()
    print("========================================")
    print("FINAL VALIDATION")
    print("========================================")
    print()

    test_names = [
        "Forward motion",
        "Reverse motion",
        "Rotation",
        "Goal progress",
        "Goal termination",
        "Collision termination",
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
            f"{name:<25} : "
            f"{status}"
        )

    print()

    if all(results):

        print(
            "V3.20 ENVIRONMENT "
            "BEHAVIOR VALIDATION: PASS"
        )

    else:

        print(
            "V3.20 ENVIRONMENT "
            "BEHAVIOR VALIDATION: FAIL"
        )


if __name__ == "__main__":
    main()
