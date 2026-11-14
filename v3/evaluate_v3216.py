import os

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from environment import DifferentialDriveEnv


# ============================================================
# V3.21.6 PPO TRAJECTORY DIAGNOSTIC
# ============================================================

MODEL_PATH = "models/ppo_v3216_heading_navigation"

GOAL_POSITION = np.array(
    [3.0, 0.0],
    dtype=np.float32,
)

MAX_STEPS = 1000

# V3.21.4 / V3.21.5 observation layout
GOAL_DISTANCE_INDEX = 46
GOAL_BEARING_INDEX = 47


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_robot_state(env):
    """Read robot ground-truth position and orientation."""

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        np.array(position[:2], dtype=np.float32),
        yaw,
        roll,
        pitch,
    )


def calculate_goal_distance(robot_xy):
    """Calculate ground-truth distance to the goal."""

    return float(
        np.linalg.norm(
            robot_xy - GOAL_POSITION
        )
    )


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""

    return np.arctan2(
        np.sin(angle),
        np.cos(angle),
    )


def calculate_goal_bearing(robot_xy, yaw):
    """Calculate ground-truth relative goal bearing."""

    dx = GOAL_POSITION[0] - robot_xy[0]
    dy = GOAL_POSITION[1] - robot_xy[1]

    world_goal_angle = np.arctan2(
        dy,
        dx,
    )

    relative_bearing = wrap_angle(
        world_goal_angle - yaw
    )

    return float(relative_bearing)


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():

    print("=== V3.21.6 PPO TRAJECTORY DIAGNOSTIC ===")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    model_file = MODEL_PATH + ".zip"

    if not os.path.exists(model_file):

        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    print("\nModel:")
    print(model_file)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = PPO.load(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=MAX_STEPS,
    )

    try:

        # ----------------------------------------------------
        # Environment check
        # ----------------------------------------------------

        print("\n--- ENVIRONMENT ---")

        print(
            "Action space      :",
            env.action_space,
        )

        print(
            "Observation space :",
            env.observation_space,
        )

        print(
            "Observation shape :",
            env.observation_space.shape,
        )

        if env.observation_space.shape != (48,):

            raise RuntimeError(
                "Expected 48-D observation, "
                f"got {env.observation_space.shape}"
            )

        print("48-D observation: PASS")

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        observation, info = env.reset(
            seed=1
        )

        print("\n--- INITIAL STATE ---")

        robot_xy, yaw, roll, pitch = get_robot_state(
            env
        )

        ground_truth_distance = calculate_goal_distance(
            robot_xy
        )

        ground_truth_bearing = calculate_goal_bearing(
            robot_xy,
            yaw,
        )

        print(
            f"Robot position       : "
            f"({robot_xy[0]:.3f}, {robot_xy[1]:.3f}) m"
        )

        print(
            f"Robot yaw            : "
            f"{np.degrees(yaw):.3f} deg"
        )

        print(
            f"Goal position        : "
            f"({GOAL_POSITION[0]:.3f}, "
            f"{GOAL_POSITION[1]:.3f}) m"
        )

        print(
            f"Ground-truth distance: "
            f"{ground_truth_distance:.3f} m"
        )

        print(
            f"Ground-truth bearing : "
            f"{np.degrees(ground_truth_bearing):.3f} deg"
        )

        print(
            f"Observed distance    : "
            f"{observation[GOAL_DISTANCE_INDEX]:.6f}"
        )

        print(
            f"Observed bearing     : "
            f"{observation[GOAL_BEARING_INDEX]:.6f}"
        )

        # ----------------------------------------------------
        # Run trajectory
        # ----------------------------------------------------

        print("\n--- TRAJECTORY ---")

        total_reward = 0.0

        previous_distance = ground_truth_distance

        terminated = False
        truncated = False

        # Print every N steps to keep output manageable.
        PRINT_INTERVAL = 10

        for step in range(1, MAX_STEPS + 1):

            # ------------------------------------------------
            # Policy action
            # ------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            left_action = float(action[0])
            right_action = float(action[1])

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            # ------------------------------------------------
            # Ground-truth robot state
            # ------------------------------------------------

            robot_xy, yaw, roll, pitch = get_robot_state(
                env
            )

            distance = calculate_goal_distance(
                robot_xy
            )

            bearing = calculate_goal_bearing(
                robot_xy,
                yaw,
            )

            # ------------------------------------------------
            # Distance change
            # ------------------------------------------------

            distance_change = (
                previous_distance - distance
            )

            previous_distance = distance

            # ------------------------------------------------
            # Print trajectory
            # ------------------------------------------------

            if (
                step == 1
                or step % PRINT_INTERVAL == 0
                or terminated
                or truncated
            ):

                print(
                    f"Step {step:04d} | "
                    f"X={robot_xy[0]:7.3f} | "
                    f"Y={robot_xy[1]:7.3f} | "
                    f"Yaw={np.degrees(yaw):7.2f}° | "
                    f"Dist={distance:7.3f} | "
                    f"Bear={np.degrees(bearing):7.2f}° | "
                    f"L={left_action:6.3f} | "
                    f"R={right_action:6.3f} | "
                    f"dD={distance_change:+7.4f} | "
                    f"Reward={reward:+7.4f}"
                )

            # ------------------------------------------------
            # Stop if episode ended
            # ------------------------------------------------

            if terminated or truncated:
                break

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        print("\n--- FINAL STATE ---")

        robot_xy, yaw, roll, pitch = get_robot_state(
            env
        )

        final_distance = calculate_goal_distance(
            robot_xy
        )

        final_bearing = calculate_goal_bearing(
            robot_xy,
            yaw,
        )

        print(
            f"Final position       : "
            f"({robot_xy[0]:.3f}, {robot_xy[1]:.3f}) m"
        )

        print(
            f"Final yaw            : "
            f"{np.degrees(yaw):.3f} deg"
        )

        print(
            f"Final distance       : "
            f"{final_distance:.3f} m"
        )

        print(
            f"Final bearing        : "
            f"{np.degrees(final_bearing):.3f} deg"
        )

        print(
            f"Final roll           : "
            f"{np.degrees(roll):.3f} deg"
        )

        print(
            f"Final pitch          : "
            f"{np.degrees(pitch):.3f} deg"
        )

        print(
            f"Total reward         : "
            f"{total_reward:.3f}"
        )

        print(
            f"Steps                : "
            f"{step}"
        )

        print(
            f"Terminated           : "
            f"{terminated}"
        )

        print(
            f"Truncated            : "
            f"{truncated}"
        )

        # ----------------------------------------------------
        # Diagnosis
        # ----------------------------------------------------

        print("\n--- DIAGNOSTIC ---")

        if final_distance < ground_truth_distance:

            print(
                "Goal distance: IMPROVED"
            )

        else:

            print(
                "Goal distance: WORSENED"
            )

        if abs(final_bearing) < np.pi / 6:

            print(
                "Final bearing: roughly aligned with goal"
            )

        else:

            print(
                "Final bearing: NOT aligned with goal"
            )

        if terminated:

            print(
                "Episode ended by TERMINATION."
            )

        elif truncated:

            print(
                "Episode ended by TIME LIMIT."
            )

        print(
            "\nV3.21.6 TRAJECTORY DIAGNOSTIC COMPLETE"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()