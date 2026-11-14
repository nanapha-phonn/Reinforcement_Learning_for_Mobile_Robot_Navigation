import os
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from environment import DifferentialDriveEnv


# ============================================================
# V3.22.1 RL STABILITY / ATTITUDE DIAGNOSTIC
# ============================================================

MODEL_PATH = "models/ppo_v3217_heading_alignment_navigation.zip"

GOAL_POSITION = np.array(
    [3.0, 0.0],
    dtype=np.float32,
)

MAX_STEPS = 1000

SUCCESS_DISTANCE = 0.20

# Observation layout
GOAL_DISTANCE_INDEX = 46
GOAL_BEARING_INDEX = 47

# Print attitude every N steps
PRINT_INTERVAL = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_robot_state(env):
    """Read ground-truth robot position and orientation."""

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    robot_xy = np.array(
        position[:2],
        dtype=np.float32,
    )

    return robot_xy, yaw, roll, pitch


def calculate_goal_distance(robot_xy):
    """Calculate ground-truth distance to goal."""

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
    """Calculate relative goal bearing."""

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
# MAIN
# ============================================================

def main():

    print("=== V3.22.1 RL STABILITY / ATTITUDE DIAGNOSTIC ===")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nModel:")
    print(MODEL_PATH)

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
        # Reset
        # ----------------------------------------------------

        observation, info = env.reset(
            seed=1
        )

        robot_xy, yaw, roll, pitch = get_robot_state(
            env
        )

        initial_distance = calculate_goal_distance(
            robot_xy
        )

        # ----------------------------------------------------
        # Maximum attitude tracking
        # ----------------------------------------------------

        max_abs_roll = abs(
            np.degrees(roll)
        )

        max_abs_pitch = abs(
            np.degrees(pitch)
        )

        max_roll = np.degrees(roll)
        max_pitch = np.degrees(pitch)

        max_roll_step = 0
        max_pitch_step = 0

        # ----------------------------------------------------
        # Attitude trajectory storage
        # ----------------------------------------------------

        trajectory = []

        trajectory.append(
            {
                "step": 0,
                "x": float(robot_xy[0]),
                "y": float(robot_xy[1]),
                "yaw_deg": float(np.degrees(yaw)),
                "roll_deg": float(np.degrees(roll)),
                "pitch_deg": float(np.degrees(pitch)),
                "distance": float(initial_distance),
                "bearing_deg": float(
                    np.degrees(
                        calculate_goal_bearing(
                            robot_xy,
                            yaw,
                        )
                    )
                ),
            }
        )

        print("\n--- INITIAL STATE ---")

        print(
            f"Position       : "
            f"({robot_xy[0]:.3f}, {robot_xy[1]:.3f}) m"
        )

        print(
            f"Yaw            : "
            f"{np.degrees(yaw):.3f} deg"
        )

        print(
            f"Roll           : "
            f"{np.degrees(roll):.3f} deg"
        )

        print(
            f"Pitch          : "
            f"{np.degrees(pitch):.3f} deg"
        )

        print(
            f"Goal distance  : "
            f"{initial_distance:.3f} m"
        )

        # ----------------------------------------------------
        # Run trajectory
        # ----------------------------------------------------

        print("\n--- ATTITUDE TRAJECTORY ---")

        total_reward = 0.0

        terminated = False
        truncated = False

        for step in range(
            1,
            MAX_STEPS + 1,
        ):

            # ------------------------------------------------
            # PPO action
            # ------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            left_action = float(
                action[0]
            )

            right_action = float(
                action[1]
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += float(
                reward
            )

            # ------------------------------------------------
            # Read state
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

            roll_deg = np.degrees(
                roll
            )

            pitch_deg = np.degrees(
                pitch
            )

            yaw_deg = np.degrees(
                yaw
            )

            bearing_deg = np.degrees(
                bearing
            )

            # ------------------------------------------------
            # Update maximum attitude
            # ------------------------------------------------

            abs_roll = abs(
                roll_deg
            )

            abs_pitch = abs(
                pitch_deg
            )

            if abs_roll > max_abs_roll:

                max_abs_roll = abs_roll
                max_roll = roll_deg
                max_roll_step = step

            if abs_pitch > max_abs_pitch:

                max_abs_pitch = abs_pitch
                max_pitch = pitch_deg
                max_pitch_step = step

            # ------------------------------------------------
            # Store trajectory
            # ------------------------------------------------

            trajectory.append(
                {
                    "step": step,
                    "x": float(robot_xy[0]),
                    "y": float(robot_xy[1]),
                    "yaw_deg": float(yaw_deg),
                    "roll_deg": float(roll_deg),
                    "pitch_deg": float(pitch_deg),
                    "distance": float(distance),
                    "bearing_deg": float(bearing_deg),
                    "left_action": left_action,
                    "right_action": right_action,
                    "reward": float(reward),
                }
            )

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
                    f"Yaw={yaw_deg:7.2f}° | "
                    f"Roll={roll_deg:7.2f}° | "
                    f"Pitch={pitch_deg:7.2f}° | "
                    f"Dist={distance:7.3f} | "
                    f"Bear={bearing_deg:7.2f}° | "
                    f"L={left_action:6.3f} | "
                    f"R={right_action:6.3f} | "
                    f"Reward={reward:+7.4f}"
                )

            if terminated or truncated:

                break

        # ====================================================
        # FINAL STATE
        # ====================================================

        robot_xy, yaw, roll, pitch = get_robot_state(
            env
        )

        final_distance = calculate_goal_distance(
            robot_xy
        )

        final_bearing = calculate_goal_bearing(
            robot_xy,
            yaw
        )

        final_roll_deg = np.degrees(
            roll
        )

        final_pitch_deg = np.degrees(
            pitch
        )

        final_yaw_deg = np.degrees(
            yaw
        )

        # ====================================================
        # FINAL REPORT
        # ====================================================

        print()
        print("=" * 70)
        print("V3.22.1 STABILITY SUMMARY")
        print("=" * 70)

        print()

        print(
            f"Initial distance       : "
            f"{initial_distance:.3f} m"
        )

        print(
            f"Final distance         : "
            f"{final_distance:.3f} m"
        )

        print(
            f"Final bearing          : "
            f"{np.degrees(final_bearing):.2f} deg"
        )

        print()

        print(
            f"Final roll             : "
            f"{final_roll_deg:.2f} deg"
        )

        print(
            f"Final pitch            : "
            f"{final_pitch_deg:.2f} deg"
        )

        print()

        print(
            f"Maximum |roll|         : "
            f"{max_abs_roll:.2f} deg"
        )

        print(
            f"Maximum roll value     : "
            f"{max_roll:.2f} deg"
        )

        print(
            f"Maximum roll step      : "
            f"{max_roll_step}"
        )

        print()

        print(
            f"Maximum |pitch|        : "
            f"{max_abs_pitch:.2f} deg"
        )

        print(
            f"Maximum pitch value    : "
            f"{max_pitch:.2f} deg"
        )

        print(
            f"Maximum pitch step     : "
            f"{max_pitch_step}"
        )

        print()

        print(
            f"Total reward           : "
            f"{total_reward:.3f}"
        )

        print(
            f"Steps                  : "
            f"{step}"
        )

        print(
            f"Terminated             : "
            f"{terminated}"
        )

        print(
            f"Truncated              : "
            f"{truncated}"
        )

        # ====================================================
        # DIAGNOSTIC INTERPRETATION
        # ====================================================

        print()
        print("--- ATTITUDE DIAGNOSTIC ---")

        if max_abs_roll < 5.0:

            print(
                "Roll stability: WITHIN 5° diagnostic threshold"
            )

        else:

            print(
                "Roll stability: EXCEEDS 5° diagnostic threshold"
            )

        if max_abs_pitch < 5.0:

            print(
                "Pitch stability: WITHIN 5° diagnostic threshold"
            )

        else:

            print(
                "Pitch stability: EXCEEDS 5° diagnostic threshold"
            )

        if final_distance <= SUCCESS_DISTANCE:

            print(
                "Goal reaching: PASS"
            )

        else:

            print(
                "Goal reaching: FAIL"
            )

        print()
        print(
            "Stored trajectory samples: "
            f"{len(trajectory)}"
        )

        print()
        print(
            "V3.22.1 ATTITUDE DIAGNOSTIC COMPLETE"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()