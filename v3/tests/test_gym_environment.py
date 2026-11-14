import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import DifferentialDriveEnv


def check(condition, message):

    if condition:
        print(f"{message}: PASS")
        return True

    print(f"{message}: FAIL")
    return False


def main():

    print()
    print("=== V3.21.7 GYMNASIUM ENVIRONMENT TEST ===")
    print()

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        # -----------------------------------------------------
        # Environment spaces
        # -----------------------------------------------------

        print("--- ENVIRONMENT SPACES ---")
        print()

        print(
            f"Action space      : "
            f"{env.action_space}"
        )

        print(
            f"Observation space : "
            f"{env.observation_space}"
        )

        print(
            f"Observation shape : "
            f"{env.observation_space.shape}"
        )

        print()

        action_space_ok = check(
            env.action_space.shape == (2,),
            "Action shape",
        )

        observation_space_ok = check(
            env.observation_space.shape == (48,),
            "Observation shape",
        )

        bounds_ok = check(
            np.all(
                env.action_space.low == -1.0
            )
            and
            np.all(
                env.action_space.high == 1.0
            ),
            "Action bounds",
        )

        # -----------------------------------------------------
        # Reset test
        # -----------------------------------------------------

        print()
        print("--- RESET TEST ---")
        print()

        observation, info = env.reset(
            seed=42
        )

        print(
            f"Observation length : "
            f"{len(observation)}"
        )

        print(
            f"Initial distance   : "
            f"{info['goal_x']:.3f} m goal X"
        )

        reset_shape_ok = check(
            observation.shape == (48,),
            "Reset observation shape",
        )

        reset_space_ok = check(
            env.observation_space.contains(
                observation
            ),
            "Reset observation space",
        )

        finite_ok = check(
            np.all(
                np.isfinite(observation)
            ),
            "Reset finite observation",
        )

        # -----------------------------------------------------
        # Zero-action step test
        # -----------------------------------------------------

        print()
        print("--- STEP TEST ---")
        print()

        zero_action = np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero_action
        )

        print(
            f"Reward             : "
            f"{reward:.6f}"
        )

        print(
            f"Terminated         : "
            f"{terminated}"
        )

        print(
            f"Truncated          : "
            f"{truncated}"
        )

        print(
            f"Distance to goal   : "
            f"{info['distance_to_goal']:.6f} m"
        )

        print(
            f"Collision          : "
            f"{info['collision']}"
        )

        step_shape_ok = check(
            next_observation.shape == (48,),
            "Step observation shape",
        )

        step_space_ok = check(
            env.observation_space.contains(
                next_observation
            ),
            "Step observation space",
        )

        zero_collision_ok = check(
            info["collision"] is False,
            "Zero-action collision",
        )

        zero_termination_ok = check(
            terminated is False,
            "Zero-action termination",
        )

        # -----------------------------------------------------
        # Forward action test
        # -----------------------------------------------------

        print()
        print("--- FORWARD ACTION TEST ---")
        print()

        forward_action = np.array(
            [1.0, 1.0],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            forward_action
        )

        print(
            f"Reward             : "
            f"{reward:.6f}"
        )

        print(
            f"Distance to goal   : "
            f"{info['distance_to_goal']:.6f} m"
        )

        print(
            f"Collision          : "
            f"{info['collision']}"
        )

        print(
            f"Terminated         : "
            f"{terminated}"
        )

        forward_shape_ok = check(
            observation.shape == (48,),
            "Forward observation shape",
        )

        forward_finite_ok = check(
            np.all(
                np.isfinite(observation)
            ),
            "Forward finite observation",
        )

        forward_collision_ok = check(
            info["collision"] is False,
            "Forward collision",
        )

        # -----------------------------------------------------
        # Gymnasium API check
        # -----------------------------------------------------

        print()
        print("--- GYMNASIUM API CHECK ---")
        print()

        api_ok = check(
            isinstance(
                reward,
                (float, np.floating),
            )
            and isinstance(
                terminated,
                bool,
            )
            and isinstance(
                truncated,
                bool,
            )
            and isinstance(
                info,
                dict,
            ),
            "Step return format",
        )

        # -----------------------------------------------------
        # Gymnasium check_env
        # -----------------------------------------------------

        print()
        print("--- GYMNASIUM CHECK_ENV ---")
        print()

        try:

            check_env(
                DifferentialDriveEnv(
                    use_noise=False,
                    render_mode="direct",
                    max_episode_steps=100,
                ),
                skip_render_check=True,
            )

            checker_ok = True

            print(
                "Gymnasium check_env: PASS"
            )

        except Exception as error:

            checker_ok = False

            print(
                "Gymnasium check_env: FAIL"
            )

            print()
            print(error)

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        all_passed = all([
            action_space_ok,
            observation_space_ok,
            bounds_ok,
            reset_shape_ok,
            reset_space_ok,
            finite_ok,
            step_shape_ok,
            step_space_ok,
            zero_collision_ok,
            zero_termination_ok,
            forward_shape_ok,
            forward_finite_ok,
            forward_collision_ok,
            api_ok,
            checker_ok,
        ])

        print()

        if all_passed:

            print(
                "V3.21.7 GYMNASIUM ENVIRONMENT TEST: PASS"
            )

        else:

            print(
                "V3.21.7 GYMNASIUM ENVIRONMENT TEST: FAIL"
            )

    finally:

        env.close()


if __name__ == "__main__":
    main()