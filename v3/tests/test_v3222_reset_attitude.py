import os
import sys

import numpy as np
import pybullet as p

# Add project root to Python import path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from environment import DifferentialDriveEnv


# ============================================================
# V3.22.2 RESET / CONTACT ATTITUDE DIAGNOSTIC
# ============================================================

MAX_PHYSICS_STEPS = 1000

CHECKPOINTS = [
    0,
    1,
    10,
    100,
    500,
    1000,
]

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1

MOTOR_FORCE = 20.0


def get_attitude(env):
    """Read ground-truth base position and orientation."""

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        np.array(position, dtype=float),
        np.degrees(roll),
        np.degrees(pitch),
        np.degrees(yaw),
    )


def print_state(label, env):
    """Print robot physical state."""

    position, roll, pitch, yaw = get_attitude(env)

    print(
        f"{label:>12} | "
        f"X={position[0]:8.4f} | "
        f"Y={position[1]:8.4f} | "
        f"Z={position[2]:8.4f} | "
        f"Roll={roll:8.3f}° | "
        f"Pitch={pitch:8.3f}° | "
        f"Yaw={yaw:8.3f}°"
    )

    return position, roll, pitch, yaw


def hold_wheels_stopped(env):
    """
    Hold both drive wheels at zero velocity.

    This is a physics diagnostic only.
    PPO is not used.
    """

    p.setJointMotorControl2(
        bodyUniqueId=env.robot_id,
        jointIndex=LEFT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=MOTOR_FORCE,
    )

    p.setJointMotorControl2(
        bodyUniqueId=env.robot_id,
        jointIndex=RIGHT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=MOTOR_FORCE,
    )


def main():

    print("=" * 90)
    print("V3.22.2 RESET / CONTACT ATTITUDE DIAGNOSTIC")
    print("=" * 90)

    print()
    print("Purpose:")
    print(
        "Determine whether the large pitch angle exists "
        "without PPO actions."
    )

    print()
    print("PPO: DISABLED")
    print("Robot actions: ZERO WHEEL VELOCITY")
    print("Reward: NOT EVALUATED")
    print("Physics: UNCHANGED")

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        env.reset(seed=1)

        print()
        print("--- AFTER RESET ---")
        print()

        print(
            f"{'Checkpoint':>12} | "
            f"{'X':>8} | "
            f"{'Y':>8} | "
            f"{'Z':>8} | "
            f"{'Roll':>9} | "
            f"{'Pitch':>9} | "
            f"{'Yaw':>9}"
        )

        print("-" * 90)

        # ----------------------------------------------------
        # State immediately after reset
        # ----------------------------------------------------

        print_state(
            "RESET",
            env,
        )

        # ----------------------------------------------------
        # Track maximum attitude
        # ----------------------------------------------------

        max_abs_roll = 0.0
        max_abs_pitch = 0.0

        max_roll_step = 0
        max_pitch_step = 0

        # ----------------------------------------------------
        # Physics-only simulation
        # ----------------------------------------------------

        for physics_step in range(
            1,
            MAX_PHYSICS_STEPS + 1,
        ):

            # ------------------------------------------------
            # Stop both drive wheels.
            #
            # No PPO action is used.
            # No environment method is called.
            # ------------------------------------------------

            hold_wheels_stopped(env)

            # ------------------------------------------------
            # Advance physics
            # ------------------------------------------------

            p.stepSimulation()

            # ------------------------------------------------
            # Read attitude
            # ------------------------------------------------

            position, roll, pitch, yaw = get_attitude(
                env
            )

            # ------------------------------------------------
            # Track maximum roll
            # ------------------------------------------------

            if abs(roll) > max_abs_roll:

                max_abs_roll = abs(roll)
                max_roll_step = physics_step

            # ------------------------------------------------
            # Track maximum pitch
            # ------------------------------------------------

            if abs(pitch) > max_abs_pitch:

                max_abs_pitch = abs(pitch)
                max_pitch_step = physics_step

            # ------------------------------------------------
            # Checkpoints
            # ------------------------------------------------

            if physics_step in CHECKPOINTS:

                print_state(
                    f"STEP {physics_step}",
                    env,
                )

        # ====================================================
        # FINAL REPORT
        # ====================================================

        print()
        print("=" * 90)
        print("V3.22.2 SUMMARY")
        print("=" * 90)

        position, roll, pitch, yaw = get_attitude(
            env
        )

        print()

        print(
            f"Final X                 : "
            f"{position[0]:.6f} m"
        )

        print(
            f"Final Y                 : "
            f"{position[1]:.6f} m"
        )

        print(
            f"Final Z                 : "
            f"{position[2]:.6f} m"
        )

        print(
            f"Final roll              : "
            f"{roll:.6f} deg"
        )

        print(
            f"Final pitch             : "
            f"{pitch:.6f} deg"
        )

        print(
            f"Final yaw               : "
            f"{yaw:.6f} deg"
        )

        print()

        print(
            f"Maximum |roll|          : "
            f"{max_abs_roll:.6f} deg"
        )

        print(
            f"Maximum |pitch|         : "
            f"{max_abs_pitch:.6f} deg"
        )

        print(
            f"Maximum roll step       : "
            f"{max_roll_step}"
        )

        print(
            f"Maximum pitch step      : "
            f"{max_pitch_step}"
        )

        # ====================================================
        # DIAGNOSTIC INTERPRETATION
        # ====================================================

        print()
        print("--- DIAGNOSTIC RESULT ---")

        if abs(pitch) < 1.0:

            print(
                "RESET PHYSICS: PASS"
            )

            print(
                "The robot remains approximately level "
                "without PPO actions."
            )

            print(
                "The large PPO pitch must therefore be "
                "investigated under commanded wheel motion."
            )

        else:

            print(
                "RESET PHYSICS: FAIL / ATTITUDE OFFSET DETECTED"
            )

            print(
                "The robot develops a significant pitch "
                "without PPO actions."
            )

            print(
                "The problem is therefore associated with "
                "physical initialization, contact, or geometry."
            )

        print()
        print(
            "V3.22.2 RESET / CONTACT ATTITUDE "
            "DIAGNOSTIC COMPLETE"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()