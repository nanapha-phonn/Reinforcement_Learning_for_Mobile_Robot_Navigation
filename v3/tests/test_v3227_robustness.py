"""
V3.22.7 — BASIC ROBUSTNESS VALIDATION

Purpose
-------
Validate the frozen V3 Gymnasium environment under:

1. Sensor noise
2. Multiple random seeds
3. Repeated resets
4. Random actions
5. Observation validity
6. Episode termination / reset recovery

This test does NOT modify environment.py.

Run from:

C:\python_projects\rl_robot\v3

Command:

    python tests\test_v3227_robustness.py
"""

import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT ENVIRONMENT
# ============================================================

from environment import DifferentialDriveEnv


# ============================================================
# TEST PARAMETERS
# ============================================================

NUM_SEEDS = 5
RESETS_PER_SEED = 3
STEPS_PER_EPISODE = 100

OBSERVATION_SIZE = 48

LOWER_BOUND = -1.00001
UPPER_BOUND = 1.00001


# ============================================================
# OBSERVATION VALIDATION
# ============================================================

def validate_observation(
    observation,
    label,
):

    # --------------------------------------------------------
    # Type / shape
    # --------------------------------------------------------

    if not isinstance(
        observation,
        np.ndarray,
    ):

        print(
            f"[FAIL] {label}: "
            f"observation is not numpy array."
        )

        return False

    if observation.shape != (
        OBSERVATION_SIZE,
    ):

        print(
            f"[FAIL] {label}: "
            f"wrong observation shape "
            f"{observation.shape}."
        )

        return False

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(observation)
    ):

        print(
            f"[FAIL] {label}: "
            f"NaN or Inf detected."
        )

        return False

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    if np.any(
        observation < LOWER_BOUND
    ):

        print(
            f"[FAIL] {label}: "
            f"value below -1."
        )

        return False

    if np.any(
        observation > UPPER_BOUND
    ):

        print(
            f"[FAIL] {label}: "
            f"value above +1."
        )

        return False

    return True


# ============================================================
# SINGLE SEED TEST
# ============================================================

def run_seed_test(
    env,
    seed,
):

    print()
    print("-" * 80)
    print(
        f"SEED {seed}"
    )
    print("-" * 80)

    rng = np.random.default_rng(
        seed
    )

    total_steps = 0
    terminated_count = 0
    truncated_count = 0

    # --------------------------------------------------------
    # REPEATED RESETS
    # --------------------------------------------------------

    for reset_index in range(
        RESETS_PER_SEED
    ):

        print(
            f"\nReset "
            f"{reset_index + 1}/"
            f"{RESETS_PER_SEED}"
        )

        try:

            observation, info = (
                env.reset(
                    seed=seed
                )
            )

        except Exception as exc:

            print(
                "[FAIL] Reset exception:"
            )

            print(
                repr(exc)
            )

            return False

        # ----------------------------------------------------
        # Validate initial observation
        # ----------------------------------------------------

        if not validate_observation(
            observation,
            f"seed={seed} "
            f"reset={reset_index + 1} "
            f"initial",
        ):

            return False

        print(
            "[PASS] Initial observation"
        )

        # ----------------------------------------------------
        # Random action episode
        # ----------------------------------------------------

        for step in range(
            STEPS_PER_EPISODE
        ):

            action = rng.uniform(
                low=-1.0,
                high=1.0,
                size=(2,),
            ).astype(
                np.float32
            )

            try:

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

            except Exception as exc:

                print()
                print(
                    "[FAIL] Step exception:"
                )

                print(
                    f"Seed       : {seed}"
                )

                print(
                    f"Reset      : "
                    f"{reset_index + 1}"
                )

                print(
                    f"Step       : "
                    f"{step + 1}"
                )

                print(
                    f"Action     : "
                    f"{action}"
                )

                print(
                    repr(exc)
                )

                return False

            total_steps += 1

            # ------------------------------------------------
            # Observation
            # ------------------------------------------------

            if not validate_observation(
                observation,
                f"seed={seed} "
                f"reset={reset_index + 1} "
                f"step={step + 1}",
            ):

                return False

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            if not np.isfinite(
                reward
            ):

                print(
                    "[FAIL] Non-finite reward."
                )

                print(
                    f"Reward = {reward}"
                )

                return False

            # ------------------------------------------------
            # Termination
            # ------------------------------------------------

            if terminated:

                terminated_count += 1

                print(
                    f"[INFO] Episode terminated "
                    f"at step {step + 1}"
                )

                break

            if truncated:

                truncated_count += 1

                print(
                    f"[INFO] Episode truncated "
                    f"at step {step + 1}"
                )

                break

        else:

            print(
                "[PASS] Random-action run completed "
                f"{STEPS_PER_EPISODE} steps."
            )

    # --------------------------------------------------------
    # Seed summary
    # --------------------------------------------------------

    print()
    print(
        f"[PASS] Seed {seed} complete"
    )

    print(
        f"  Total steps       : {total_steps}"
    )

    print(
        f"  Terminated       : {terminated_count}"
    )

    print(
        f"  Truncated        : {truncated_count}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "V3.22.7 — BASIC ROBUSTNESS VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Environment: DifferentialDriveEnv"
    )

    print(
        "Sensor noise: ENABLED"
    )

    print(
        f"Seeds: {NUM_SEEDS}"
    )

    print(
        f"Resets per seed: {RESETS_PER_SEED}"
    )

    print(
        f"Maximum steps per reset: "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        "PPO training: DISABLED"
    )

    print(
        "Production environment modified: NO"
    )

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    results = []

    try:

        # ----------------------------------------------------
        # TEST EACH SEED
        # ----------------------------------------------------

        for seed in range(
            1,
            NUM_SEEDS + 1,
        ):

            passed = run_seed_test(
                env,
                seed,
            )

            results.append(
                (
                    seed,
                    passed,
                )
            )

            if not passed:

                print()
                print(
                    f"[FAIL] Seed {seed}"
                )

                # Stop immediately.
                # There is no value in hiding the
                # first robustness failure.

                break

    finally:

        env.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "V3.22.7 SUMMARY"
    )
    print("=" * 80)

    passed_count = sum(
        1
        for _, passed in results
        if passed
    )

    failed_count = sum(
        1
        for _, passed in results
        if not passed
    )

    print()

    for seed, passed in results:

        if passed:

            print(
                f"[PASS] Seed {seed}"
            )

        else:

            print(
                f"[FAIL] Seed {seed}"
            )

    print()
    print(
        f"Seeds passed: "
        f"{passed_count}/{len(results)}"
    )

    print(
        f"Seeds failed: "
        f"{failed_count}/{len(results)}"
    )

    print()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if (
        len(results) == NUM_SEEDS
        and failed_count == 0
    ):

        print(
            "[PASS] V3.22.7 ROBUSTNESS COMPLETE"
        )

        print()
        print(
            "Validated:"
        )

        print(
            "  - sensor noise"
        )

        print(
            "  - multiple random seeds"
        )

        print(
            "  - repeated environment resets"
        )

        print(
            "  - random actions"
        )

        print(
            "  - finite observations"
        )

        print(
            "  - observation normalization"
        )

        print(
            "  - finite rewards"
        )

        print(
            "  - termination / reset recovery"
        )

        print()
        print(
            "Ready for V3.22.8 final regression."
        )

    else:

        print(
            "[FAIL] V3.22.7 requires investigation."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":

    main()