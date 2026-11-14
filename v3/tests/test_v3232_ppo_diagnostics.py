
"""
V3.23.2 — PPO TRAINING DIAGNOSTICS

Purpose
-------
Analyze the already-trained V3.23.1 PPO baseline.

This script DOES NOT:
    - retrain PPO
    - modify the environment
    - modify the reward
    - modify robot.py
    - modify environment.py
    - tune hyperparameters

It analyzes:

    1. TensorBoard training statistics
    2. PPO model structure
    3. Policy action behavior
    4. Wheel-command utilization
    5. Deterministic trajectory behavior
    6. Observation validity
    7. Training stability indicators

Run from:

    C:\\python_projects\\rl_robot\\v3

Command:

    python tests\\test_v3232_ppo_diagnostics.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO


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
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v3231_ppo_baseline.zip"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
    / "v3231_ppo_baseline"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DIAGNOSTIC_EPISODES = 3

MAX_EPISODE_STEPS = 1000

MAX_ACTION_SAMPLES = 1000


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 80)
    print("LOADING V3.23.1 PPO BASELINE")
    print("=" * 80)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = PPO.load(
        str(MODEL_PATH),
        device="auto",
    )

    print()
    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Policy: {model.policy.__class__.__name__}"
    )

    print(
        f"Observation space: "
        f"{model.observation_space}"
    )

    print(
        f"Action space: "
        f"{model.action_space}"
    )

    print(
        f"Training timestep counter: "
        f"{model.num_timesteps:,}"
    )

    print(
        f"Device: {model.device}"
    )

    return model


# ============================================================
# MODEL STRUCTURE
# ============================================================

def inspect_model(model):

    print()
    print("=" * 80)
    print("MODEL STRUCTURE")
    print("=" * 80)

    print()
    print("Policy architecture:")
    print(model.policy)

    print()

    print(
        f"Policy parameters: "
        f"{sum(p.numel() for p in model.policy.parameters()):,}"
    )

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.policy.parameters() if p.requires_grad):,}"
    )

    print()

    print(
        "Expected:"
    )

    print(
        "  Observation = 48"
    )

    print(
        "  Action = 2"
    )

    print(
        "  Policy = MlpPolicy"
    )


# ============================================================
# TENSORBOARD DIAGNOSTICS
# ============================================================

def load_tensorboard_data():

    print()
    print("=" * 80)
    print("TENSORBOARD TRAINING DIAGNOSTICS")
    print("=" * 80)

    if not LOG_DIR.exists():

        print()
        print(
            "[WARNING] TensorBoard log directory "
            "does not exist:"
        )

        print(
            f"  {LOG_DIR}"
        )

        return None

    try:

        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator
        )

    except ImportError:

        print()
        print(
            "[WARNING] TensorBoard is not installed."
        )

        print(
            "Install with:"
        )

        print(
            "  pip install tensorboard"
        )

        return None

    event_files = list(
        LOG_DIR.rglob(
            "events.out.tfevents.*"
        )
    )

    if not event_files:

        print()
        print(
            "[WARNING] No TensorBoard event files found."
        )

        return None

    print()
    print(
        f"Found {len(event_files)} TensorBoard event file(s)."
    )

    # Use the newest event file.
    event_file = max(
        event_files,
        key=lambda path: path.stat().st_mtime,
    )

    print(
        f"Using:\n  {event_file}"
    )

    accumulator = EventAccumulator(
        str(event_file)
    )

    accumulator.Reload()

    scalar_tags = (
        accumulator.Tags()
        .get("scalars", [])
    )

    print()
    print("Available scalar metrics:")

    for tag in sorted(
        scalar_tags
    ):

        print(
            f"  {tag}"
        )

    return accumulator


# ============================================================
# PRINT METRIC
# ============================================================

def print_metric(
    accumulator,
    tag,
):

    if accumulator is None:

        return None

    if tag not in accumulator.Tags().get(
        "scalars",
        [],
    ):

        print(
            f"  {tag:<30} NOT AVAILABLE"
        )

        return None

    events = accumulator.Scalars(
        tag
    )

    if not events:

        print(
            f"  {tag:<30} EMPTY"
        )

        return None

    values = np.array(
        [
            event.value
            for event in events
        ],
        dtype=np.float64,
    )

    steps = np.array(
        [
            event.step
            for event in events
        ],
        dtype=np.int64,
    )

    last_value = values[-1]

    first_value = values[0]

    print(
        f"  {tag:<30} "
        f"first={first_value:.6f} "
        f"last={last_value:.6f} "
        f"min={np.min(values):.6f} "
        f"max={np.max(values):.6f}"
    )

    return {
        "steps": steps,
        "values": values,
    }


# ============================================================
# TRAINING METRICS
# ============================================================

def analyze_training_metrics(
    accumulator
):

    if accumulator is None:

        return

    print()
    print(
        "KEY PPO METRICS"
    )

    tags = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/policy_gradient_loss",
        "train/value_loss",
    ]

    metrics = {}

    for tag in tags:

        metrics[tag] = print_metric(
            accumulator,
            tag,
        )

    # --------------------------------------------------------
    # Explained variance interpretation
    # --------------------------------------------------------

    explained = metrics.get(
        "train/explained_variance"
    )

    if explained is not None:

        values = explained["values"]

        final = values[-1]

        print()
        print(
            "Explained variance interpretation:"
        )

        if final > 0.8:

            print(
                "  [GOOD] Value function explains "
                "most return variation."
            )

        elif final > 0.5:

            print(
                "  [INFO] Value function has learned "
                "a substantial amount."
            )

        elif final > 0.0:

            print(
                "  [WARNING] Value function learning "
                "is relatively weak."
            )

        else:

            print(
                "  [WARNING] Explained variance is "
                "non-positive."
            )


# ============================================================
# ENVIRONMENT
# ============================================================

def create_environment():

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    return env


# ============================================================
# ACTION STATISTICS
# ============================================================

def analyze_actions(
    model,
    env,
):

    print()
    print("=" * 80)
    print("POLICY ACTION DIAGNOSTICS")
    print("=" * 80)

    observation, _ = env.reset(
        seed=SEED
    )

    left_actions = []
    right_actions = []

    action_magnitudes = []

    left_right_difference = []

    observations_seen = 0

    terminated = False
    truncated = False

    while (
        not terminated
        and not truncated
        and observations_seen
        < MAX_ACTION_SAMPLES
    ):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        left = float(
            action[0]
        )

        right = float(
            action[1]
        )

        left_actions.append(
            left
        )

        right_actions.append(
            right
        )

        action_magnitudes.append(
            math.sqrt(
                left ** 2
                +
                right ** 2
            )
        )

        left_right_difference.append(
            abs(left - right)
        )

        observation, _, terminated, truncated, _ = (
            env.step(action)
        )

        observations_seen += 1

    left_actions = np.array(
        left_actions
    )

    right_actions = np.array(
        right_actions
    )

    action_magnitudes = np.array(
        action_magnitudes
    )

    left_right_difference = np.array(
        left_right_difference
    )

    print()

    print(
        f"Samples analyzed = "
        f"{len(left_actions)}"
    )

    print()

    print("LEFT wheel command:")
    print(
        f"  Mean   = {np.mean(left_actions):.6f}"
    )
    print(
        f"  Std    = {np.std(left_actions):.6f}"
    )
    print(
        f"  Min    = {np.min(left_actions):.6f}"
    )
    print(
        f"  Max    = {np.max(left_actions):.6f}"
    )

    print()

    print("RIGHT wheel command:")
    print(
        f"  Mean   = {np.mean(right_actions):.6f}"
    )
    print(
        f"  Std    = {np.std(right_actions):.6f}"
    )
    print(
        f"  Min    = {np.min(right_actions):.6f}"
    )
    print(
        f"  Max    = {np.max(right_actions):.6f}"
    )

    print()

    print("Action magnitude:")
    print(
        f"  Mean   = "
        f"{np.mean(action_magnitudes):.6f}"
    )

    print(
        f"  Max    = "
        f"{np.max(action_magnitudes):.6f}"
    )

    print()

    print("Left/right command difference:")
    print(
        f"  Mean   = "
        f"{np.mean(left_right_difference):.6f}"
    )

    print(
        f"  Max    = "
        f"{np.max(left_right_difference):.6f}"
    )

    print()

    # --------------------------------------------------------
    # Detect degenerate policy
    # --------------------------------------------------------

    left_std = np.std(
        left_actions
    )

    right_std = np.std(
        right_actions
    )

    if (
        left_std < 1e-4
        and right_std < 1e-4
    ):

        print(
            "[WARNING] Policy commands are "
            "nearly constant."
        )

        print(
            "This may indicate a very simple "
            "fixed trajectory policy."
        )

    else:

        print(
            "[PASS] Policy produces "
            "non-constant wheel commands."
        )

    if (
        np.mean(left_right_difference)
        < 0.01
    ):

        print(
            "[INFO] Left/right commands are "
            "mostly symmetric."
        )

        print(
            "This is expected for a straight "
            "fixed-goal trajectory."
        )

    else:

        print(
            "[PASS] Policy uses differential "
            "wheel commands."
        )


# ============================================================
# TRAJECTORY DIAGNOSTIC
# ============================================================

def trajectory_diagnostic(
    model,
    env,
):

    print()
    print("=" * 80)
    print("SINGLE TRAJECTORY DIAGNOSTIC")
    print("=" * 80)

    observation, _ = env.reset(
        seed=SEED
    )

    initial_position, _ = (
        _get_robot_pose(
            env
        )
    )

    initial_distance = _get_goal_distance(
        env
    )

    terminated = False
    truncated = False

    step = 0

    positions = []

    rewards = []

    while (
        not terminated
        and not truncated
        and step < MAX_EPISODE_STEPS
    ):

        position, orientation = (
            _get_robot_pose(
                env
            )
        )

        roll, pitch, yaw = (
            _orientation_to_euler(
                orientation
            )
        )

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        positions.append(
            (
                position[0],
                position[1],
            )
        )

        rewards.append(
            float(reward)
        )

        step += 1

    final_position, final_orientation = (
        _get_robot_pose(
            env
        )
    )

    final_distance = _get_goal_distance(
        env
    )

    final_roll, final_pitch, final_yaw = (
        _orientation_to_euler(
            final_orientation
        )
    )

    positions = np.array(
        positions,
        dtype=np.float64,
    )

    rewards = np.array(
        rewards,
        dtype=np.float64,
    )

    print()

    print(
        f"Initial distance = "
        f"{initial_distance:.6f} m"
    )

    print(
        f"Final distance   = "
        f"{final_distance:.6f} m"
    )

    print(
        f"Steps            = "
        f"{step}"
    )

    print()

    if len(positions) > 1:

        displacement = np.linalg.norm(
            positions[-1]
            -
            positions[0]
        )

        path_lengths = np.linalg.norm(
            np.diff(
                positions,
                axis=0,
            ),
            axis=1,
        )

        path_length = np.sum(
            path_lengths
        )

        print(
            f"Net displacement = "
            f"{displacement:.6f} m"
        )

        print(
            f"Path length      = "
            f"{path_length:.6f} m"
        )

        if displacement > 1e-9:

            path_efficiency = (
                displacement
                /
                path_length
            )

        else:

            path_efficiency = 0.0

        print(
            f"Path efficiency  = "
            f"{path_efficiency * 100:.3f}%"
        )

    print()

    print(
        f"Final roll       = "
        f"{math.degrees(final_roll):.6f}°"
    )

    print(
        f"Final pitch      = "
        f"{math.degrees(final_pitch):.6f}°"
    )

    print(
        f"Final yaw        = "
        f"{math.degrees(final_yaw):.6f}°"
    )

    print()

    if final_distance < 0.20:

        print(
            "[PASS] Goal reached."
        )

    else:

        print(
            "[FAIL] Goal not reached."
        )


# ============================================================
# MULTI-EPISODE CONSISTENCY
# ============================================================

def consistency_test(
    model,
):

    print()
    print("=" * 80)
    print("DETERMINISTIC POLICY CONSISTENCY")
    print("=" * 80)

    results = []

    for episode in range(
        DIAGNOSTIC_EPISODES
    ):

        env = create_environment()

        try:

            observation, _ = env.reset(
                seed=SEED
            )

            total_reward = 0.0

            steps = 0

            terminated = False
            truncated = False

            while (
                not terminated
                and not truncated
                and steps < MAX_EPISODE_STEPS
            ):

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += float(
                    reward
                )

                steps += 1

            distance = _get_goal_distance(
                env
            )

            results.append(
                (
                    total_reward,
                    steps,
                    distance,
                )
            )

        finally:

            env.close()

    print()

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"Episode {i}: "
            f"reward={result[0]:.6f}, "
            f"steps={result[1]}, "
            f"distance={result[2]:.6f} m"
        )

    rewards = np.array(
        [
            result[0]
            for result in results
        ]
    )

    steps = np.array(
        [
            result[1]
            for result in results
        ]
    )

    distances = np.array(
        [
            result[2]
            for result in results
        ]
    )

    print()

    print(
        f"Reward std     = "
        f"{np.std(rewards):.10f}"
    )

    print(
        f"Steps std      = "
        f"{np.std(steps):.10f}"
    )

    print(
        f"Distance std   = "
        f"{np.std(distances):.10f}"
    )

    if (
        np.std(rewards) < 1e-8
        and np.std(steps) < 1e-8
        and np.std(distances) < 1e-8
    ):

        print()

        print(
            "[INFO] Deterministic evaluation "
            "is perfectly repeatable."
        )

        print(
            "This is expected for identical "
            "start/goal conditions."
        )


# ============================================================
# PYBULLET HELPERS
# ============================================================

def _get_robot_pose(env):

    import pybullet as p

    return p.getBasePositionAndOrientation(
        env.robot_id
    )


def _orientation_to_euler(
    orientation
):

    import pybullet as p

    return p.getEulerFromQuaternion(
        orientation
    )


def _get_goal_distance(env):

    position, _ = _get_robot_pose(
        env
    )

    goal_x = env.goal_position[0]

    goal_y = env.goal_position[1]

    return math.sqrt(
        (position[0] - goal_x) ** 2
        +
        (position[1] - goal_y) ** 2
    )


# ============================================================
# OBSERVATION VALIDATION
# ============================================================

def observation_test():

    print()
    print("=" * 80)
    print("OBSERVATION VALIDATION")
    print("=" * 80)

    env = create_environment()

    try:

        observation, _ = env.reset(
            seed=SEED
        )

        observations = []

        for _ in range(100):

            action = env.action_space.sample()

            observation, _, terminated, truncated, _ = (
                env.step(action)
            )

            observations.append(
                observation
            )

            if terminated or truncated:

                observation, _ = env.reset(
                    seed=SEED
                )

        observations = np.asarray(
            observations,
            dtype=np.float64,
        )

    finally:

        env.close()

    print()

    print(
        f"Samples = "
        f"{len(observations)}"
    )

    print(
        f"Shape = "
        f"{observations.shape}"
    )

    print(
        f"Minimum = "
        f"{np.min(observations):.6f}"
    )

    print(
        f"Maximum = "
        f"{np.max(observations):.6f}"
    )

    print(
        f"NaN count = "
        f"{np.isnan(observations).sum()}"
    )

    print(
        f"Inf count = "
        f"{np.isinf(observations).sum()}"
    )

    if observations.shape[1] != 48:

        print(
            "[FAIL] Observation dimension "
            "is not 48."
        )

    elif not np.all(
        np.isfinite(observations)
    ):

        print(
            "[FAIL] Observation contains "
            "NaN/Inf."
        )

    elif (
        np.min(observations) < -1.00001
        or
        np.max(observations) > 1.00001
    ):

        print(
            "[FAIL] Observation outside "
            "[-1, 1]."
        )

    else:

        print(
            "[PASS] 48-dimensional "
            "normalized observation."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("V3.23.2 — PPO TRAINING DIAGNOSTICS")
    print("=" * 80)

    print()
    print(
        "Analyzing existing V3.23.1 model."
    )

    print(
        "NO RETRAINING."
    )

    print(
        "NO ENVIRONMENT MODIFICATIONS."
    )

    print(
        "NO REWARD MODIFICATIONS."
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Model structure
    # --------------------------------------------------------

    inspect_model(
        model
    )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    accumulator = load_tensorboard_data()

    analyze_training_metrics(
        accumulator
    )

    # --------------------------------------------------------
    # Observation validation
    # --------------------------------------------------------

    observation_test()

    # --------------------------------------------------------
    # Action behavior
    # --------------------------------------------------------

    env = create_environment()

    try:

        analyze_actions(
            model,
            env,
        )

        trajectory_diagnostic(
            model,
            env,
        )

    finally:

        env.close()

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    consistency_test(
        model
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("V3.23.2 DIAGNOSTICS COMPLETE")
    print("=" * 80)

    print()
    print(
        "No files were modified."
    )

    print(
        "No PPO training was performed."
    )

    print()
    print(
        "Next phase:"
    )

    print(
        "  V3.23.3 — Single-episode evaluation"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()