from robot_env_v22 import RobotReachEnvV22
from stable_baselines3 import PPO
import numpy as np
import csv
import os


# ==========================================================
# EXPERIMENT SETTINGS
# ==========================================================

TRAINING_SEEDS = [0, 1, 2, 3, 4]

NUM_EVAL_EPISODES = 100

EVAL_SEEDS = list(range(100))

TOTAL_TIMESTEPS = 100_000


# ==========================================================
# RESULTS
# ==========================================================

results = []


# ==========================================================
# TRAIN + EVALUATE EACH SEED
# ==========================================================

for seed in TRAINING_SEEDS:

    print()
    print("=" * 60)
    print(f"TRAINING V2.2 — SEED {seed}")
    print("=" * 60)

    # ------------------------------------------------------
    # Create environment
    # ------------------------------------------------------

    env = RobotReachEnvV22()

    # ------------------------------------------------------
    # PPO
    # ------------------------------------------------------

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        learning_rate=0.0003,

        n_steps=2048,

        batch_size=64,

        gamma=0.99,

        gae_lambda=0.95,

        ent_coef=0.01,

        policy_kwargs=dict(
            net_arch=[64, 64]
        ),

        seed=seed
    )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS
    )

    model_path = (
        f"ppo_diff_drive_v22_seed_{seed}"
    )

    model.save(model_path)

    print()
    print(f"Model saved: {model_path}")

    env.close()

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print(f"EVALUATING V2.2 — SEED {seed}")
    print("=" * 60)

    eval_env = RobotReachEnvV22()

    successes = 0

    steps_list = []
    rewards_list = []
    final_distances = []

    for episode_seed in EVAL_SEEDS:

        obs, info = eval_env.reset(
            seed=episode_seed
        )

        terminated = False
        truncated = False

        total_reward = 0.0
        steps = 0

        while not terminated and not truncated:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                eval_env.step(action)
            )

            total_reward += reward

            steps += 1

        final_distance = float(obs[0])

        success = (
            final_distance
            < eval_env.success_radius
        )

        if success:
            successes += 1

        steps_list.append(steps)
        rewards_list.append(total_reward)
        final_distances.append(final_distance)

    eval_env.close()

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    success_rate = (
        successes
        / NUM_EVAL_EPISODES
        * 100
    )

    avg_steps = np.mean(
        steps_list
    )

    avg_reward = np.mean(
        rewards_list
    )

    avg_distance = np.mean(
        final_distances
    )

    min_distance = np.min(
        final_distances
    )

    max_distance = np.max(
        final_distances
    )

    # ------------------------------------------------------
    # Store result
    # ------------------------------------------------------

    result = {
        "training_seed": seed,
        "successes": successes,
        "episodes": NUM_EVAL_EPISODES,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "avg_final_distance": avg_distance,
        "min_final_distance": min_distance,
        "max_final_distance": max_distance
    }

    results.append(result)

    print()
    print(
        f"Seed {seed}: "
        f"{successes}/{NUM_EVAL_EPISODES} "
        f"({success_rate:.1f}%)"
    )

    print(
        f"Average steps: "
        f"{avg_steps:.1f}"
    )

    print(
        f"Average reward: "
        f"{avg_reward:.2f}"
    )

    print(
        f"Average final distance: "
        f"{avg_distance:.3f} m"
    )


# ==========================================================
# FINAL TABLE
# ==========================================================

print()
print()
print("=" * 90)
print("V2.2 REPRODUCIBILITY RESULTS")
print("=" * 90)

print(
    f"{'Seed':>6} "
    f"{'Success':>12} "
    f"{'Rate':>10} "
    f"{'Avg Steps':>12} "
    f"{'Avg Reward':>12} "
    f"{'Avg Dist.':>12}"
)

print("-" * 90)

for result in results:

    print(
        f"{result['training_seed']:>6} "
        f"{result['successes']:>5}/100 "
        f"{result['success_rate']:>9.1f}% "
        f"{result['avg_steps']:>12.1f} "
        f"{result['avg_reward']:>12.2f} "
        f"{result['avg_final_distance']:>12.3f}"
    )


# ==========================================================
# OVERALL STATISTICS
# ==========================================================

success_rates = [
    r["success_rate"]
    for r in results
]

avg_steps_all = [
    r["avg_steps"]
    for r in results
]

avg_rewards_all = [
    r["avg_reward"]
    for r in results
]

avg_distances_all = [
    r["avg_final_distance"]
    for r in results
]


print("-" * 90)

print(
    f"{'MEAN':>6} "
    f"{'':>12} "
    f"{np.mean(success_rates):>9.1f}% "
    f"{np.mean(avg_steps_all):>12.1f} "
    f"{np.mean(avg_rewards_all):>12.2f} "
    f"{np.mean(avg_distances_all):>12.3f}"
)

print(
    f"{'STD':>6} "
    f"{'':>12} "
    f"{np.std(success_rates):>9.1f}% "
    f"{np.std(avg_steps_all):>12.1f} "
    f"{np.std(avg_rewards_all):>12.2f} "
    f"{np.std(avg_distances_all):>12.3f}"
)


# ==========================================================
# SAVE CSV
# ==========================================================

csv_filename = (
    "v22_reproducibility_results.csv"
)

with open(
    csv_filename,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "training_seed",
            "successes",
            "episodes",
            "success_rate",
            "avg_steps",
            "avg_reward",
            "avg_final_distance",
            "min_final_distance",
            "max_final_distance"
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


print()
print(
    f"Results saved to: "
    f"{csv_filename}"
)