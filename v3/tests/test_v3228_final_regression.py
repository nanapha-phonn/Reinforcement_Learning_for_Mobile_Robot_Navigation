"""
V3.22.8 — FINAL REGRESSION VALIDATION

Purpose
-------
Final production regression test before V3.23 PPO.

Validates:

1. Static stability
2. Forward motion
3. Rotation
4. Curved motion
5. Sensor noise
6. Goal termination
7. Collision detection
8. Known reverse-motion limitation

Important
---------
The reverse-motion caster problem was already identified in
V3.22.5q. It is documented here as a known limitation and
does NOT block V3.23.

This test does NOT modify:

    robot.py
    environment.py

Run from:

    C:\python_projects\rl_robot\v3

Command:

    python tests\test_v3228_final_regression.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pybullet as p


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENVIRONMENT
# ============================================================

from environment import DifferentialDriveEnv


# ============================================================
# CONSTANTS
# ============================================================

CHASSIS_Z = 0.13

TIME_STEP = 1.0 / 240.0

PHYSICS_STEPS = 1000

MAX_STATIC_ROLL = 1.0
MAX_STATIC_PITCH = 1.0

MAX_FORWARD_ROLL = 2.0
MAX_FORWARD_PITCH = 2.0

MAX_ROTATION_ROLL = 5.0
MAX_ROTATION_PITCH = 5.0

MAX_CURVE_ROLL = 5.0
MAX_CURVE_PITCH = 5.0

EXPECTED_WHEEL_RADIUS = 0.08
EXPECTED_WHEEL_BASE = 0.30


# ============================================================
# ANGLE HELPERS
# ============================================================

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


# ============================================================
# POSE
# ============================================================

def get_pose(env):

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
        position[0],
        position[1],
        position[2],
        roll,
        pitch,
        yaw,
    )


# ============================================================
# RESET
# ============================================================

def reset_robot(env):

    observation, info = env.reset(
        seed=42
    )

    return observation


# ============================================================
# OBSERVATION VALIDATION
# ============================================================

def validate_observation(
    observation,
    label,
):

    if not isinstance(
        observation,
        np.ndarray,
    ):

        print(
            f"[FAIL] {label}: "
            "observation is not numpy.ndarray."
        )

        return False

    if observation.shape != (48,):

        print(
            f"[FAIL] {label}: "
            f"shape = {observation.shape}"
        )

        return False

    if not np.all(
        np.isfinite(observation)
    ):

        print(
            f"[FAIL] {label}: "
            "NaN/Inf detected."
        )

        return False

    if np.any(
        observation < -1.00001
    ):

        print(
            f"[FAIL] {label}: "
            "observation below -1."
        )

        return False

    if np.any(
        observation > 1.00001
    ):

        print(
            f"[FAIL] {label}: "
            "observation above +1."
        )

        return False

    return True


# ============================================================
# TEST 1 — STATIC STABILITY
# ============================================================

def test_static(env):

    print()
    print("=" * 80)
    print("TEST 1 — STATIC STABILITY")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "static initial observation",
    ):

        return False

    max_roll = 0.0
    max_pitch = 0.0

    for _ in range(
        PHYSICS_STEPS
    ):

        p.stepSimulation()

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_pose(env)

        max_roll = max(
            max_roll,
            abs(math.degrees(roll)),
        )

        max_pitch = max(
            max_pitch,
            abs(math.degrees(pitch)),
        )

    print(
        f"Final position = "
        f"({x:.6f}, {y:.6f}, {z:.6f})"
    )

    print(
        f"Max |roll|  = "
        f"{max_roll:.6f}°"
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch:.6f}°"
    )

    if abs(z - CHASSIS_Z) > 0.01:

        print(
            "[FAIL] Unexpected chassis Z position."
        )

        return False

    if max_roll > MAX_STATIC_ROLL:

        print(
            "[FAIL] Static roll instability."
        )

        return False

    if max_pitch > MAX_STATIC_PITCH:

        print(
            "[FAIL] Static pitch instability."
        )

        return False

    print(
        "[PASS] Static stability."
    )

    return True


# ============================================================
# TEST 2 — FORWARD MOTION
# ============================================================

def test_forward(env):

    print()
    print("=" * 80)
    print("TEST 2 — FORWARD MOTION")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "forward initial observation",
    ):

        return False

    action = np.array(
        [1.0, 1.0],
        dtype=np.float32,
    )

    start_x, start_y, _, _, _, _ = (
        get_pose(env)
    )

    max_roll = 0.0
    max_pitch = 0.0

    for _ in range(
        PHYSICS_STEPS // 12
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        if not validate_observation(
            observation,
            "forward observation",
        ):

            return False

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_pose(env)

        max_roll = max(
            max_roll,
            abs(math.degrees(roll)),
        )

        max_pitch = max(
            max_pitch,
            abs(math.degrees(pitch)),
        )

        if terminated or truncated:
            break

    displacement = math.sqrt(
        (x - start_x) ** 2
        +
        (y - start_y) ** 2
    )

    print(
        f"Final position = "
        f"({x:.6f}, {y:.6f}, {z:.6f})"
    )

    print(
        f"Displacement = "
        f"{displacement:.6f} m"
    )

    print(
        f"Max |roll| = "
        f"{max_roll:.6f}°"
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch:.6f}°"
    )

    if displacement < 0.10:

        print(
            "[FAIL] Robot did not move forward."
        )

        return False

    if max_roll > MAX_FORWARD_ROLL:

        print(
            "[FAIL] Excessive forward roll."
        )

        return False

    if max_pitch > MAX_FORWARD_PITCH:

        print(
            "[FAIL] Excessive forward pitch."
        )

        return False

    print(
        "[PASS] Forward motion."
    )

    return True


# ============================================================
# TEST 3 — ROTATION
# ============================================================

def test_rotation(env):

    print()
    print("=" * 80)
    print("TEST 3 — ROTATION")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "rotation initial observation",
    ):

        return False

    action = np.array(
        [-1.0, 1.0],
        dtype=np.float32,
    )

    _, _, _, _, _, start_yaw = (
        get_pose(env)
    )

    max_roll = 0.0
    max_pitch = 0.0

    for _ in range(
        PHYSICS_STEPS // 12
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        if not validate_observation(
            observation,
            "rotation observation",
        ):

            return False

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_pose(env)

        max_roll = max(
            max_roll,
            abs(math.degrees(roll)),
        )

        max_pitch = max(
            max_pitch,
            abs(math.degrees(pitch)),
        )

        if terminated or truncated:
            break

    yaw_change = abs(
        wrap_angle(
            yaw - start_yaw
        )
    )

    wheel_velocity = 4.0

    v_left = (
        -wheel_velocity
        * EXPECTED_WHEEL_RADIUS
    )

    v_right = (
        wheel_velocity
        * EXPECTED_WHEEL_RADIUS
    )

    theoretical_yaw_rate = (
        v_right - v_left
    ) / EXPECTED_WHEEL_BASE

    test_duration = (
        (PHYSICS_STEPS // 12)
        * 12
        * TIME_STEP
    )

    theoretical_yaw = (
        theoretical_yaw_rate
        * test_duration
    )

    print(
        f"Observed |yaw change| = "
        f"{math.degrees(yaw_change):.6f}°"
    )

    print(
        f"Theoretical yaw rate = "
        f"{theoretical_yaw_rate:.6f} rad/s"
    )

    print(
        f"Theoretical yaw over test = "
        f"{math.degrees(theoretical_yaw):.6f}°"
    )

    print(
        f"Max |roll| = "
        f"{max_roll:.6f}°"
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch:.6f}°"
    )

    if yaw_change < math.radians(20.0):

        print(
            "[FAIL] Robot did not rotate."
        )

        return False

    if max_roll > MAX_ROTATION_ROLL:

        print(
            "[FAIL] Excessive rotation roll."
        )

        return False

    if max_pitch > MAX_ROTATION_PITCH:

        print(
            "[FAIL] Excessive rotation pitch."
        )

        return False

    print(
        "[PASS] Rotation."
    )

    return True


# ============================================================
# TEST 4 — CURVED MOTION
# ============================================================

def test_curve(env):

    print()
    print("=" * 80)
    print("TEST 4 — CURVED MOTION")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "curve initial observation",
    ):

        return False

    action = np.array(
        [0.5, 1.0],
        dtype=np.float32,
    )

    start_x, start_y, _, _, _, start_yaw = (
        get_pose(env)
    )

    max_roll = 0.0
    max_pitch = 0.0

    for _ in range(
        PHYSICS_STEPS // 12
    ):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        if not validate_observation(
            observation,
            "curve observation",
        ):

            return False

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_pose(env)

        max_roll = max(
            max_roll,
            abs(math.degrees(roll)),
        )

        max_pitch = max(
            max_pitch,
            abs(math.degrees(pitch)),
        )

        if terminated or truncated:
            break

    displacement = math.sqrt(
        (x - start_x) ** 2
        +
        (y - start_y) ** 2
    )

    yaw_change = abs(
        wrap_angle(
            yaw - start_yaw
        )
    )

    print(
        f"Final position = "
        f"({x:.6f}, {y:.6f}, {z:.6f})"
    )

    print(
        f"Displacement = "
        f"{displacement:.6f} m"
    )

    print(
        f"|Yaw change| = "
        f"{math.degrees(yaw_change):.6f}°"
    )

    print(
        f"Max |roll| = "
        f"{max_roll:.6f}°"
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch:.6f}°"
    )

    if displacement < 0.10:

        print(
            "[FAIL] Robot did not move."
        )

        return False

    if yaw_change < math.radians(5.0):

        print(
            "[FAIL] Robot did not curve."
        )

        return False

    if max_roll > MAX_CURVE_ROLL:

        print(
            "[FAIL] Excessive curve roll."
        )

        return False

    if max_pitch > MAX_CURVE_PITCH:

        print(
            "[FAIL] Excessive curve pitch."
        )

        return False

    print(
        "[PASS] Curved motion."
    )

    return True


# ============================================================
# TEST 5 — SENSOR NOISE
# ============================================================

def test_sensor_noise():

    print()
    print("=" * 80)
    print("TEST 5 — SENSOR NOISE")
    print("=" * 80)

    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    )

    try:

        observation, info = env.reset(
            seed=123
        )

        if not validate_observation(
            observation,
            "noise observation",
        ):

            return False

        observations = []

        for _ in range(20):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            )

            if not validate_observation(
                observation,
                "noise step observation",
            ):

                return False

            observations.append(
                observation.copy()
            )

        observations = np.asarray(
            observations
        )

        variation = np.max(
            np.ptp(
                observations,
                axis=0
            )
        )

        print(
            f"Maximum observed sensor "
            f"variation = {variation:.8f}"
        )

        print(
            "[PASS] Sensor noise remains "
            "finite and normalized."
        )

        return True

    finally:

        env.close()


# ============================================================
# TEST 6 — GOAL TERMINATION
# ============================================================

def test_goal_termination(env):

    print()
    print("=" * 80)
    print("TEST 6 — GOAL TERMINATION")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "goal initial observation",
    ):

        return False

    position, _ = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    env.goal_position = np.array(
        [
            position[0],
            position[1],
        ],
        dtype=np.float32,
    )

    env.previous_distance = (
        env._distance_to_goal(
            position
        )
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        )
    )

    print(
        f"Distance to goal = "
        f"{info['distance_to_goal']:.8f} m"
    )

    print(
        f"Goal reached = "
        f"{info['goal_reached']}"
    )

    print(
        f"Terminated = "
        f"{terminated}"
    )

    print(
        f"Goal reward = "
        f"{info['goal_reward']}"
    )

    if not info["goal_reached"]:

        print(
            "[FAIL] Goal was not detected."
        )

        return False

    if not terminated:

        print(
            "[FAIL] Goal did not terminate episode."
        )

        return False

    print(
        "[PASS] Goal termination."
    )

    return True


# ============================================================
# TEST 7 — COLLISION DETECTION
# ============================================================

def test_collision_detection(env):

    print()
    print("=" * 80)
    print("TEST 7 — COLLISION DETECTION")
    print("=" * 80)

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "collision initial observation",
    ):

        return False

    # --------------------------------------------------------
    # Get actual robot position.
    # --------------------------------------------------------

    position, orientation = (
        p.getBasePositionAndOrientation(
            env.robot_id
        )
    )

    # --------------------------------------------------------
    # Create temporary static obstacle.
    #
    # The obstacle intentionally overlaps the robot.
    # No simulation steps are performed before checking
    # the contact.
    # --------------------------------------------------------

    obstacle_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.15,
            0.15,
            0.15,
        ],
    )

    obstacle_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.15,
            0.15,
            0.15,
        ],
    )

    obstacle_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=(
            obstacle_collision
        ),
        baseVisualShapeIndex=(
            obstacle_visual
        ),
        basePosition=[
            position[0],
            position[1],
            position[2],
        ],
    )

    # --------------------------------------------------------
    # Ask Bullet to update broadphase/narrowphase collision
    # detection without running the dynamics solver.
    # --------------------------------------------------------

    p.performCollisionDetection()

    # --------------------------------------------------------
    # Direct contact query.
    # --------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=env.robot_id,
        bodyB=obstacle_id,
    )

    print(
        f"Temporary obstacle ID = "
        f"{obstacle_id}"
    )

    print(
        f"Direct robot-obstacle contacts = "
        f"{len(contacts)}"
    )

    # --------------------------------------------------------
    # Environment collision checker.
    # --------------------------------------------------------

    collision = env._check_collision()

    print(
        f"Environment collision result = "
        f"{collision}"
    )

    # --------------------------------------------------------
    # Additional contact information.
    # --------------------------------------------------------

    if contacts:

        first_contact = contacts[0]

        print(
            f"First contact distance = "
            f"{first_contact[8]:.8f} m"
        )

        print(
            f"First contact normal force = "
            f"{first_contact[9]:.8f} N"
        )

        print(
            f"Robot link index = "
            f"{first_contact[3]}"
        )

        print(
            f"Obstacle link index = "
            f"{first_contact[4]}"
        )

    # --------------------------------------------------------
    # Validation.
    # --------------------------------------------------------

    if len(contacts) == 0:

        print()
        print(
            "[FAIL] PyBullet produced no direct "
            "robot-obstacle contact."
        )

        print(
            "The failure occurs before "
            "environment._check_collision()."
        )

        return False

    if not collision:

        print()
        print(
            "[FAIL] PyBullet detected the contact, "
            "but environment._check_collision() "
            "returned False."
        )

        print(
            "The collision-checking logic needs "
            "to be investigated."
        )

        return False

    print()
    print(
        "[PASS] Collision detection."
    )

    return True


# ============================================================
# TEST 8 — KNOWN REVERSE LIMITATION
# ============================================================

def test_known_reverse_behavior(env):

    print()
    print("=" * 80)
    print("TEST 8 — KNOWN REVERSE-MOTION LIMITATION")
    print("=" * 80)

    print()
    print(
        "This is a documented limitation."
    )

    print(
        "It is NOT a V3.22.8 failure criterion."
    )

    print(
        "The behavior was identified during V3.22.5q."
    )

    observation = reset_robot(env)

    if not validate_observation(
        observation,
        "reverse initial observation",
    ):

        return False

    max_pitch = 0.0
    contact_count = 0

    steps = PHYSICS_STEPS // 12

    for _ in range(steps):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            np.array(
                [-1.0, -1.0],
                dtype=np.float32,
            )
        )

        if not validate_observation(
            observation,
            "reverse observation",
        ):

            return False

        (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        ) = get_pose(env)

        max_pitch = max(
            max_pitch,
            abs(
                math.degrees(pitch)
            ),
        )

        contacts = p.getContactPoints(
            bodyA=env.robot_id
        )

        caster_contact = False

        for contact in contacts:

            link_a = contact[3]
            link_b = contact[4]

            if (
                link_a == 3
                or link_b == 3
            ):

                caster_contact = True
                break

        if caster_contact:

            contact_count += 1

        if terminated or truncated:
            break

    contact_fraction = (
        contact_count / steps
    )

    print(
        f"Max |pitch| = "
        f"{max_pitch:.6f}°"
    )

    print(
        f"Caster contact fraction = "
        f"{contact_fraction * 100:.2f}%"
    )

    print()
    print(
        "[KNOWN LIMITATION]"
    )

    print(
        "Active reverse wheel control causes "
        "the previously observed caster-contact "
        "loss and chassis pitch."
    )

    print(
        "This behavior is documented and does "
        "NOT block V3.23."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("V3.22.8 — FINAL REGRESSION VALIDATION")
    print("=" * 80)

    print()
    print(
        "Environment: DifferentialDriveEnv"
    )

    print(
        "Observation: 48 values"
    )

    print(
        "Action: 2 continuous wheel commands"
    )

    print(
        "PPO training: DISABLED"
    )

    print(
        "Production environment modified: NO"
    )

    print()
    print(
        "This is the final validation gate "
        "before V3.23 PPO."
    )

    env = DifferentialDriveEnv(
        use_noise=False,
        render_mode="direct",
        max_episode_steps=1000,
    )

    results = []

    try:

        # ----------------------------------------------------
        # TEST 1
        # ----------------------------------------------------

        try:

            passed = test_static(env)

        except Exception as exc:

            print(
                "[FAIL] Static test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Static stability", passed)
        )

        # ----------------------------------------------------
        # TEST 2
        # ----------------------------------------------------

        try:

            passed = test_forward(env)

        except Exception as exc:

            print(
                "[FAIL] Forward test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Forward motion", passed)
        )

        # ----------------------------------------------------
        # TEST 3
        # ----------------------------------------------------

        try:

            passed = test_rotation(env)

        except Exception as exc:

            print(
                "[FAIL] Rotation test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Rotation", passed)
        )

        # ----------------------------------------------------
        # TEST 4
        # ----------------------------------------------------

        try:

            passed = test_curve(env)

        except Exception as exc:

            print(
                "[FAIL] Curve test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Curved motion", passed)
        )

        # ----------------------------------------------------
        # TEST 5
        # ----------------------------------------------------

        try:

            passed = test_sensor_noise()

        except Exception as exc:

            print(
                "[FAIL] Sensor noise test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Sensor noise", passed)
        )

        # ----------------------------------------------------
        # TEST 6
        # ----------------------------------------------------

        try:

            passed = test_goal_termination(
                env
            )

        except Exception as exc:

            print(
                "[FAIL] Goal test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Goal termination", passed)
        )

        # ----------------------------------------------------
        # TEST 7
        # ----------------------------------------------------

        try:

            passed = test_collision_detection(
                env
            )

        except Exception as exc:

            print(
                "[FAIL] Collision test exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            ("Collision detection", passed)
        )

        # ----------------------------------------------------
        # TEST 8
        # ----------------------------------------------------

        try:

            passed = test_known_reverse_behavior(
                env
            )

        except Exception as exc:

            print(
                "[FAIL] Reverse diagnostic exception:"
            )

            print(
                repr(exc)
            )

            passed = False

        results.append(
            (
                "Known reverse diagnostic",
                passed,
            )
        )

    finally:

        env.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("V3.22.8 FINAL REGRESSION SUMMARY")
    print("=" * 80)

    print()

    passed_count = 0
    failed_count = 0

    for name, passed in results:

        if passed:

            print(
                f"[PASS] {name}"
            )

            passed_count += 1

        else:

            print(
                f"[FAIL] {name}"
            )

            failed_count += 1

    print()

    print(
        f"Passed: {passed_count}/{len(results)}"
    )

    print(
        f"Failed: {failed_count}/{len(results)}"
    )

    print()

    # ========================================================
    # FINAL GATE
    # ========================================================

    if failed_count == 0:

        print(
            "[PASS] V3.22.8 FINAL REGRESSION COMPLETE"
        )

        print()
        print(
            "================================================"
        )

        print(
            "V3.22 VALIDATION COMPLETE"
        )

        print(
            "================================================"
        )

        print()
        print(
            "The environment is READY FOR V3.23 PPO."
        )

        print()
        print(
            "Next phase:"
        )

        print(
            "  V3.23.1 — PPO baseline training"
        )

    else:

        print(
            "[FAIL] V3.22.8 FINAL REGRESSION FAILED."
        )

        print()
        print(
            "Do NOT start PPO training yet."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()