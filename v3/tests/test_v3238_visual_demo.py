import sys
from pathlib import Path

# ------------------------------------------------------------
# Make project root importable when running from tests\
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math
import time

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from environment import DifferentialDriveEnv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v3236_generalized_ppo.zip"
)

WHEEL_MAX_VELOCITY = 4.0

MAX_STEPS = 1000

SETTLE_STEPS = 120

CAMERA_DISTANCE = 4.0
CAMERA_YAW = 45.0
CAMERA_PITCH = -35.0

SIMULATION_DELAY = 1.0 / 120.0

GOAL_RADIUS = 0.20

NUM_SCENARIOS = 10


# ============================================================
# TEST SCENARIOS
# ============================================================

SCENARIOS = [
    {
        "start": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (3.0, 0.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (0.0, 3.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (0.0, -3.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (3.0, 1.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 0.0,
        "goal": (3.0, -1.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 90.0,
        "goal": (3.0, 0.0),
    },

    {
        "start": (0.0, 0.0),
        "yaw": 180.0,
        "goal": (3.0, 0.0),
    },

    {
        "start": (1.0, 1.0),
        "yaw": 0.0,
        "goal": (3.0, 1.0),
    },

    {
        "start": (-1.0, 1.0),
        "yaw": 180.0,
        "goal": (2.0, -1.0),
    },

    {
        "start": (0.5, -0.5),
        "yaw": 135.0,
        "goal": (2.0, 2.0),
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_robot_state(env):
    """
    Return robot position and orientation.
    """

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        orientation
    )

    return (
        np.array(position, dtype=float),
        np.array([roll, pitch, yaw], dtype=float),
    )


def get_goal_metrics(env):
    """
    Calculate distance and bearing from robot to goal.
    """

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id
    )

    _, _, yaw = p.getEulerFromQuaternion(
        orientation
    )

    dx = env.goal_position[0] - position[0]
    dy = env.goal_position[1] - position[1]

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    world_angle = math.atan2(dy, dx)

    bearing = wrap_angle(
        world_angle - yaw
    )

    return distance, bearing


def set_camera(env):
    """
    Make the camera follow the robot.
    """

    position, _ = p.getBasePositionAndOrientation(
        env.robot_id
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=position,
    )


def create_goal_marker(goal):
    """
    Create a visible red sphere at the goal position.
    """

    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=GOAL_RADIUS,
        rgbaColor=[1.0, 0.0, 0.0, 1.0],
    )

    goal_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        baseCollisionShapeIndex=-1,
        basePosition=[
            goal[0],
            goal[1],
            GOAL_RADIUS,
        ],
    )

    return goal_id


def create_start_marker(start):
    """
    Create a small visual marker at the start position.
    """

    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.05,
        length=0.01,
        rgbaColor=[0.0, 1.0, 0.0, 1.0],
    )

    marker_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        baseCollisionShapeIndex=-1,
        basePosition=[
            start[0],
            start[1],
            0.005,
        ],
    )

    return marker_id


def create_trajectory_line():
    """
    Return an empty list used to track trajectory debug lines.
    """

    return []


def draw_trajectory(points, line_ids):
    """
    Draw trajectory as connected debug lines.
    """

    if len(points) < 2:
        return line_ids

    p1 = points[-2]
    p2 = points[-1]

    line_id = p.addUserDebugLine(
        [p1[0], p1[1], 0.03],
        [p2[0], p2[1], 0.03],
        lineColorRGB=[0.0, 0.8, 1.0],
        lineWidth=2.0,
        lifeTime=0,
    )

    line_ids.append(line_id)

    return line_ids


def setup_scenario(env, scenario):
    """
    Reset robot and manually configure the selected
    start pose and goal.
    """

    observation, info = env.reset(
        seed=1000
    )

    start_x, start_y = scenario["start"]

    yaw = math.radians(
        scenario["yaw"]
    )

    p.resetBasePositionAndOrientation(
        env.robot_id,
        [
            start_x,
            start_y,
            env.chassis_center_z,
        ],
        p.getQuaternionFromEuler(
            [0.0, 0.0, yaw]
        ),
    )

    p.resetBaseVelocity(
        env.robot_id,
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
    )

    env.goal_position = np.array(
        scenario["goal"],
        dtype=np.float32,
    )

    env.step_count = 0

    # Let the robot settle before starting.
    for _ in range(SETTLE_STEPS):
        p.stepSimulation()

    distance, _ = get_goal_metrics(env)

    env.previous_distance = distance

    observation = env._get_observation()

    return observation


def print_status(
    scenario_index,
    step,
    action,
    distance,
    bearing,
    success,
    collision,
):
    """
    Print compact simulation status.
    """

    left = float(action[0])
    right = float(action[1])

    print(
        f"\r"
        f"Scenario {scenario_index:02d} | "
        f"Step {step:03d} | "
        f"Dist {distance:.3f} m | "
        f"Bearing {math.degrees(bearing):7.2f}° | "
        f"L {left:+.2f} | "
        f"R {right:+.2f} | "
        f"Success {success} | "
        f"Collision {collision}",
        end="",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V3.23.8 VISUAL PPO SIMULATION")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print()
    print("Loading PPO model...")

    model = PPO.load(
        str(MODEL_PATH)
    )

    print("[PASS] PPO model loaded.")

    print()
    print("Opening PyBullet GUI...")

    physics_client = p.connect(
        p.GUI
    )

    if physics_client < 0:
        raise RuntimeError(
            "Could not open PyBullet GUI."
        )

    print("[PASS] PyBullet GUI connected.")

    # Create environment after GUI is connected.
    env = DifferentialDriveEnv(
        use_noise=True,
        render_mode="direct",
    )

    env.render_mode = "gui"

    print()
    print("=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print("Close the PyBullet window to stop.")
    print()
    print("The PPO policy is running automatically.")
    print("The camera follows the robot.")
    print("Blue line = robot trajectory.")
    print("Red sphere = goal.")
    print("Green disk = starting position.")
    print("=" * 70)

    total_successes = 0
    total_collisions = 0

    for scenario_index in range(
        min(NUM_SCENARIOS, len(SCENARIOS))
    ):

        scenario = SCENARIOS[
            scenario_index
        ]

        print()
        print()
        print("=" * 70)
        print(
            f"SCENARIO {scenario_index + 1:02d}"
        )
        print("=" * 70)

        print(
            f"Start = "
            f"({scenario['start'][0]:.2f}, "
            f"{scenario['start'][1]:.2f})"
        )

        print(
            f"Yaw   = "
            f"{scenario['yaw']:.1f}°"
        )

        print(
            f"Goal  = "
            f"({scenario['goal'][0]:.2f}, "
            f"{scenario['goal'][1]:.2f})"
        )

        observation = setup_scenario(
            env,
            scenario,
        )

        goal_id = create_goal_marker(
            scenario["goal"]
        )

        start_id = create_start_marker(
            scenario["start"]
        )

        trajectory_points = []
        trajectory_lines = []

        success = False
        collision = False

        previous_position, _ = get_robot_state(
            env
        )

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

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Physics/environment step
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            position, orientation = get_robot_state(
                env
            )

            distance, bearing = get_goal_metrics(
                env
            )

            trajectory_points.append(
                position.copy()
            )

            trajectory_lines = draw_trajectory(
                trajectory_points,
                trajectory_lines,
            )

            collision = env._check_collision()

            success = (
                distance < GOAL_RADIUS
            )

            # ------------------------------------------------
            # Camera
            # ------------------------------------------------

            set_camera(env)

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            print_status(
                scenario_index + 1,
                step,
                action,
                distance,
                bearing,
                success,
                collision,
            )

            # ------------------------------------------------
            # Slow simulation slightly so it is visible.
            # ------------------------------------------------

            time.sleep(
                SIMULATION_DELAY
            )

            # ------------------------------------------------
            # Stop condition
            # ------------------------------------------------

            if success or collision:
                break

            if terminated or truncated:
                break

            previous_position = position

        print()

        if success:
            total_successes += 1
            print(
                f"[PASS] Scenario "
                f"{scenario_index + 1:02d} reached goal."
            )
        else:
            print(
                f"[INFO] Scenario "
                f"{scenario_index + 1:02d} ended."
            )

        if collision:
            total_collisions += 1
            print(
                "[WARNING] Collision detected."
            )

        print(
            f"Final distance = "
            f"{distance:.4f} m"
        )

        print(
            f"Final bearing = "
            f"{math.degrees(bearing):.2f}°"
        )

        print(
            f"Final yaw = "
            f"{math.degrees(orientation[2]):.2f}°"
        )

        print(
            f"Final roll = "
            f"{math.degrees(orientation[0]):.4f}°"
        )

        print(
            f"Final pitch = "
            f"{math.degrees(orientation[1]):.4f}°"
        )

        # Give the user time to observe the result.
        print()
        print(
            "Displaying result for 2 seconds..."
        )

        end_time = time.time() + 2.0

        while (
            time.time() < end_time
            and p.isConnected()
        ):
            set_camera(env)
            p.stepSimulation()
            time.sleep(1.0 / 60.0)

        # Remove markers and trajectory.
        if p.isConnected():

            p.removeBody(goal_id)

            p.removeBody(start_id)

            for line_id in trajectory_lines:
                try:
                    p.removeUserDebugItem(
                        line_id
                    )
                except Exception:
                    pass

        if not p.isConnected():
            print()
            print(
                "PyBullet window was closed."
            )
            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("V3.23.8 VISUAL SIMULATION COMPLETE")
    print("=" * 70)

    print(
        f"Successful scenarios = "
        f"{total_successes}/{min(NUM_SCENARIOS, len(SCENARIOS))}"
    )

    print(
        f"Collisions           = "
        f"{total_collisions}"
    )

    print()
    print(
        "The visualization is a demonstration of the "
        "already-trained PPO policy."
    )

    print(
        "No robot.py or environment.py files were modified."
    )

    print("=" * 70)

    if p.isConnected():
        p.disconnect()

    env.close()


if __name__ == "__main__":
    main()