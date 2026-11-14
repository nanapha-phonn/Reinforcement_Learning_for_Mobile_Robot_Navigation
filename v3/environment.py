import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from robot import (
    CHASSIS_CENTER_Z,
    create_robot,
    set_wheel_velocity,
    stop_robot,
)

from sensors.sensor_manager import SensorManager
from sensors.noise import set_noise_seed


class DifferentialDriveEnv(gym.Env):
    """
    V3.22.x Gymnasium environment for the V3
    differential-drive robot.

    Raw sensors:
        4 encoder values
        6 IMU values
        36 LiDAR ranges

    Goal information:
        1 goal distance
        1 goal bearing

    Total observation values:
        48

    RL observation:
        48 normalized values

    Action:
        [left_wheel_command, right_wheel_command]

    Each action value is in [-1, 1].

    Current reward:
        progress
        + heading alignment
        - action penalty
        + collision penalty
        + goal reward

    This environment uses the physical robot defined in
    robot.py.

    V3.22 structural consistency:
        - robot reset height uses CHASSIS_CENTER_Z
          from robot.py
        - robot hierarchy is defined in robot.py
    """

    metadata = {
        "render_modes": [
            "direct",
            "human",
        ]
    }

    # =========================================================
    # NORMALIZATION PARAMETERS
    # =========================================================

    MAX_WHEEL_VELOCITY = 4.0
    MAX_ANGULAR_VELOCITY = 10.0
    WHEEL_ANGLE_SCALE = 10.0

    GOAL_DISTANCE_SCALE = 5.0

    # =========================================================
    # V3.21.7 REWARD PARAMETERS
    # =========================================================

    HEADING_DISTANCE = 1.0
    HEADING_WEIGHT = 0.02

    ACTION_PENALTY_WEIGHT = 0.001

    COLLISION_PENALTY = -10.0
    GOAL_REWARD = 10.0

    def __init__(
        self,
        use_noise=True,
        render_mode="direct",
        max_episode_steps=1000,
    ):
        super().__init__()

        self.use_noise = use_noise
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # -----------------------------------------------------
        # Robot parameters
        # -----------------------------------------------------

        self.wheel_radius = 0.08

        self.left_wheel_joint = 0
        self.right_wheel_joint = 1

        self.max_wheel_velocity = 4.0

        self.time_step = 1.0 / 240.0

        # One RL action is held for several physics steps.
        self.frame_skip = 12

        # -----------------------------------------------------
        # IMPORTANT:
        # Use the physical robot geometry directly.
        #
        # robot.py:
        #
        # WHEEL_RADIUS = 0.08
        # CHASSIS_HEIGHT = 0.10
        #
        # Therefore:
        #
        # CHASSIS_CENTER_Z = 0.13 m
        # -----------------------------------------------------

        self.chassis_center_z = CHASSIS_CENTER_Z

        # -----------------------------------------------------
        # LiDAR parameters
        # -----------------------------------------------------

        self.num_lidar_rays = 36
        self.lidar_max_range = 5.0
        self.lidar_sensor_height = 0.20

        # -----------------------------------------------------
        # Gymnasium action space
        # -----------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # Gymnasium observation space
        #
        # Encoder: 4
        # IMU:     6
        # LiDAR:  36
        # Goal:    2
        #
        # Total: 48
        # -----------------------------------------------------

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(48,),
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # PyBullet
        # -----------------------------------------------------

        self.physics_client = p.connect(
            p.DIRECT
        )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
        )

        p.setTimeStep(
            self.time_step
        )

        # -----------------------------------------------------
        # Runtime state
        # -----------------------------------------------------

        self.robot_id = None

        self.goal_position = np.array(
            [3.0, 0.0],
            dtype=np.float32,
        )

        self.previous_distance = None

        self.step_count = 0

        # -----------------------------------------------------
        # Sensor manager
        # -----------------------------------------------------

        self.sensor_manager = SensorManager(
            left_joint=self.left_wheel_joint,
            right_joint=self.right_wheel_joint,
            wheel_radius=self.wheel_radius,
            num_lidar_rays=self.num_lidar_rays,
            lidar_max_range=self.lidar_max_range,
            lidar_sensor_height=self.lidar_sensor_height,
            use_noise=self.use_noise,
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        # -----------------------------------------------------
        # Reset simulation
        # -----------------------------------------------------

        p.resetSimulation()

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
        )

        p.setTimeStep(
            self.time_step
        )

        # -----------------------------------------------------
        # Ground
        # -----------------------------------------------------

        p.loadURDF(
            "plane.urdf"
        )

        # -----------------------------------------------------
        # Robot
        # -----------------------------------------------------

        self.robot_id = create_robot()

        # -----------------------------------------------------
        # Place robot at the correct geometric height.
        #
        # The robot is initially level.
        #
        # Z = CHASSIS_CENTER_Z = 0.13 m
        #
        # The previous version used 0.12 m, which was
        # inconsistent with the wheel/chassis geometry.
        # -----------------------------------------------------

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [
                0.0,
                0.0,
                self.chassis_center_z,
            ],
            p.getQuaternionFromEuler(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )

        # -----------------------------------------------------
        # Stop drive wheels
        # -----------------------------------------------------

        stop_robot(
            self.robot_id
        )

        # -----------------------------------------------------
        # Goal
        # -----------------------------------------------------

        self.goal_position = np.array(
            [3.0, 0.0],
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # Reset counters
        # -----------------------------------------------------

        self.step_count = 0

        # -----------------------------------------------------
        # Reset noise generator
        # -----------------------------------------------------

        if seed is not None:
            set_noise_seed(seed)

        # -----------------------------------------------------
        # Let physical model settle
        # -----------------------------------------------------

        for _ in range(120):

            p.stepSimulation()

        # -----------------------------------------------------
        # Initial distance
        # -----------------------------------------------------

        robot_position, _ = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        self.previous_distance = (
            self._distance_to_goal(
                robot_position
            )
        )

        # -----------------------------------------------------
        # Initial observation
        # -----------------------------------------------------

        observation = (
            self._get_observation()
        )

        info = {
            "step": self.step_count,

            "goal_x": float(
                self.goal_position[0]
            ),

            "goal_y": float(
                self.goal_position[1]
            ),
        }

        return (
            observation,
            info,
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        left_command = float(
            action[0]
        )

        right_command = float(
            action[1]
        )

        # -----------------------------------------------------
        # Convert normalized action to wheel velocity
        # -----------------------------------------------------

        left_velocity = (
            left_command *
            self.max_wheel_velocity
        )

        right_velocity = (
            right_command *
            self.max_wheel_velocity
        )

        # -----------------------------------------------------
        # Apply wheel commands
        # -----------------------------------------------------

        set_wheel_velocity(
            self.robot_id,
            left_velocity,
            right_velocity,
        )

        # -----------------------------------------------------
        # Run physics
        # -----------------------------------------------------

        for _ in range(
            self.frame_skip
        ):

            p.stepSimulation()

        self.step_count += 1

        # -----------------------------------------------------
        # Observation
        # -----------------------------------------------------

        observation = (
            self._get_observation()
        )

        # =====================================================
        # INTERNAL ROBOT STATE
        # =====================================================

        robot_position, robot_orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        current_distance = (
            self._distance_to_goal(
                robot_position
            )
        )

        # =====================================================
        # GOAL BEARING
        # =====================================================

        robot_x = robot_position[0]
        robot_y = robot_position[1]

        dx = (
            self.goal_position[0] -
            robot_x
        )

        dy = (
            self.goal_position[1] -
            robot_y
        )

        _, _, robot_yaw = (
            p.getEulerFromQuaternion(
                robot_orientation
            )
        )

        goal_angle = math.atan2(
            dy,
            dx,
        )

        current_goal_bearing = (
            goal_angle -
            robot_yaw
        )

        current_goal_bearing = math.atan2(
            math.sin(
                current_goal_bearing
            ),
            math.cos(
                current_goal_bearing
            ),
        )

        # =====================================================
        # REWARD
        # =====================================================

        # -----------------------------------------------------
        # Distance progress
        # -----------------------------------------------------

        progress = (
            self.previous_distance -
            current_distance
        )

        # -----------------------------------------------------
        # Distance-dependent heading weight
        # -----------------------------------------------------

        heading_weight = np.clip(
            1.0 -
            current_distance /
            self.HEADING_DISTANCE,
            0.0,
            1.0,
        )

        # -----------------------------------------------------
        # Absolute heading alignment
        # -----------------------------------------------------

        heading_alignment = math.cos(
            current_goal_bearing
        )

        heading_reward = (
            self.HEADING_WEIGHT *
            heading_weight *
            heading_alignment
        )

        # -----------------------------------------------------
        # Combined reward
        # -----------------------------------------------------

        reward = float(
            progress +
            heading_reward
        )

        self.previous_distance = (
            current_distance
        )

        # -----------------------------------------------------
        # Action penalty
        # -----------------------------------------------------

        action_penalty = (
            self.ACTION_PENALTY_WEIGHT *
            float(
                np.sum(
                    np.square(action)
                )
            )
        )

        reward -= action_penalty

        # -----------------------------------------------------
        # Collision
        # -----------------------------------------------------

        collision = (
            self._check_collision()
        )

        collision_reward = 0.0

        if collision:

            collision_reward = (
                self.COLLISION_PENALTY
            )

            reward += collision_reward

        # -----------------------------------------------------
        # Goal
        # -----------------------------------------------------

        goal_reached = (
            current_distance < 0.20
        )

        goal_reward = 0.0

        if goal_reached:

            goal_reward = (
                self.GOAL_REWARD
            )

            reward += goal_reward

        # -----------------------------------------------------
        # Termination
        # -----------------------------------------------------

        terminated = (
            collision or
            goal_reached
        )

        # -----------------------------------------------------
        # Truncation
        # -----------------------------------------------------

        truncated = (
            self.step_count >=
            self.max_episode_steps
        )

        # -----------------------------------------------------
        # Debug information
        # -----------------------------------------------------

        info = {

            "step":
                self.step_count,

            "distance_to_goal":
                float(
                    current_distance
                ),

            "progress":
                float(
                    progress
                ),

            "goal_bearing":
                float(
                    current_goal_bearing
                ),

            "heading_weight":
                float(
                    heading_weight
                ),

            "heading_alignment":
                float(
                    heading_alignment
                ),

            "heading_reward":
                float(
                    heading_reward
                ),

            "action_penalty":
                float(
                    action_penalty
                ),

            "collision_reward":
                float(
                    collision_reward
                ),

            "goal_reward":
                float(
                    goal_reward
                ),

            "collision":
                collision,

            "goal_reached":
                goal_reached,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self):

        sensor_data = (
            self.sensor_manager.read(
                self.robot_id
            )
        )

        encoder = sensor_data[
            "encoder"
        ]

        imu = sensor_data[
            "imu"
        ]

        lidar = sensor_data[
            "lidar"
        ]

        # =====================================================
        # ENCODER
        # =====================================================

        left_angle = np.tanh(
            encoder["left_angle"] /
            self.WHEEL_ANGLE_SCALE
        )

        right_angle = np.tanh(
            encoder["right_angle"] /
            self.WHEEL_ANGLE_SCALE
        )

        left_velocity = np.clip(
            encoder["left_velocity"] /
            self.MAX_WHEEL_VELOCITY,
            -1.0,
            1.0,
        )

        right_velocity = np.clip(
            encoder["right_velocity"] /
            self.MAX_WHEEL_VELOCITY,
            -1.0,
            1.0,
        )

        # =====================================================
        # IMU
        # =====================================================

        roll = np.clip(
            imu["roll"] / np.pi,
            -1.0,
            1.0,
        )

        pitch = np.clip(
            imu["pitch"] / np.pi,
            -1.0,
            1.0,
        )

        yaw = np.arctan2(
            np.sin(
                imu["yaw"]
            ),
            np.cos(
                imu["yaw"]
            ),
        )

        yaw = yaw / np.pi

        angular_vel_x = np.clip(
            imu["angular_vel_x"] /
            self.MAX_ANGULAR_VELOCITY,
            -1.0,
            1.0,
        )

        angular_vel_y = np.clip(
            imu["angular_vel_y"] /
            self.MAX_ANGULAR_VELOCITY,
            -1.0,
            1.0,
        )

        angular_vel_z = np.clip(
            imu["angular_vel_z"] /
            self.MAX_ANGULAR_VELOCITY,
            -1.0,
            1.0,
        )

        # =====================================================
        # LIDAR
        # =====================================================

        lidar_ranges = np.asarray(
            lidar["ranges"],
            dtype=np.float32,
        )

        lidar_ranges = np.clip(
            lidar_ranges /
            self.lidar_max_range,
            0.0,
            1.0,
        )

        # =====================================================
        # GOAL INFORMATION
        # =====================================================

        robot_position, robot_orientation = (
            p.getBasePositionAndOrientation(
                self.robot_id
            )
        )

        robot_x = robot_position[0]
        robot_y = robot_position[1]

        dx = (
            self.goal_position[0] -
            robot_x
        )

        dy = (
            self.goal_position[1] -
            robot_y
        )

        goal_distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        goal_distance = np.clip(
            goal_distance /
            self.GOAL_DISTANCE_SCALE,
            0.0,
            1.0,
        )

        _, _, robot_yaw = (
            p.getEulerFromQuaternion(
                robot_orientation
            )
        )

        goal_angle = math.atan2(
            dy,
            dx,
        )

        goal_bearing = (
            goal_angle -
            robot_yaw
        )

        goal_bearing = math.atan2(
            math.sin(
                goal_bearing
            ),
            math.cos(
                goal_bearing
            ),
        )

        goal_bearing = (
            goal_bearing /
            np.pi
        )

        # =====================================================
        # FINAL OBSERVATION
        # =====================================================

        observation = np.array(
            [
                # Encoder: 4
                left_angle,
                right_angle,
                left_velocity,
                right_velocity,

                # IMU: 6
                roll,
                pitch,
                yaw,
                angular_vel_x,
                angular_vel_y,
                angular_vel_z,

                # LiDAR: 36
                *lidar_ranges,

                # Goal: 2
                goal_distance,
                goal_bearing,
            ],
            dtype=np.float32,
        )

        observation = np.clip(
            observation,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        return observation

    # =========================================================
    # GOAL DISTANCE
    # =========================================================

    def _distance_to_goal(
        self,
        robot_position,
    ):

        dx = (
            self.goal_position[0] -
            robot_position[0]
        )

        dy = (
            self.goal_position[1] -
            robot_position[1]
        )

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # =========================================================
    # COLLISION
    # =========================================================

    def _check_collision(self):

        contacts = p.getContactPoints(
            bodyA=self.robot_id
        )

        for contact in contacts:

            other_body = contact[2]

            # Ignore self-contact.
            if other_body == self.robot_id:
                continue

            # Ignore ground plane.
            if other_body == 0:
                continue

            # Any other body is an obstacle.
            if other_body >= 0:
                return True

        return False

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        if self.physics_client is not None:

            p.disconnect(
                self.physics_client
            )

            self.physics_client = None
