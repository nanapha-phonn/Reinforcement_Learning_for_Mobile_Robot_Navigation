import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class RobotReachEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # ACTION SPACE
        # ==================================================
        # [left wheel velocity, right wheel velocity]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ==================================================
        # OBSERVATION SPACE
        # ==================================================
        # [distance_to_target, target_angle]

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -np.pi],
                dtype=np.float32
            ),
            high=np.array(
                [5.0, np.pi],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # ==================================================
        # PYBULLET
        # ==================================================

        self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        self.robot = None
        self.target_position = None

        self.previous_distance = None
        self.previous_angle = None

        self.step_count = 0

        # ==================================================
        # EPISODE SETTINGS
        # ==================================================

        self.max_steps = 300

        # V2.5 SUCCESS CRITERION
        self.success_radius = 0.20

        # ==================================================
        # DIFFERENTIAL DRIVE
        # ==================================================

        self.wheel_radius = 0.08
        self.wheel_base = 0.30

        self.dt = 0.1

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        p.resetSimulation()

        p.setGravity(
            0,
            0,
            -9.81
        )

        p.loadURDF("plane.urdf")

        # ==================================================
        # ROBOT BODY
        # ==================================================

        body_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                0.20,
                0.15,
                0.10
            ]
        )

        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                0.20,
                0.15,
                0.10
            ]
        )

        self.robot = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[
                0,
                0,
                0.10
            ]
        )

        # ==================================================
        # RANDOM INITIAL HEADING
        # ==================================================

        heading = self.np_random.uniform(
            -np.pi,
            np.pi
        )

        orientation = p.getQuaternionFromEuler(
            [0, 0, heading]
        )

        p.resetBasePositionAndOrientation(
            self.robot,
            [0, 0, 0.10],
            orientation
        )

        # ==================================================
        # RANDOM TARGET
        # ==================================================

        self.target_position = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.0, 2.0)
        ])

        self.step_count = 0

        # ==================================================
        # INITIAL TARGET STATE
        # ==================================================

        distance, angle = self._calculate_target_state()

        self.previous_distance = distance
        self.previous_angle = abs(angle)

        observation = self._get_observation()

        return observation, {}

    def _calculate_target_state(self):

        position, orientation = (
            p.getBasePositionAndOrientation(
                self.robot
            )
        )

        robot_x = position[0]
        robot_y = position[1]

        heading = p.getEulerFromQuaternion(
            orientation
        )[2]

        # Target relative position

        dx = (
            self.target_position[0]
            - robot_x
        )

        dy = (
            self.target_position[1]
            - robot_y
        )

        # Distance

        distance = np.sqrt(
            dx * dx + dy * dy
        )

        # Direction to target

        target_heading = np.arctan2(
            dy,
            dx
        )

        # Relative angle

        angle = (
            target_heading
            - heading
        )

        # Wrap to [-pi, pi]

        angle = (
            angle + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        return distance, angle

    def _get_observation(self):

        distance, angle = (
            self._calculate_target_state()
        )

        return np.array(
            [
                distance,
                angle
            ],
            dtype=np.float32
        )

    def step(self, action):

        # ==================================================
        # CURRENT ROBOT STATE
        # ==================================================

        position, orientation = (
            p.getBasePositionAndOrientation(
                self.robot
            )
        )

        x, y, z = position

        heading = p.getEulerFromQuaternion(
            orientation
        )[2]

        # ==================================================
        # WHEEL COMMANDS
        # ==================================================

        left_velocity = float(action[0])
        right_velocity = float(action[1])

        # ==================================================
        # DIFFERENTIAL DRIVE KINEMATICS
        # ==================================================

        v_left = (
            left_velocity
            * self.wheel_radius
        )

        v_right = (
            right_velocity
            * self.wheel_radius
        )

        linear_velocity = (
            v_right + v_left
        ) / 2.0

        angular_velocity = (
            v_right - v_left
        ) / self.wheel_base

        # ==================================================
        # UPDATE POSITION
        # ==================================================

        new_x = (
            x
            + linear_velocity
            * np.cos(heading)
            * self.dt
        )

        new_y = (
            y
            + linear_velocity
            * np.sin(heading)
            * self.dt
        )

        # ==================================================
        # UPDATE HEADING
        # ==================================================

        new_heading = (
            heading
            + angular_velocity
            * self.dt
        )

        # Wrap heading

        new_heading = (
            new_heading + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        new_orientation = (
            p.getQuaternionFromEuler(
                [0, 0, new_heading]
            )
        )

        # ==================================================
        # UPDATE ROBOT
        # ==================================================

        p.resetBasePositionAndOrientation(
            self.robot,
            [
                new_x,
                new_y,
                z
            ],
            new_orientation
        )

        # ==================================================
        # NEW TARGET STATE
        # ==================================================

        distance, angle = (
            self._calculate_target_state()
        )

        # ==================================================
        # V2.5 REWARD
        # ==================================================

        # 1. Distance progress

        distance_progress = (
            self.previous_distance
            - distance
        )

        # 2. Orientation progress

        angle_progress = (
            self.previous_angle
            - abs(angle)
        )

        # 3. Small control penalty

        control_penalty = (
            0.001
            * np.sum(
                np.square(action)
            )
        )

        # 4. Small time penalty

        time_penalty = 0.001

        # Total reward

        reward = (
            distance_progress
            + 0.05 * angle_progress
            - control_penalty
            - time_penalty
        )

        # ==================================================
        # UPDATE PREVIOUS STATE
        # ==================================================

        self.previous_distance = distance
        self.previous_angle = abs(angle)

        self.step_count += 1

        # ==================================================
        # TERMINATION
        # ==================================================

        terminated = (
            distance
            < self.success_radius
        )

        truncated = (
            self.step_count
            >= self.max_steps
        )

        # ==================================================
        # SUCCESS BONUS
        # ==================================================

        if terminated:
            reward += 20.0

        # ==================================================
        # TIMEOUT PENALTY
        # ==================================================

        if truncated and not terminated:
            reward -= 2.0

        # ==================================================
        # OBSERVATION
        # ==================================================

        observation = (
            self._get_observation()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )

    def close(self):

        if p.isConnected():
            p.disconnect(
                self.physics_client
            )