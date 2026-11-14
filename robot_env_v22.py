import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class RobotReachEnvV22(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # ACTION
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ==================================================
        # OBSERVATION
        # [distance, angle]
        # ==================================================

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

        # V2.2
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
        # ROBOT
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
        # RANDOM HEADING
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

        distance, angle = (
            self._calculate_target_state()
        )

        self.previous_distance = distance
        self.previous_angle = abs(angle)

        return self._get_observation(), {}

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

        dx = (
            self.target_position[0]
            - robot_x
        )

        dy = (
            self.target_position[1]
            - robot_y
        )

        distance = np.sqrt(
            dx * dx + dy * dy
        )

        target_heading = np.arctan2(
            dy,
            dx
        )

        angle = (
            target_heading
            - heading
        )

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
            [distance, angle],
            dtype=np.float32
        )

    def step(self, action):

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
        # WHEEL VELOCITIES
        # ==================================================

        left_velocity = float(action[0])
        right_velocity = float(action[1])

        v_left = (
            left_velocity
            * self.wheel_radius
        )

        v_right = (
            right_velocity
            * self.wheel_radius
        )

        # ==================================================
        # DIFFERENTIAL DRIVE
        # ==================================================

        linear_velocity = (
            v_right + v_left
        ) / 2.0

        angular_velocity = (
            v_right - v_left
        ) / self.wheel_base

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

        new_heading = (
            heading
            + angular_velocity
            * self.dt
        )

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
        # NEW STATE
        # ==================================================

        distance, angle = (
            self._calculate_target_state()
        )

        # ==================================================
        # V2.2 REWARD
        # ==================================================

        distance_progress = (
            self.previous_distance
            - distance
        )

        angle_progress = (
            self.previous_angle
            - abs(angle)
        )

        control_penalty = (
            0.001
            * np.sum(
                np.square(action)
            )
        )

        time_penalty = 0.001

        reward = (
            distance_progress
            + 0.05 * angle_progress
            - control_penalty
            - time_penalty
        )

        self.previous_distance = distance
        self.previous_angle = abs(angle)

        self.step_count += 1

        # ==================================================
        # TERMINATION
        # ==================================================

        terminated = (
            distance < self.success_radius
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        # ==================================================
        # SUCCESS
        # ==================================================

        if terminated:
            reward += 20.0

        return (
            self._get_observation(),
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