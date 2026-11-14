from sensors.encoder import read_encoder
from sensors.imu import read_imu
from sensors.lidar import read_lidar

from sensors.noise import (
    add_encoder_noise,
    add_imu_noise,
    add_lidar_noise,
)


class SensorManager:
    def __init__(
        self,
        left_joint,
        right_joint,
        wheel_radius,
        num_lidar_rays=36,
        lidar_max_range=5.0,
        lidar_sensor_height=0.20,
        use_noise=False,
    ):
        self.left_joint = left_joint
        self.right_joint = right_joint
        self.wheel_radius = wheel_radius

        self.num_lidar_rays = num_lidar_rays
        self.lidar_max_range = lidar_max_range
        self.lidar_sensor_height = lidar_sensor_height

        self.use_noise = use_noise

    def read(self, robot_id):
        # ---------------------------------------------------------
        # 1. Read ideal sensor measurements
        # ---------------------------------------------------------

        encoder_data = read_encoder(
            robot_id,
            self.left_joint,
            self.right_joint,
            self.wheel_radius,
        )

        imu_data = read_imu(robot_id)

        lidar_data = read_lidar(
            robot_id,
            num_rays=self.num_lidar_rays,
            max_range=self.lidar_max_range,
            sensor_height=self.lidar_sensor_height,
        )

        # ---------------------------------------------------------
        # 2. Apply sensor noise if enabled
        # ---------------------------------------------------------

        if self.use_noise:
            encoder_data = add_encoder_noise(encoder_data)
            imu_data = add_imu_noise(imu_data)
            lidar_data = add_lidar_noise(lidar_data)

        # ---------------------------------------------------------
        # 3. Return unified sensor measurement
        # ---------------------------------------------------------

        return {
            "encoder": encoder_data,
            "imu": imu_data,
            "lidar": lidar_data,
        }