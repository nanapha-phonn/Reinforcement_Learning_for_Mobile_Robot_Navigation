import pybullet as p
import pybullet_data
import time
import math


# ============================================================
# V3.14 PARAMETERS
# ============================================================

WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.04
WHEEL_MASS = 0.5
WHEEL_BASE = 0.30

BASE_LENGTH = 0.40
BASE_WIDTH = 0.24
BASE_HEIGHT = 0.10
BASE_MASS = 5.0

CASTER_RADIUS = 0.045
CASTER_MASS = 0.20
CASTER_X = -0.15

GRAVITY = -9.81
TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 3.0

LEFT_WHEEL_COMMAND = -4.0
RIGHT_WHEEL_COMMAND = 4.0
MOTOR_FORCE = 20.0
ROTATION_TIME = 3.0


# ============================================================
# HELPERS
# ============================================================

def quat_to_euler(quaternion):
    return p.getEulerFromQuaternion(quaternion)


def get_body_orientation(body_id):
    _, orientation = p.getBasePositionAndOrientation(body_id)
    return quat_to_euler(orientation)


def get_normal_force(body_id, link_index, plane_id):
    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=plane_id,
        linkIndexA=link_index
    )

    total_force = 0.0

    for contact in contacts:
        total_force += contact[9]

    return total_force


def get_contact_count(body_id, link_index, plane_id):
    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=plane_id,
        linkIndexA=link_index
    )

    return len(contacts)


def get_caster_contact_data(caster_id, plane_id):

    contacts = p.getContactPoints(
        bodyA=caster_id,
        bodyB=plane_id
    )

    total_force = 0.0
    contact_points = []

    for contact in contacts:
        total_force += contact[9]

        contact_points.append({
            "position": contact[5],
            "normal": contact[7],
            "force": contact[9]
        })

    return len(contacts), total_force, contact_points


# ============================================================
# CREATE ROBOT
# ============================================================

def create_robot():

    # --------------------------------------------------------
    # CHASSIS
    # --------------------------------------------------------

    chassis_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            BASE_LENGTH / 2,
            BASE_WIDTH / 2,
            BASE_HEIGHT / 2
        ]
    )

    chassis_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            BASE_LENGTH / 2,
            BASE_WIDTH / 2,
            BASE_HEIGHT / 2
        ]
    )

    # --------------------------------------------------------
    # WHEEL
    # --------------------------------------------------------

    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # --------------------------------------------------------
    # WHEEL POSITIONS
    # --------------------------------------------------------

    wheel_positions = [
        [0, WHEEL_BASE / 2, -BASE_HEIGHT / 2],
        [0, -WHEEL_BASE / 2, -BASE_HEIGHT / 2]
    ]

    # --------------------------------------------------------
    # ROBOT
    # --------------------------------------------------------

    robot_id = p.createMultiBody(
        baseMass=BASE_MASS,
        baseCollisionShapeIndex=chassis_collision,
        baseVisualShapeIndex=chassis_visual,
        basePosition=[0, 0, 0.13],

        linkMasses=[
            WHEEL_MASS,
            WHEEL_MASS
        ],

        linkCollisionShapeIndices=[
            wheel_collision,
            wheel_collision
        ],

        linkVisualShapeIndices=[
            wheel_visual,
            wheel_visual
        ],

        linkPositions=wheel_positions,

        linkOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],

        linkInertialFramePositions=[
            [0, 0, 0],
            [0, 0, 0]
        ],

        linkInertialFrameOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],

        linkParentIndices=[
            0,
            0
        ],

        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE
        ],

        linkJointAxis=[
            [0, 1, 0],
            [0, 1, 0]
        ]
    )

    # --------------------------------------------------------
    # DISABLE CHASSIS-WHEEL SELF COLLISION
    # --------------------------------------------------------

    p.setCollisionFilterPair(
        robot_id, robot_id, -1, 0, 0
    )

    p.setCollisionFilterPair(
        robot_id, robot_id, -1, 1, 0
    )

    p.setCollisionFilterPair(
        robot_id, robot_id, 0, 1, 0
    )

    # --------------------------------------------------------
    # CHASSIS DYNAMICS
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.8,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # --------------------------------------------------------
    # WHEEL DYNAMICS
    # --------------------------------------------------------

    for joint in [0, 1]:

        p.changeDynamics(
            robot_id,
            joint,
            lateralFriction=0.8,
            rollingFriction=0.0,
            spinningFriction=0.0
        )

        p.setJointMotorControl2(
            robot_id,
            joint,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # --------------------------------------------------------
    # CASTER
    # --------------------------------------------------------

    caster_collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=CASTER_RADIUS
    )

    caster_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=CASTER_RADIUS
    )

    caster_id = p.createMultiBody(
        baseMass=CASTER_MASS,
        baseCollisionShapeIndex=caster_collision,
        baseVisualShapeIndex=caster_visual,
        basePosition=[
            CASTER_X,
            0,
            CASTER_RADIUS
        ]
    )

    p.changeDynamics(
        caster_id,
        -1,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # --------------------------------------------------------
    # CASTER CONSTRAINT
    # --------------------------------------------------------

    caster_constraint = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1,
        childBodyUniqueId=caster_id,
        childLinkIndex=-1,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=[0, 0, 0],
        parentFramePosition=[
            CASTER_X,
            0,
            -BASE_HEIGHT / 2
        ],
        childFramePosition=[
            0, 0, 0
        ]
    )

    return robot_id, caster_id, caster_constraint


# ============================================================
# MAIN
# ============================================================

def main():

    physics_client = p.connect(p.GUI)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        GRAVITY
    )

    p.setTimeStep(TIME_STEP)

    # --------------------------------------------------------
    # GROUND
    # --------------------------------------------------------

    plane_id = p.loadURDF("plane.urdf")

    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=0.8,
        rollingFriction=0.0,
        spinningFriction=0.0
    )

    # --------------------------------------------------------
    # CREATE ROBOT
    # --------------------------------------------------------

    robot_id, caster_id, caster_constraint = create_robot()

    print()
    print("=" * 44)
    print("V3.14 THREE-WHEEL ROBOT")
    print("ROTATION + CASTER DIAGNOSTIC")
    print("=" * 44)

    print()
    print("MODEL")
    print("-" * 44)
    print("Two-wheel differential drive")
    print("Real PyBullet rigid-body dynamics")
    print("Passive spherical caster")
    print("Opposite wheel velocity command")
    print("No PPO")

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    print()
    print("=" * 44)
    print("GEOMETRY")
    print("=" * 44)

    print()
    print("CHASSIS")
    print("-" * 44)
    print(f"Length              : {BASE_LENGTH:.3f} m")
    print(f"Width               : {BASE_WIDTH:.3f} m")
    print(f"Height              : {BASE_HEIGHT:.3f} m")

    print()
    print("DRIVE WHEELS")
    print("-" * 44)
    print(f"Radius              : {WHEEL_RADIUS:.3f} m")
    print(f"Width               : {WHEEL_WIDTH:.3f} m")
    print(f"Wheel base          : {WHEEL_BASE:.3f} m")

    print()
    print("CASTER")
    print("-" * 44)
    print(f"Radius              : {CASTER_RADIUS:.3f} m")
    print(f"X position          : {CASTER_X:.3f} m")

    # --------------------------------------------------------
    # ROBOT INFORMATION
    # --------------------------------------------------------

    print()
    print(f"Robot ID            : {robot_id}")
    print(f"Caster ID           : {caster_id}")
    print(
        f"Number of joints    : "
        f"{p.getNumJoints(robot_id)}"
    )

    # --------------------------------------------------------
    # SETTLE
    # --------------------------------------------------------

    print()
    print("Settling robot...")

    settle_steps = int(
        SETTLE_TIME / TIME_STEP
    )

    for _ in range(settle_steps):
        p.stepSimulation()
        time.sleep(TIME_STEP)

    # --------------------------------------------------------
    # SETTLED STATE
    # --------------------------------------------------------

    start_position, start_orientation = \
        p.getBasePositionAndOrientation(robot_id)

    start_roll, start_pitch, start_yaw = \
        quat_to_euler(start_orientation)

    print()
    print("=" * 44)
    print("SETTLED STATE")
    print("=" * 44)

    print(
        f"Position : "
        f"({start_position[0]:.6f}, "
        f"{start_position[1]:.6f}, "
        f"{start_position[2]:.6f})"
    )

    print(
        f"Roll     : "
        f"{math.degrees(start_roll):.4f} deg"
    )

    print(
        f"Pitch    : "
        f"{math.degrees(start_pitch):.4f} deg"
    )

    print(
        f"Yaw      : "
        f"{math.degrees(start_yaw):.4f} deg"
    )

    # --------------------------------------------------------
    # INITIAL CASTER STATE
    # --------------------------------------------------------

    caster_start_position, caster_start_orientation = \
        p.getBasePositionAndOrientation(caster_id)

    caster_start_velocity, caster_start_angular_velocity = \
        p.getBaseVelocity(caster_id)

    print()
    print("INITIAL CASTER")
    print("-" * 44)

    print(
        f"Position : "
        f"({caster_start_position[0]:.6f}, "
        f"{caster_start_position[1]:.6f}, "
        f"{caster_start_position[2]:.6f})"
    )

    print(
        f"Velocity : "
        f"({caster_start_velocity[0]:.6f}, "
        f"{caster_start_velocity[1]:.6f}, "
        f"{caster_start_velocity[2]:.6f})"
    )

    # --------------------------------------------------------
    # THEORETICAL ROTATION
    # --------------------------------------------------------

    left_linear_velocity = (
        WHEEL_RADIUS * LEFT_WHEEL_COMMAND
    )

    right_linear_velocity = (
        WHEEL_RADIUS * RIGHT_WHEEL_COMMAND
    )

    theoretical_yaw_rate = (
        right_linear_velocity
        - left_linear_velocity
    ) / WHEEL_BASE

    print()
    print("=" * 44)
    print("V3.14 ROTATION DIAGNOSTIC")
    print("=" * 44)

    print(
        f"Left wheel command  : "
        f"{LEFT_WHEEL_COMMAND:.4f} rad/s"
    )

    print(
        f"Right wheel command : "
        f"{RIGHT_WHEEL_COMMAND:.4f} rad/s"
    )

    print(
        f"Motor force         : "
        f"{MOTOR_FORCE:.4f} N*m"
    )

    print(
        f"Rotation duration   : "
        f"{ROTATION_TIME:.4f} s"
    )

    print(
        f"Theoretical yaw rate: "
        f"{theoretical_yaw_rate:.4f} rad/s"
    )

    # --------------------------------------------------------
    # MOTOR COMMAND
    # --------------------------------------------------------

    p.setJointMotorControl2(
        robot_id,
        0,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=LEFT_WHEEL_COMMAND,
        force=MOTOR_FORCE
    )

    p.setJointMotorControl2(
        robot_id,
        1,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=RIGHT_WHEEL_COMMAND,
        force=MOTOR_FORCE
    )

    # --------------------------------------------------------
    # DIAGNOSTIC ARRAYS
    # --------------------------------------------------------

    left_velocity_samples = []
    right_velocity_samples = []

    yaw_rate_samples = []

    roll_samples = []
    pitch_samples = []
    yaw_samples = []

    caster_x_samples = []
    caster_y_samples = []
    caster_z_samples = []

    caster_vx_samples = []
    caster_vy_samples = []
    caster_vz_samples = []

    left_force_samples = []
    right_force_samples = []
    caster_force_samples = []

    caster_contact_samples = []

    # --------------------------------------------------------
    # ROTATION SIMULATION
    # --------------------------------------------------------

    rotation_steps = int(
        ROTATION_TIME / TIME_STEP
    )

    print()
    print("Running rotation diagnostic...")

    for step in range(rotation_steps):

        p.stepSimulation()

        # ----------------------------------------------------
        # WHEEL VELOCITIES
        # ----------------------------------------------------

        left_state = p.getJointState(
            robot_id,
            0
        )

        right_state = p.getJointState(
            robot_id,
            1
        )

        left_velocity = left_state[1]
        right_velocity = right_state[1]

        left_velocity_samples.append(
            left_velocity
        )

        right_velocity_samples.append(
            right_velocity
        )

        # ----------------------------------------------------
        # ROBOT VELOCITY
        # ----------------------------------------------------

        _, angular_velocity = \
            p.getBaseVelocity(robot_id)

        yaw_rate_samples.append(
            angular_velocity[2]
        )

        # ----------------------------------------------------
        # ROBOT ORIENTATION
        # ----------------------------------------------------

        _, orientation = \
            p.getBasePositionAndOrientation(
                robot_id
            )

        roll, pitch, yaw = \
            quat_to_euler(orientation)

        roll_samples.append(
            math.degrees(roll)
        )

        pitch_samples.append(
            math.degrees(pitch)
        )

        yaw_samples.append(
            math.degrees(yaw)
        )

        # ----------------------------------------------------
        # CASTER POSITION
        # ----------------------------------------------------

        caster_position, _ = \
            p.getBasePositionAndOrientation(
                caster_id
            )

        caster_velocity, _ = \
            p.getBaseVelocity(
                caster_id
            )

        caster_x_samples.append(
            caster_position[0]
        )

        caster_y_samples.append(
            caster_position[1]
        )

        caster_z_samples.append(
            caster_position[2]
        )

        caster_vx_samples.append(
            caster_velocity[0]
        )

        caster_vy_samples.append(
            caster_velocity[1]
        )

        caster_vz_samples.append(
            caster_velocity[2]
        )

        # ----------------------------------------------------
        # NORMAL FORCES
        # ----------------------------------------------------

        left_force = get_normal_force(
            robot_id,
            0,
            plane_id
        )

        right_force = get_normal_force(
            robot_id,
            1,
            plane_id
        )

        caster_contacts, caster_force, _ = \
            get_caster_contact_data(
                caster_id,
                plane_id
            )

        left_force_samples.append(
            left_force
        )

        right_force_samples.append(
            right_force
        )

        caster_force_samples.append(
            caster_force
        )

        caster_contact_samples.append(
            caster_contacts
        )

        # ----------------------------------------------------
        # SMALL DELAY FOR GUI
        # ----------------------------------------------------

        time.sleep(TIME_STEP)

    # --------------------------------------------------------
    # STOP MOTORS
    # --------------------------------------------------------

    p.setJointMotorControl2(
        robot_id,
        0,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )

    p.setJointMotorControl2(
        robot_id,
        1,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    final_position, final_orientation = \
        p.getBasePositionAndOrientation(robot_id)

    final_linear_velocity, final_angular_velocity = \
        p.getBaseVelocity(robot_id)

    final_roll, final_pitch, final_yaw = \
        quat_to_euler(final_orientation)

    # --------------------------------------------------------
    # AVERAGES
    # --------------------------------------------------------

    average_left = (
        sum(left_velocity_samples)
        / len(left_velocity_samples)
    )

    average_right = (
        sum(right_velocity_samples)
        / len(right_velocity_samples)
    )

    average_yaw_rate = (
        sum(yaw_rate_samples)
        / len(yaw_rate_samples)
    )

    average_roll = (
        sum(roll_samples)
        / len(roll_samples)
    )

    average_pitch = (
        sum(pitch_samples)
        / len(pitch_samples)
    )

    average_caster_x = (
        sum(caster_x_samples)
        / len(caster_x_samples)
    )

    average_caster_y = (
        sum(caster_y_samples)
        / len(caster_y_samples)
    )

    average_caster_z = (
        sum(caster_z_samples)
        / len(caster_z_samples)
    )

    average_caster_vx = (
        sum(caster_vx_samples)
        / len(caster_vx_samples)
    )

    average_caster_vy = (
        sum(caster_vy_samples)
        / len(caster_vy_samples)
    )

    average_caster_vz = (
        sum(caster_vz_samples)
        / len(caster_vz_samples)
    )

    average_left_force = (
        sum(left_force_samples)
        / len(left_force_samples)
    )

    average_right_force = (
        sum(right_force_samples)
        / len(right_force_samples)
    )

    average_caster_force = (
        sum(caster_force_samples)
        / len(caster_force_samples)
    )

    # --------------------------------------------------------
    # FINAL CASTER STATE
    # --------------------------------------------------------

    caster_final_position, caster_final_orientation = \
        p.getBasePositionAndOrientation(caster_id)

    caster_final_velocity, caster_final_angular_velocity = \
        p.getBaseVelocity(caster_id)

    final_caster_contacts, final_caster_force, \
        final_caster_contact_data = \
        get_caster_contact_data(
            caster_id,
            plane_id
        )

    # --------------------------------------------------------
    # FINAL CONTACTS
    # --------------------------------------------------------

    final_left_contacts = get_contact_count(
        robot_id,
        0,
        plane_id
    )

    final_right_contacts = get_contact_count(
        robot_id,
        1,
        plane_id
    )

    # --------------------------------------------------------
    # ROTATION RATIO
    # --------------------------------------------------------

    rotation_ratio = (
        abs(average_yaw_rate)
        / abs(theoretical_yaw_rate)
        * 100.0
    )

    # --------------------------------------------------------
    # TOTAL FORCE
    # --------------------------------------------------------

    total_average_force = (
        average_left_force
        + average_right_force
        + average_caster_force
    )

    expected_weight = (
        BASE_MASS
        + 2 * WHEEL_MASS
        + CASTER_MASS
    ) * abs(GRAVITY)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 44)
    print("V3.14 DIAGNOSTIC RESULTS")
    print("=" * 44)

    # --------------------------------------------------------
    # WHEEL DATA
    # --------------------------------------------------------

    print()
    print("WHEEL VELOCITIES")
    print("-" * 44)

    print(
        f"Average left      : "
        f"{average_left:.6f} rad/s"
    )

    print(
        f"Average right     : "
        f"{average_right:.6f} rad/s"
    )

    print(
        f"Final left        : "
        f"{left_velocity_samples[-1]:.6f} rad/s"
    )

    print(
        f"Final right       : "
        f"{right_velocity_samples[-1]:.6f} rad/s"
    )

    # --------------------------------------------------------
    # ROTATION
    # --------------------------------------------------------

    print()
    print("ROTATION")
    print("-" * 44)

    print(
        f"Theoretical yaw rate : "
        f"{theoretical_yaw_rate:.6f} rad/s"
    )

    print(
        f"Average yaw rate     : "
        f"{average_yaw_rate:.6f} rad/s"
    )

    print(
        f"Final yaw rate       : "
        f"{final_angular_velocity[2]:.6f} rad/s"
    )

    print(
        f"Rotation ratio       : "
        f"{rotation_ratio:.2f}%"
    )

    # --------------------------------------------------------
    # ROBOT ORIENTATION
    # --------------------------------------------------------

    print()
    print("ROBOT ORIENTATION")
    print("-" * 44)

    print(
        f"Final roll           : "
        f"{math.degrees(final_roll):.6f} deg"
    )

    print(
        f"Final pitch          : "
        f"{math.degrees(final_pitch):.6f} deg"
    )

    print(
        f"Final yaw            : "
        f"{math.degrees(final_yaw):.6f} deg"
    )

    print(
        f"Average roll         : "
        f"{average_roll:.6f} deg"
    )

    print(
        f"Average pitch        : "
        f"{average_pitch:.6f} deg"
    )

    print(
        f"Maximum |pitch|      : "
        f"{max(abs(x) for x in pitch_samples):.6f} deg"
    )

    print(
        f"Maximum |roll|       : "
        f"{max(abs(x) for x in roll_samples):.6f} deg"
    )

    # --------------------------------------------------------
    # CASTER POSITION
    # --------------------------------------------------------

    print()
    print("CASTER POSITION")
    print("-" * 44)

    print(
        f"Initial X            : "
        f"{caster_start_position[0]:.6f} m"
    )

    print(
        f"Initial Y            : "
        f"{caster_start_position[1]:.6f} m"
    )

    print(
        f"Initial Z            : "
        f"{caster_start_position[2]:.6f} m"
    )

    print(
        f"Final X              : "
        f"{caster_final_position[0]:.6f} m"
    )

    print(
        f"Final Y              : "
        f"{caster_final_position[1]:.6f} m"
    )

    print(
        f"Final Z              : "
        f"{caster_final_position[2]:.6f} m"
    )

    print(
        f"Average X            : "
        f"{average_caster_x:.6f} m"
    )

    print(
        f"Average Y            : "
        f"{average_caster_y:.6f} m"
    )

    print(
        f"Average Z            : "
        f"{average_caster_z:.6f} m"
    )

    # --------------------------------------------------------
    # CASTER VELOCITY
    # --------------------------------------------------------

    print()
    print("CASTER VELOCITY")
    print("-" * 44)

    print(
        f"Average Vx           : "
        f"{average_caster_vx:.6f} m/s"
    )

    print(
        f"Average Vy           : "
        f"{average_caster_vy:.6f} m/s"
    )

    print(
        f"Average Vz           : "
        f"{average_caster_vz:.6f} m/s"
    )

    print(
        f"Final Vx             : "
        f"{caster_final_velocity[0]:.6f} m/s"
    )

    print(
        f"Final Vy             : "
        f"{caster_final_velocity[1]:.6f} m/s"
    )

    print(
        f"Final Vz             : "
        f"{caster_final_velocity[2]:.6f} m/s"
    )

    # --------------------------------------------------------
    # CONTACTS
    # --------------------------------------------------------

    print()
    print("CONTACTS")
    print("-" * 44)

    print(
        f"Left wheel           : "
        f"{final_left_contacts}"
    )

    print(
        f"Right wheel          : "
        f"{final_right_contacts}"
    )

    print(
        f"Caster               : "
        f"{final_caster_contacts}"
    )

    # --------------------------------------------------------
    # NORMAL FORCES
    # --------------------------------------------------------

    print()
    print("AVERAGE NORMAL FORCES")
    print("-" * 44)

    print(
        f"Left wheel           : "
        f"{average_left_force:.6f} N"
    )

    print(
        f"Right wheel          : "
        f"{average_right_force:.6f} N"
    )

    print(
        f"Caster               : "
        f"{average_caster_force:.6f} N"
    )

    print(
        f"Total                : "
        f"{total_average_force:.6f} N"
    )

    print(
        f"Expected weight      : "
        f"{expected_weight:.6f} N"
    )

    print(
        f"Force error          : "
        f"{total_average_force - expected_weight:.6f} N"
    )

    # --------------------------------------------------------
    # CASTER CONTACT DETAIL
    # --------------------------------------------------------

    print()
    print("FINAL CASTER CONTACT DETAIL")
    print("-" * 44)

    if final_caster_contact_data:

        for i, contact in enumerate(
            final_caster_contact_data,
            start=1
        ):

            position = contact["position"]
            normal = contact["normal"]
            force = contact["force"]

            print(f"Contact {i}")

            print(
                f"  Position : "
                f"({position[0]:.6f}, "
                f"{position[1]:.6f}, "
                f"{position[2]:.6f})"
            )

            print(
                f"  Normal   : "
                f"({normal[0]:.6f}, "
                f"{normal[1]:.6f}, "
                f"{normal[2]:.6f})"
            )

            print(
                f"  Force    : "
                f"{force:.6f} N"
            )

    else:
        print("No caster-ground contact detected.")

    # --------------------------------------------------------
    # DIAGNOSTIC INTERPRETATION
    # --------------------------------------------------------

    print()
    print("=" * 44)
    print("V3.14 DIAGNOSTIC SUMMARY")
    print("=" * 44)

    if rotation_ratio >= 70:
        print(
            f"PASS — Yaw mechanics = "
            f"{rotation_ratio:.2f}%"
        )
    else:
        print(
            f"FAIL — Yaw mechanics = "
            f"{rotation_ratio:.2f}%"
        )

    if abs(final_pitch) < math.radians(5):
        print("PASS — Final pitch < 5 deg")
    else:
        print(
            f"WARNING — Final pitch = "
            f"{math.degrees(final_pitch):.2f} deg"
        )

    if abs(final_roll) < math.radians(5):
        print("PASS — Final roll < 5 deg")
    else:
        print(
            f"WARNING — Final roll = "
            f"{math.degrees(final_roll):.2f} deg"
        )

    if average_caster_force > 1.0:
        print(
            f"PASS — Caster carries load: "
            f"{average_caster_force:.3f} N"
        )
    else:
        print(
            f"WARNING — Caster carries very little "
            f"load: {average_caster_force:.3f} N"
        )

    print()
    print(
        "V3.14 is diagnostic only."
    )

    print(
        "No geometry or friction parameters "
        "were changed."
    )

    print(
        "Use these results to determine the "
        "V3.15 physical correction."
    )

    print()
    print("Press Ctrl+C to close PyBullet.")

    # --------------------------------------------------------
    # KEEP GUI OPEN
    # --------------------------------------------------------

    try:

        while True:

            p.stepSimulation()

            time.sleep(TIME_STEP)

    except KeyboardInterrupt:

        pass

    p.disconnect()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()