import pybullet as p
import pybullet_data
import time
import math


# ============================================================
# V3.13 PARAMETERS
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

# Rotation test
LEFT_WHEEL_COMMAND = -4.0
RIGHT_WHEEL_COMMAND = 4.0
MOTOR_FORCE = 20.0
ROTATION_TIME = 3.0


# ============================================================
# HELPERS
# ============================================================

def quat_to_euler(quaternion):
    return p.getEulerFromQuaternion(quaternion)


def print_orientation(orientation):
    roll, pitch, yaw = quat_to_euler(orientation)

    print(f"Roll  : {math.degrees(roll):.4f} deg")
    print(f"Pitch : {math.degrees(pitch):.4f} deg")
    print(f"Yaw   : {math.degrees(yaw):.4f} deg")


def get_normal_force(body_id, link_index):
    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=0,
        linkIndexA=link_index
    )

    total_force = 0.0

    for contact in contacts:
        total_force += contact[9]

    return total_force


def get_contact_count(body_id, link_index):
    contacts = p.getContactPoints(
        bodyA=body_id,
        bodyB=0,
        linkIndexA=link_index
    )

    return len(contacts)


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
    # LINK POSITIONS
    # --------------------------------------------------------

    wheel_positions = [
        [0, WHEEL_BASE / 2, -BASE_HEIGHT / 2],
        [0, -WHEEL_BASE / 2, -BASE_HEIGHT / 2]
    ]

    # --------------------------------------------------------
    # ROBOT MULTIBODY
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
        robot_id,
        robot_id,
        -1,
        0,
        enableCollision=0
    )

    p.setCollisionFilterPair(
        robot_id,
        robot_id,
        -1,
        1,
        enableCollision=0
    )

    p.setCollisionFilterPair(
        robot_id,
        robot_id,
        0,
        1,
        enableCollision=0
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

        # Disable default motors
        p.setJointMotorControl2(
            robot_id,
            joint,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # --------------------------------------------------------
    # PASSIVE CASTER
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
    # CONNECT CASTER TO CHASSIS
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
            0,
            0,
            0
        ]
    )

    return robot_id, caster_id, caster_constraint


# ============================================================
# MAIN
# ============================================================

def main():

    physics_client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

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
    print("=" * 40)
    print("V3.13 THREE-WHEEL ROBOT")
    print("PURE ROTATION PHYSICS VALIDATION")
    print("=" * 40)

    print()
    print("MODEL")
    print("-" * 40)
    print("Two-wheel differential drive")
    print("Real PyBullet rigid-body dynamics")
    print("Passive spherical caster")
    print("Opposite wheel velocity command")
    print("No PPO")

    print()
    print("=" * 40)
    print("GEOMETRY CHECK")
    print("=" * 40)

    print()
    print("CHASSIS")
    print("-" * 40)
    print(f"Length              : {BASE_LENGTH:.3f} m")
    print(f"Width               : {BASE_WIDTH:.3f} m")
    print(f"Height              : {BASE_HEIGHT:.3f} m")

    print()
    print("DRIVE WHEELS")
    print("-" * 40)
    print(f"Radius              : {WHEEL_RADIUS:.3f} m")
    print(f"Width               : {WHEEL_WIDTH:.3f} m")
    print(f"Wheel base          : {WHEEL_BASE:.3f} m")
    print(f"Left center Y       : {WHEEL_BASE / 2:.3f} m")
    print(f"Right center Y      : {-WHEEL_BASE / 2:.3f} m")

    print()
    print("CASTER")
    print("-" * 40)
    print(f"Radius              : {CASTER_RADIUS:.3f} m")
    print(f"X position          : {CASTER_X:.3f} m")

    print()
    print("VERTICAL GEOMETRY")
    print("-" * 40)
    print(f"Wheel center Z      : {WHEEL_RADIUS:.3f} m")
    print(f"Chassis center Z    : {0.13:.3f} m")
    print(f"Chassis bottom Z    : {0.08:.3f} m")
    print(f"Caster center Z     : {CASTER_RADIUS:.3f} m")

    clearance = (
        WHEEL_BASE / 2
        - WHEEL_WIDTH / 2
        - BASE_WIDTH / 2
    )

    print()
    print("GEOMETRY VALIDATION")
    print("-" * 40)

    if clearance > 0:
        print(
            f"PASS — Wheel/chassis clearance = "
            f"{clearance:.3f} m"
        )
    else:
        print(
            f"FAIL — Wheel/chassis clearance = "
            f"{clearance:.3f} m"
        )

    if abs(0.08 - 0.08) < 1e-6:
        print(
            "PASS — Chassis bottom equals "
            "wheel center height"
        )

    if CASTER_RADIUS < WHEEL_RADIUS:
        print(
            "PASS — Caster smaller than drive wheels"
        )

    print()
    print(f"Robot ID          : {robot_id}")
    print(f"Caster ID         : {caster_id}")
    print(
        f"Number of joints  : "
        f"{p.getNumJoints(robot_id)}"
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    base_position, base_orientation = \
        p.getBasePositionAndOrientation(robot_id)

    print()
    print("=" * 40)
    print("INITIAL STATE")
    print("=" * 40)

    print()
    print("BASE")
    print("-" * 40)

    print(
        f"Position: "
        f"({base_position[0]:.6f}, "
        f"{base_position[1]:.6f}, "
        f"{base_position[2]:.6f})"
    )

    print_orientation(base_orientation)

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
    print("=" * 40)
    print("SETTLED STATE")
    print("=" * 40)

    print(
        f"Position : "
        f"({start_position[0]:.6f}, "
        f"{start_position[1]:.6f}, "
        f"{start_position[2]:.6f})"
    )

    print(
        f"Roll     : {math.degrees(start_roll):.4f} deg"
    )

    print(
        f"Pitch    : {math.degrees(start_pitch):.4f} deg"
    )

    print(
        f"Yaw      : {math.degrees(start_yaw):.4f} deg"
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

    theoretical_yaw_change = (
        theoretical_yaw_rate * ROTATION_TIME
    )

    print()
    print("=" * 40)
    print("V3.13 PURE ROTATION TEST")
    print("=" * 40)

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

    print(
        f"Theoretical yaw change: "
        f"{theoretical_yaw_change:.4f} rad"
    )

    # --------------------------------------------------------
    # APPLY MOTOR COMMAND
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
    # ROTATION SIMULATION
    # --------------------------------------------------------

    rotation_steps = int(
        ROTATION_TIME / TIME_STEP
    )

    yaw_rate_samples = []
    left_velocity_samples = []
    right_velocity_samples = []

    for _ in range(rotation_steps):

        p.stepSimulation()

        # Robot angular velocity
        _, angular_velocity = \
            p.getBaseVelocity(robot_id)

        yaw_rate_samples.append(
            angular_velocity[2]
        )

        # Wheel velocities
        left_state = p.getJointState(
            robot_id,
            0
        )

        right_state = p.getJointState(
            robot_id,
            1
        )

        left_velocity_samples.append(
            left_state[1]
        )

        right_velocity_samples.append(
            right_state[1]
        )

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

    avg_yaw_rate = (
        sum(yaw_rate_samples)
        / len(yaw_rate_samples)
    )

    avg_left_velocity = (
        sum(left_velocity_samples)
        / len(left_velocity_samples)
    )

    avg_right_velocity = (
        sum(right_velocity_samples)
        / len(right_velocity_samples)
    )

    final_left_velocity = \
        left_velocity_samples[-1]

    final_right_velocity = \
        right_velocity_samples[-1]

    actual_yaw_rate = final_angular_velocity[2]

    yaw_rate_ratio = (
        abs(avg_yaw_rate)
        / abs(theoretical_yaw_rate)
        * 100.0
    )

    # --------------------------------------------------------
    # CONTACTS
    # --------------------------------------------------------

    left_contacts = get_contact_count(
        robot_id,
        0
    )

    right_contacts = get_contact_count(
        robot_id,
        1
    )

    caster_contacts = len(
        p.getContactPoints(
            bodyA=caster_id,
            bodyB=plane_id
        )
    )

    left_force = get_normal_force(
        robot_id,
        0
    )

    right_force = get_normal_force(
        robot_id,
        1
    )

    caster_force = 0.0

    caster_contact_points = p.getContactPoints(
        bodyA=caster_id,
        bodyB=plane_id
    )

    for contact in caster_contact_points:
        caster_force += contact[9]

    total_force = (
        left_force
        + right_force
        + caster_force
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
    print("=" * 40)
    print("V3.13 FINAL STATE")
    print("=" * 40)

    print()
    print("POSITION")
    print("-" * 40)

    print(
        f"Start X           : "
        f"{start_position[0]:.6f}"
    )

    print(
        f"Start Y           : "
        f"{start_position[1]:.6f}"
    )

    print(
        f"Final X           : "
        f"{final_position[0]:.6f}"
    )

    print(
        f"Final Y           : "
        f"{final_position[1]:.6f}"
    )

    print(
        f"X displacement    : "
        f"{final_position[0] - start_position[0]:.6f} m"
    )

    print(
        f"Y displacement    : "
        f"{final_position[1] - start_position[1]:.6f} m"
    )

    print()
    print("WHEEL VELOCITIES")
    print("-" * 40)

    print(
        f"Average left      : "
        f"{avg_left_velocity:.6f} rad/s"
    )

    print(
        f"Average right     : "
        f"{avg_right_velocity:.6f} rad/s"
    )

    print(
        f"Final left        : "
        f"{final_left_velocity:.6f} rad/s"
    )

    print(
        f"Final right       : "
        f"{final_right_velocity:.6f} rad/s"
    )

    print()
    print("ROBOT VELOCITY")
    print("-" * 40)

    print(
        f"Linear velocity   : "
        f"({final_linear_velocity[0]:.6f}, "
        f"{final_linear_velocity[1]:.6f}, "
        f"{final_linear_velocity[2]:.6f})"
    )

    print(
        f"Angular velocity  : "
        f"({final_angular_velocity[0]:.6f}, "
        f"{final_angular_velocity[1]:.6f}, "
        f"{final_angular_velocity[2]:.6f})"
    )

    print()
    print("ROTATION")
    print("-" * 40)

    print(
        f"Theoretical yaw rate : "
        f"{theoretical_yaw_rate:.6f} rad/s"
    )

    print(
        f"Average yaw rate     : "
        f"{avg_yaw_rate:.6f} rad/s"
    )

    print(
        f"Final yaw rate       : "
        f"{actual_yaw_rate:.6f} rad/s"
    )

    print(
        f"Rotation ratio       : "
        f"{yaw_rate_ratio:.2f}%"
    )

    print()
    print("ORIENTATION")
    print("-" * 40)

    print(
        f"Roll              : "
        f"{math.degrees(final_roll):.6f} deg"
    )

    print(
        f"Pitch             : "
        f"{math.degrees(final_pitch):.6f} deg"
    )

    print(
        f"Yaw               : "
        f"{math.degrees(final_yaw):.6f} deg"
    )

    print()
    print("CONTACTS")
    print("-" * 40)

    print(
        f"Left wheel        : "
        f"{left_contacts}"
    )

    print(
        f"Right wheel       : "
        f"{right_contacts}"
    )

    print(
        f"Caster            : "
        f"{caster_contacts}"
    )

    print()
    print("NORMAL FORCES")
    print("-" * 40)

    print(
        f"Left wheel        : "
        f"{left_force:.6f} N"
    )

    print(
        f"Right wheel       : "
        f"{right_force:.6f} N"
    )

    print(
        f"Caster            : "
        f"{caster_force:.6f} N"
    )

    print(
        f"Total             : "
        f"{total_force:.6f} N"
    )

    print(
        f"Expected weight   : "
        f"{expected_weight:.6f} N"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    wheel_tolerance = 0.25

    left_wheel_pass = (
        abs(avg_left_velocity - LEFT_WHEEL_COMMAND)
        <= abs(LEFT_WHEEL_COMMAND) * wheel_tolerance
    )

    right_wheel_pass = (
        abs(avg_right_velocity - RIGHT_WHEEL_COMMAND)
        <= abs(RIGHT_WHEEL_COMMAND) * wheel_tolerance
    )

    rotation_pass = (
        yaw_rate_ratio >= 70.0
    )

    roll_pass = (
        abs(math.degrees(final_roll)) < 5.0
    )

    pitch_pass = (
        abs(math.degrees(final_pitch)) < 5.0
    )

    wheel_contact_pass = (
        left_contacts > 0
        and right_contacts > 0
    )

    caster_contact_pass = (
        caster_contacts > 0
    )

    force_balance_pass = (
        abs(total_force - expected_weight) < 0.5
    )

    # --------------------------------------------------------
    # VALIDATION OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 40)
    print("V3.13 VALIDATION")
    print("=" * 40)

    if left_wheel_pass:
        print("PASS — Left wheel velocity")
    else:
        print("FAIL — Left wheel velocity")

    if right_wheel_pass:
        print("PASS — Right wheel velocity")
    else:
        print("FAIL — Right wheel velocity")

    if rotation_pass:
        print(
            f"PASS — Rotation ratio = "
            f"{yaw_rate_ratio:.2f}%"
        )
    else:
        print(
            f"FAIL — Rotation ratio = "
            f"{yaw_rate_ratio:.2f}%"
        )

    if roll_pass:
        print("PASS — Roll stability")
    else:
        print("FAIL — Roll stability")

    if pitch_pass:
        print("PASS — Pitch stability")
    else:
        print("FAIL — Pitch stability")

    if wheel_contact_pass:
        print("PASS — Drive wheel contact")
    else:
        print("FAIL — Drive wheel contact")

    if caster_contact_pass:
        print("PASS — Caster contact")
    else:
        print("FAIL — Caster contact")

    if force_balance_pass:
        print("PASS — Weight balance")
    else:
        print("FAIL — Weight balance")

    overall_pass = (
        left_wheel_pass
        and right_wheel_pass
        and rotation_pass
        and roll_pass
        and pitch_pass
        and wheel_contact_pass
        and caster_contact_pass
        and force_balance_pass
    )

    print()

    if overall_pass:
        print("=" * 40)
        print("V3.13 PURE ROTATION PHYSICS: PASS")
        print("=" * 40)
    else:
        print("=" * 40)
        print("V3.13 PURE ROTATION PHYSICS: DIAGNOSTIC")
        print("=" * 40)

    print()
    print(
        "The robot was driven using actual "
        "PyBullet wheel joints."
    )

    print(
        "No position was manually controlled."
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


if __name__ == "__main__":
    main()