import math

import pybullet as p


# ============================================================
# V3 — REAL-PHYSICS DIFFERENTIAL-DRIVE ROBOT
#
# Robot:
#   - rigid chassis
#   - two powered drive wheels
#   - passive swivel caster
#   - passive caster wheel
#
# Drive wheels:
#   cylinder axis = Y
#   revolute joint axis = Y
#
# Caster:
#   swivel axis = Z
#   wheel axle = Y
#
# This file contains the physical robot model only.
#
# No Gymnasium
# No PPO
# ============================================================


# ============================================================
# ROBOT PARAMETERS
# ============================================================

WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.04
WHEEL_BASE = 0.30

CHASSIS_LENGTH = 0.40
CHASSIS_WIDTH = 0.24
CHASSIS_HEIGHT = 0.10

CHASSIS_MASS = 5.0
WHEEL_MASS = 0.5

CASTER_FORK_MASS = 0.05
CASTER_WHEEL_MASS = 0.15

CASTER_X = -0.15
CASTER_TRAIL = 0.05

CASTER_WHEEL_RADIUS = 0.04
CASTER_WHEEL_WIDTH = 0.025

MOTOR_FORCE = 20.0

TIME_STEP = 1.0 / 240.0
GRAVITY = -9.81


# ============================================================
# DERIVED GEOMETRY
# ============================================================

# Drive-wheel center height.
WHEEL_CENTER_Z = WHEEL_RADIUS

# Chassis center height.
#
# The bottom of the chassis is at the wheel-center height.
#
# Therefore:
#
# chassis center =
#     wheel radius + chassis height / 2
#
#             = 0.08 + 0.05
#             = 0.13 m
#
CHASSIS_CENTER_Z = (
    WHEEL_CENTER_Z +
    CHASSIS_HEIGHT / 2.0
)


# ============================================================
# JOINT INDICES
# ============================================================

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1
CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3


# ============================================================
# CREATE ROBOT
# ============================================================

def create_robot():

    # ========================================================
    # CHASSIS
    # ========================================================

    chassis_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            CHASSIS_LENGTH / 2.0,
            CHASSIS_WIDTH / 2.0,
            CHASSIS_HEIGHT / 2.0,
        ],
    )

    chassis_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            CHASSIS_LENGTH / 2.0,
            CHASSIS_WIDTH / 2.0,
            CHASSIS_HEIGHT / 2.0,
        ],
    )

    # ========================================================
    # DRIVE WHEELS
    # ========================================================
    #
    # PyBullet cylinder default axis = Z.
    #
    # Rotate cylinder around X by +90 degrees:
    #
    #     cylinder axis = Y
    #     wheel axle    = Y
    #
    # ========================================================

    wheel_rotation = p.getQuaternionFromEuler(
        [math.pi / 2.0, 0.0, 0.0]
    )

    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
        collisionFrameOrientation=wheel_rotation,
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
        visualFrameOrientation=wheel_rotation,
    )

    # --------------------------------------------------------
    # Drive wheel positions
    #
    # Forward = +X
    # Left    = +Y
    # Right   = -Y
    # --------------------------------------------------------

    left_wheel_position = [
        0.0,
        +WHEEL_BASE / 2.0,
        -CHASSIS_HEIGHT / 2.0,
    ]

    right_wheel_position = [
        0.0,
        -WHEEL_BASE / 2.0,
        -CHASSIS_HEIGHT / 2.0,
    ]

    # ========================================================
    # CASTER FORK
    # ========================================================

    caster_fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.020,
        ],
    )

    caster_fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.020,
        ],
    )

    # ========================================================
    # CASTER WHEEL
    # ========================================================
    #
    # Default cylinder axis = Z.
    #
    # Rotate around X by +90 degrees:
    #
    #     caster wheel axle = Y
    #
    # ========================================================

    caster_wheel_rotation = p.getQuaternionFromEuler(
        [math.pi / 2.0, 0.0, 0.0]
    )

    caster_wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        height=CASTER_WHEEL_WIDTH,
        collisionFrameOrientation=caster_wheel_rotation,
    )

    caster_wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        length=CASTER_WHEEL_WIDTH,
        visualFrameOrientation=caster_wheel_rotation,
    )

    # ========================================================
    # LINK POSITIONS
    # ========================================================

    link_positions = [

        # Link 0:
        # Left drive wheel
        left_wheel_position,

        # Link 1:
        # Right drive wheel
        right_wheel_position,

        # Link 2:
        # Caster swivel fork
        [
            CASTER_X,
            0.0,
            -0.05,
        ],

        # Link 3:
        # Caster wheel relative to caster fork
        [
            -CASTER_TRAIL,
            0.0,
            -0.04,
        ],
    ]

    # ========================================================
    # MULTIBODY CONSTRUCTION
    # ========================================================

    print()
    print("===== CREATE MULTIBODY INPUT =====")

    print("linkMasses:")
    print([
        WHEEL_MASS,
        WHEEL_MASS,
        CASTER_FORK_MASS,
        CASTER_WHEEL_MASS,
    ])

    print()
    print("linkPositions:")
    print(link_positions)

    print()
    print("linkParentIndices:")
    print([
        0,
        0,
        0,
        3,
    ])

    print()
    print("linkJointTypes:")
    print([
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
    ])

    print()
    print("linkJointAxis:")
    print([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    print()
    print("Expected hierarchy:")
    print("  Link 0 = LEFT WHEEL  -> BASE")
    print("  Link 1 = RIGHT WHEEL -> BASE")
    print("  Link 2 = CASTER FORK -> BASE")
    print("  Link 3 = CASTER WHEEL -> CASTER FORK")

    print()
    print("Expected axes:")
    print("  Link 0 = Y")
    print("  Link 1 = Y")
    print("  Link 2 = Z")
    print("  Link 3 = Y")

    print()
    print(
        f"Expected chassis center Z: "
        f"{CHASSIS_CENTER_Z:.3f} m"
    )

    print("==================================")
    print()

    robot_id = p.createMultiBody(

        # ----------------------------------------------------
        # BASE
        # ----------------------------------------------------

        baseMass=CHASSIS_MASS,

        baseCollisionShapeIndex=chassis_collision,

        baseVisualShapeIndex=chassis_visual,

        basePosition=[
            0.0,
            0.0,
            CHASSIS_CENTER_Z,
        ],

        baseOrientation=[
            0.0,
            0.0,
            0.0,
            1.0,
        ],

        # ----------------------------------------------------
        # LINKS
        # ----------------------------------------------------

        linkMasses=[
            WHEEL_MASS,
            WHEEL_MASS,
            CASTER_FORK_MASS,
            CASTER_WHEEL_MASS,
        ],

        linkCollisionShapeIndices=[
            wheel_collision,
            wheel_collision,
            caster_fork_collision,
            caster_wheel_collision,
        ],

        linkVisualShapeIndices=[
            wheel_visual,
            wheel_visual,
            caster_fork_visual,
            caster_wheel_visual,
        ],

        linkPositions=link_positions,

        linkOrientations=[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],

        # ----------------------------------------------------
        # INERTIAL FRAME POSITIONS
        # ----------------------------------------------------

        linkInertialFramePositions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],

        # ----------------------------------------------------
        # INERTIAL FRAME ORIENTATIONS
        # ----------------------------------------------------

        linkInertialFrameOrientations=[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],

        # ----------------------------------------------------
        # PARENT HIERARCHY
        #
        # PyBullet createMultiBody uses:
        #
        #     0 = BASE
        #
        # Therefore:
        #
        #     Link 0 -> 0 = BASE
        #     Link 1 -> 0 = BASE
        #     Link 2 -> 0 = BASE
        #     Link 3 -> 3 = Link 2
        #
        # The last value is intentionally 3.
        # ----------------------------------------------------

        linkParentIndices=[
            0,
            0,
            0,
            3,
        ],

        # ----------------------------------------------------
        # JOINT TYPES
        # ----------------------------------------------------

        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
        ],

        # ----------------------------------------------------
        # JOINT AXES
        #
        # Left wheel  = Y
        # Right wheel = Y
        # Caster      = Z
        # Caster axle = Y
        # ----------------------------------------------------

        linkJointAxis=[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
    )

    # ========================================================
    # DYNAMICS
    # ========================================================

    # --------------------------------------------------------
    # Chassis
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.05,
        rollingFriction=0.0,
        spinningFriction=0.0,
    )

    # --------------------------------------------------------
    # Drive wheels
    # --------------------------------------------------------

    print(
        "DEBUG: Number of joints =",
        p.getNumJoints(robot_id),
    )

    print(
        "DEBUG: LEFT_WHEEL_JOINT =",
        LEFT_WHEEL_JOINT,
    )

    print(
        "DEBUG: RIGHT_WHEEL_JOINT =",
        RIGHT_WHEEL_JOINT,
    )

    for joint_index in [
        LEFT_WHEEL_JOINT,
        RIGHT_WHEEL_JOINT,
    ]:

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0.0,
            force=0.0,
        )

        p.changeDynamics(
            robot_id,
            joint_index,
            lateralFriction=0.5,
            rollingFriction=0.0,
            spinningFriction=0.0,
        )

    # --------------------------------------------------------
    # Passive caster swivel
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        CASTER_SWIVEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0,
    )

    # --------------------------------------------------------
    # Passive caster wheel
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        CASTER_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0.0,
        spinningFriction=0.0,
    )

    # ========================================================
    # DISABLE SELF COLLISIONS
    # ========================================================

    collision_pairs = [
        (-1, LEFT_WHEEL_JOINT),
        (-1, RIGHT_WHEEL_JOINT),
        (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT),
        (-1, CASTER_SWIVEL_JOINT),
        (-1, CASTER_WHEEL_JOINT),
        (CASTER_SWIVEL_JOINT, CASTER_WHEEL_JOINT),
    ]

    for link_a, link_b in collision_pairs:

        p.setCollisionFilterPair(
            robot_id,
            robot_id,
            link_a,
            link_b,
            enableCollision=0,
        )

    return robot_id


# ============================================================
# SET DRIVE-WHEEL VELOCITIES
# ============================================================

def set_wheel_velocity(
    robot_id,
    left_velocity,
    right_velocity,
):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=LEFT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=left_velocity,
        force=MOTOR_FORCE,
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=RIGHT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=right_velocity,
        force=MOTOR_FORCE,
    )


# ============================================================
# STOP DRIVE WHEELS
# ============================================================

def stop_robot(robot_id):

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=LEFT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=RIGHT_WHEEL_JOINT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )