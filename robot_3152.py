import pybullet as p
import pybullet_data
import math


# ============================================================
# V3.15.2 ROBOT PARAMETERS
# ============================================================

# ---------- Drive system ----------
WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.04
WHEEL_MASS = 0.50
WHEEL_BASE = 0.30

# ---------- Chassis ----------
BASE_LENGTH = 0.40
BASE_WIDTH = 0.24
BASE_HEIGHT = 0.10
BASE_MASS = 5.0

# ---------- Caster ----------
CASTER_X = -0.15

CASTER_FORK_MASS = 0.05
CASTER_WHEEL_MASS = 0.15

CASTER_WHEEL_RADIUS = 0.04
CASTER_WHEEL_WIDTH = 0.025

# Distance between swivel axis and caster wheel axle
CASTER_TRAIL = 0.05

# ---------- Physics ----------
GRAVITY = -9.81
TIME_STEP = 1.0 / 240.0


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

    # Wheel collision
    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # Wheel visual
    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # ========================================================
    # ROBOT MULTIBODY
    # ========================================================

    robot_id = p.createMultiBody(
        baseMass=BASE_MASS,
        baseCollisionShapeIndex=chassis_collision,
        baseVisualShapeIndex=chassis_visual,

        basePosition=[
            0,
            0,
            BASE_HEIGHT / 2 + WHEEL_RADIUS
        ],

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

        linkPositions=[
            [0, +WHEEL_BASE / 2, -BASE_HEIGHT / 2],
            [0, -WHEEL_BASE / 2, -BASE_HEIGHT / 2]
        ],

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

    # ========================================================
    # COLLISION SETTINGS
    # ========================================================

    # Do not allow chassis and wheels to collide with each
    # other through their overlapping link geometry.
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

    # ========================================================
    # FRICTION
    # ========================================================

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.8,
        spinningFriction=0.02,
        rollingFriction=0.01
    )

    p.changeDynamics(
        robot_id,
        0,
        lateralFriction=0.8,
        spinningFriction=0.02,
        rollingFriction=0.01
    )

    p.changeDynamics(
        robot_id,
        1,
        lateralFriction=0.8,
        spinningFriction=0.02,
        rollingFriction=0.01
    )

    # ========================================================
    # CASTer SWIVEL FORK
    # ========================================================

    # The swivel axis is located at:
    #
    # x = -0.15
    # z = chassis bottom = 0.08
    #
    # The fork itself extends downward.

    fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.04
        ]
    )

    fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.04
        ]
    )

    fork_id = p.createMultiBody(
        baseMass=CASTER_FORK_MASS,

        baseCollisionShapeIndex=fork_collision,

        baseVisualShapeIndex=fork_visual,

        basePosition=[
            CASTER_X,
            0,
            0.04
        ]
    )

    # ========================================================
    # CASTER WHEEL
    # ========================================================

    caster_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        height=CASTER_WHEEL_WIDTH
    )

    caster_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        length=CASTER_WHEEL_WIDTH
    )

    caster_wheel_id = p.createMultiBody(
        baseMass=CASTER_WHEEL_MASS,

        baseCollisionShapeIndex=caster_collision,

        baseVisualShapeIndex=caster_visual,

        basePosition=[
            CASTER_X - CASTER_TRAIL,
            0,
            CASTER_WHEEL_RADIUS
        ],

        baseOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # ========================================================
    # SWIVEL CONSTRAINT
    # ========================================================

    # The fork is connected to the robot through a vertical
    # revolute axis.
    #
    # This allows the caster to swivel freely.

    swivel_constraint = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=-1,

        childBodyUniqueId=fork_id,
        childLinkIndex=-1,

        jointType=p.JOINT_REVOLUTE,

        jointAxis=[0, 0, 1],

        parentFramePosition=[
            CASTER_X,
            0,
            -BASE_HEIGHT / 2
        ],

        childFramePosition=[
            0,
            0,
            0.04
        ]
    )

    p.changeConstraint(
        swivel_constraint,
        maxForce=0
    )

    # ========================================================
    # CASTER WHEEL AXLE
    # ========================================================

    # The caster wheel rotates around the Y axis.

    wheel_constraint = p.createConstraint(
        parentBodyUniqueId=fork_id,
        parentLinkIndex=-1,

        childBodyUniqueId=caster_wheel_id,
        childLinkIndex=-1,

        jointType=p.JOINT_REVOLUTE,

        jointAxis=[0, 1, 0],

        parentFramePosition=[
            -CASTER_TRAIL,
            0,
            -0.04
        ],

        childFramePosition=[
            0,
            0,
            0
        ]
    )

    p.changeConstraint(
        wheel_constraint,
        maxForce=0
    )

    # ========================================================
    # CASTER FRICTION
    # ========================================================

    p.changeDynamics(
        fork_id,
        -1,
        lateralFriction=0.05,
        spinningFriction=0.001,
        rollingFriction=0.001
    )

    p.changeDynamics(
        caster_wheel_id,
        -1,
        lateralFriction=0.05,
        spinningFriction=0.001,
        rollingFriction=0.001
    )

    # ========================================================
    # STORE COMPONENT IDs
    # ========================================================

    # Useful for diagnostics while keeping the original
    # create_robot() interface unchanged.

    global CASTER_FORK_ID
    global CASTER_WHEEL_ID
    global SWIVEL_CONSTRAINT_ID
    global CASTER_WHEEL_CONSTRAINT_ID

    CASTER_FORK_ID = fork_id
    CASTER_WHEEL_ID = caster_wheel_id
    SWIVEL_CONSTRAINT_ID = swivel_constraint
    CASTER_WHEEL_CONSTRAINT_ID = wheel_constraint

    return robot_id


# ============================================================
# GET CASTER INFORMATION
# ============================================================

def get_caster_ids():

    return (
        CASTER_FORK_ID,
        CASTER_WHEEL_ID,
        SWIVEL_CONSTRAINT_ID,
        CASTER_WHEEL_CONSTRAINT_ID
    )