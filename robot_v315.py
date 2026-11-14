import pybullet as p
import pybullet_data
import math


# ============================================================
# V3.15.1 PARAMETERS
# ============================================================

# ----------------------------
# Drive wheels
# ----------------------------
WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.04
WHEEL_MASS = 0.50

# Differential-drive geometry
WHEEL_BASE = 0.30

# ----------------------------
# Chassis
# ----------------------------
BASE_LENGTH = 0.40
BASE_WIDTH = 0.24
BASE_HEIGHT = 0.10
BASE_MASS = 5.0

BASE_Z = 0.13

# ----------------------------
# Passive caster
# ----------------------------
CASTER_WHEEL_RADIUS = 0.04
CASTER_WHEEL_WIDTH = 0.025

CASTER_TOTAL_MASS = 0.20
CASTER_FORK_MASS = 0.05
CASTER_WHEEL_MASS = (
    CASTER_TOTAL_MASS - CASTER_FORK_MASS
)

# Position of caster swivel axis
CASTER_X = -0.15

# Horizontal trail from swivel axis to wheel center
CASTER_TRAIL = 0.05

# ----------------------------
# Physics
# ----------------------------
GRAVITY = -9.81
TIME_STEP = 1.0 / 240.0

# ----------------------------
# Drive motor
# ----------------------------
TARGET_WHEEL_VELOCITY = 4.0
MOTOR_FORCE = 20.0


# ============================================================
# CREATE ROBOT
# ============================================================

def create_robot():

    # ========================================================
    # CHASSIS
    # ========================================================

    base_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            BASE_LENGTH / 2,
            BASE_WIDTH / 2,
            BASE_HEIGHT / 2
        ]
    )

    base_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            BASE_LENGTH / 2,
            BASE_WIDTH / 2,
            BASE_HEIGHT / 2
        ]
    )

    # Chassis center:
    # bottom = 0.08 m
    # top    = 0.18 m
    base_position = [
        0,
        0,
        BASE_Z
    ]

    # ========================================================
    # DRIVE WHEEL
    # ========================================================

    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH
    )

    # Cylinder axis Z -> rotate to Y
    wheel_orientation = p.getQuaternionFromEuler(
        [math.pi / 2, 0, 0]
    )

    # ========================================================
    # CASTER FORK
    # ========================================================

    fork_radius = 0.018
    fork_height = 0.06

    fork_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=fork_radius,
        height=fork_height
    )

    fork_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=fork_radius,
        length=fork_height
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

    caster_wheel_orientation = p.getQuaternionFromEuler(
        [math.pi / 2, 0, 0]
    )

    # ========================================================
    # IMPORTANT V3.15.1 GEOMETRY
    # ========================================================

    # Ground = z = 0
    #
    # Caster wheel center must be:
    #
    # z = caster radius
    #
    # therefore:
    #
    # wheel center z = 0.04 m
    #
    # The caster fork/swivel axis is positioned above
    # the wheel center.

    CASTER_WHEEL_Z = CASTER_WHEEL_RADIUS

    # Swivel axis is at chassis bottom.
    CASTER_SWIVEL_Z = (
        -BASE_HEIGHT / 2
    )

    # Relative wheel position from swivel axis
    CASTER_WHEEL_RELATIVE_Z = (
        CASTER_WHEEL_Z
        - (BASE_Z + CASTER_SWIVEL_Z)
    )

    # The above expression is only used to calculate the
    # physical world-relative geometry.
    #
    # In the multi-body hierarchy the caster fork is attached
    # to the chassis, so the caster wheel position relative to
    # the fork is explicitly set below.

    # ========================================================
    # MULTI-BODY
    # ========================================================

    robot_id = p.createMultiBody(

        baseMass=BASE_MASS,

        baseCollisionShapeIndex=base_collision,
        baseVisualShapeIndex=base_visual,

        basePosition=base_position,
        baseOrientation=[0, 0, 0, 1],

        # ----------------------------------------------------
        # LINK MASSES
        # ----------------------------------------------------

        linkMasses=[
            WHEEL_MASS,          # 0 left wheel
            WHEEL_MASS,          # 1 right wheel
            CASTER_FORK_MASS,    # 2 caster fork
            CASTER_WHEEL_MASS    # 3 caster wheel
        ],

        # ----------------------------------------------------
        # LINK COLLISION
        # ----------------------------------------------------

        linkCollisionShapeIndices=[
            wheel_collision,
            wheel_collision,
            fork_collision,
            caster_collision
        ],

        # ----------------------------------------------------
        # LINK VISUAL
        # ----------------------------------------------------

        linkVisualShapeIndices=[
            wheel_visual,
            wheel_visual,
            fork_visual,
            caster_visual
        ],

        # ----------------------------------------------------
        # LINK POSITIONS
        # ----------------------------------------------------

        linkPositions=[

            # Left drive wheel
            [
                0,
                +WHEEL_BASE / 2,
                0.08 - BASE_Z
            ],

            # Right drive wheel
            [
                0,
                -WHEEL_BASE / 2,
                0.08 - BASE_Z
            ],

            # Caster swivel axis
            [
                CASTER_X,
                0,
                -BASE_HEIGHT / 2
            ],

            # Caster wheel
            #
            # The wheel center is exactly 0.04 m above
            # the ground in the initial configuration.
            #
            # Relative to the fork:
            #
            # swivel axis z = 0.08 m
            # wheel center  = 0.04 m
            #
            # therefore relative z = -0.04 m
            [
                -CASTER_TRAIL,
                0,
                -0.04
            ]
        ],

        # ----------------------------------------------------
        # LINK ORIENTATIONS
        # ----------------------------------------------------

        linkOrientations=[
            wheel_orientation,
            wheel_orientation,
            [0, 0, 0, 1],
            caster_wheel_orientation
        ],

        # ----------------------------------------------------
        # PARENT LINKS
        # ----------------------------------------------------

        linkParentIndices=[
            0,
            0,
            0,
            2
        ],

        # ----------------------------------------------------
        # JOINT TYPES
        # ----------------------------------------------------

        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE
        ],

        # ----------------------------------------------------
        # JOINT AXES
        # ----------------------------------------------------

        linkJointAxis=[
            # Left drive wheel
            [0, 1, 0],

            # Right drive wheel
            [0, 1, 0],

            # Caster swivel
            [0, 0, 1],

            # Caster wheel axle
            [0, 1, 0]
        ],

        # ----------------------------------------------------
        # INERTIAL FRAMES
        # ----------------------------------------------------

        linkInertialFramePositions=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],

        linkInertialFrameOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
    )

    # ========================================================
    # DRIVE WHEEL FRICTION
    # ========================================================

    p.changeDynamics(
        robot_id,
        0,
        lateralFriction=0.8,
        spinningFriction=0.01,
        rollingFriction=0.001
    )

    p.changeDynamics(
        robot_id,
        1,
        lateralFriction=0.8,
        spinningFriction=0.01,
        rollingFriction=0.001
    )

    # ========================================================
    # CASTER FORK FRICTION
    # ========================================================

    p.changeDynamics(
        robot_id,
        2,
        lateralFriction=0.05,
        spinningFriction=0.001,
        rollingFriction=0.001
    )

    # ========================================================
    # CASTER WHEEL FRICTION
    # ========================================================

    p.changeDynamics(
        robot_id,
        3,
        lateralFriction=0.20,
        spinningFriction=0.001,
        rollingFriction=0.001
    )

    # ========================================================
    # CHASSIS
    # ========================================================

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.5
    )

    # ========================================================
    # DISABLE CHASSIS / DRIVE-WHEEL COLLISION
    # ========================================================

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

    # ========================================================
    # PASSIVE CASTER SWIVEL
    # ========================================================

    p.setJointMotorControl2(
        robot_id,
        2,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    # ========================================================
    # PASSIVE CASTER WHEEL
    # ========================================================

    p.setJointMotorControl2(
        robot_id,
        3,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    return robot_id


# ============================================================
# WORLD
# ============================================================

def create_world():

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        GRAVITY
    )

    p.setTimeStep(
        TIME_STEP
    )

    p.setPhysicsEngineParameter(
        numSolverIterations=100
    )

    plane_id = p.loadURDF(
        "plane.urdf"
    )

    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=0.8,
        spinningFriction=0.01,
        rollingFriction=0.001
    )

    return plane_id


# ============================================================
# DRIVE
# ============================================================

def set_drive_velocity(
    robot_id,
    left_velocity,
    right_velocity
):

    p.setJointMotorControl2(
        robot_id,
        0,
        p.VELOCITY_CONTROL,
        targetVelocity=left_velocity,
        force=MOTOR_FORCE
    )

    p.setJointMotorControl2(
        robot_id,
        1,
        p.VELOCITY_CONTROL,
        targetVelocity=right_velocity,
        force=MOTOR_FORCE
    )


# ============================================================
# STOP
# ============================================================

def stop_drive(robot_id):

    p.setJointMotorControl2(
        robot_id,
        0,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )

    p.setJointMotorControl2(
        robot_id,
        1,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )


# ============================================================
# ROBOT STATE
# ============================================================

def get_robot_state(robot_id):

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    linear_velocity, angular_velocity = (
        p.getBaseVelocity(
            robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return {
        "position": position,
        "orientation": orientation,
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity
    }


# ============================================================
# JOINT STATES
# ============================================================

def get_joint_states(robot_id):

    states = p.getJointStates(
        robot_id,
        [0, 1, 2, 3]
    )

    return {
        "left_wheel_velocity": states[0][1],
        "right_wheel_velocity": states[1][1],
        "caster_swivel_velocity": states[2][1],
        "caster_wheel_velocity": states[3][1]
    }


# ============================================================
# CONTACTS
# ============================================================

def get_contacts(
    robot_id,
    plane_id
):

    return {

        "left": p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=0
        ),

        "right": p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=1
        ),

        "caster_fork": p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=2
        ),

        "caster_wheel": p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=3
        )
    }


# ============================================================
# NORMAL FORCE
# ============================================================

def get_normal_force(
    contact_list
):

    return sum(
        contact[9]
        for contact in contact_list
    )


# ============================================================
# STATIC VALIDATION
# ============================================================

if __name__ == "__main__":

    physics_client = p.connect(
        p.GUI
    )

    plane_id = create_world()

    robot_id = create_robot()

    print(
        "=============================================="
    )
    print(
        "V3.15.1 STATIC VALIDATION"
    )
    print(
        "=============================================="
    )

    print("\nSettling robot...")

    for _ in range(
        int(3.0 / TIME_STEP)
    ):
        p.stepSimulation()

    state = get_robot_state(
        robot_id
    )

    print("\nSETTLED STATE")
    print("----------------------------------------------")

    print(
        f"Position: "
        f"({state['position'][0]:.6f}, "
        f"{state['position'][1]:.6f}, "
        f"{state['position'][2]:.6f})"
    )

    print(
        f"Roll:  "
        f"{state['roll']:.4f} deg"
    )

    print(
        f"Pitch: "
        f"{state['pitch']:.4f} deg"
    )

    print(
        f"Yaw:   "
        f"{state['yaw']:.4f} deg"
    )

    # --------------------------------------------------------
    # Contacts
    # --------------------------------------------------------

    contacts = get_contacts(
        robot_id,
        plane_id
    )

    left_force = get_normal_force(
        contacts["left"]
    )

    right_force = get_normal_force(
        contacts["right"]
    )

    caster_force = get_normal_force(
        contacts["caster_wheel"]
    )

    total_force = (
        left_force
        + right_force
        + caster_force
    )

    expected_force = (
        BASE_MASS
        + WHEEL_MASS
        + WHEEL_MASS
        + CASTER_TOTAL_MASS
    ) * abs(GRAVITY)

    print("\nCONTACTS")
    print("----------------------------------------------")

    print(
        f"Left: "
        f"{len(contacts['left'])}"
    )

    print(
        f"Right: "
        f"{len(contacts['right'])}"
    )

    print(
        f"Caster fork: "
        f"{len(contacts['caster_fork'])}"
    )

    print(
        f"Caster wheel: "
        f"{len(contacts['caster_wheel'])}"
    )

    print("\nNORMAL FORCES")
    print("----------------------------------------------")

    print(
        f"Left:   "
        f"{left_force:.4f} N"
    )

    print(
        f"Right:  "
        f"{right_force:.4f} N"
    )

    print(
        f"Caster: "
        f"{caster_force:.4f} N"
    )

    print(
        f"Total:  "
        f"{total_force:.4f} N"
    )

    print(
        f"Expected: "
        f"{expected_force:.4f} N"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    roll_pass = abs(state["roll"]) < 2.0
    pitch_pass = abs(state["pitch"]) < 2.0
    yaw_pass = abs(state["yaw"]) < 2.0

    left_contact_pass = (
        len(contacts["left"]) > 0
    )

    right_contact_pass = (
        len(contacts["right"]) > 0
    )

    caster_contact_pass = (
        len(contacts["caster_wheel"]) > 0
    )

    caster_load_pass = (
        caster_force > 1.0
    )

    force_pass = (
        abs(total_force - expected_force)
        < 2.0
    )

    print("\nVALIDATION")
    print("----------------------------------------------")

    print(
        f"Roll: "
        f"{'PASS' if roll_pass else 'FAIL'}"
    )

    print(
        f"Pitch: "
        f"{'PASS' if pitch_pass else 'FAIL'}"
    )

    print(
        f"Yaw: "
        f"{'PASS' if yaw_pass else 'FAIL'}"
    )

    print(
        f"Left wheel contact: "
        f"{'PASS' if left_contact_pass else 'FAIL'}"
    )

    print(
        f"Right wheel contact: "
        f"{'PASS' if right_contact_pass else 'FAIL'}"
    )

    print(
        f"Caster contact: "
        f"{'PASS' if caster_contact_pass else 'FAIL'}"
    )

    print(
        f"Caster load: "
        f"{'PASS' if caster_load_pass else 'FAIL'}"
    )

    print(
        f"Weight balance: "
        f"{'PASS' if force_pass else 'FAIL'}"
    )

    print(
        "\n=============================================="
    )
    print(
        "END V3.15.1"
    )
    print(
        "=============================================="
    )

    input(
        "\nPress ENTER to close..."
    )

    p.disconnect()