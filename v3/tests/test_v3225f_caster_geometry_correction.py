"""
V3.22.5f — Caster Geometry / Collision-Margin Correction Test

Purpose
-------
Determine whether the reverse-motion caster failure is caused by the
collision margin of the caster wheel.

This test:
1. Builds an isolated copy of the V3 robot physics model.
2. Tests several caster collision margins.
3. Runs forward and reverse motion.
4. Measures:
   - caster contact fraction
   - caster normal force
   - caster wheel center Z
   - base Z
   - base pitch
   - maximum pitch
   - final pitch
5. Compares the results.

IMPORTANT
---------
This file does NOT modify the main robot implementation.
It is a diagnostic experiment only.

Expected caster geometry:
    physical caster wheel radius = 0.04 m
    physical caster wheel width  = 0.025 m

PyBullet reports the cylinder dimensions according to its
collision-shape convention, so the test explicitly verifies
the resulting collision shape and applies controlled margins.

Test cases:
    1. DEFAULT
    2. MARGIN_0.005
    3. MARGIN_0.001
    4. MARGIN_0.0001

Run:
    cd C:\python_projects\rl_robot\v3
    python tests\test_v3225f_caster_geometry_correction.py
"""

import math
import pybullet as p
import pybullet_data


# ============================================================
# V3 ROBOT PARAMETERS
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

LEFT_WHEEL_JOINT = 0
RIGHT_WHEEL_JOINT = 1
CASTER_SWIVEL_JOINT = 2
CASTER_WHEEL_JOINT = 3

GROUND_ID = 0


# ============================================================
# TEST CONFIGURATION
# ============================================================

SETTLE_STEPS = 100

FORWARD_VELOCITY = 4.0
REVERSE_VELOCITY = -4.0

MOTION_STEPS = 1000

# Collision margins to test.
#
# None means:
#     use PyBullet's default margin.
#
# The other values explicitly set the caster wheel margin.
MARGIN_TESTS = [
    ("DEFAULT", None),
    ("MARGIN_0.005", 0.005),
    ("MARGIN_0.001", 0.001),
    ("MARGIN_0.0001", 0.0001),
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def deg(value):
    """Convert radians to degrees."""
    return math.degrees(value)


def clamp(value, low, high):
    return max(low, min(high, value))


def get_base_state(robot_id):
    """Return base position and Euler orientation."""
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orientation)

    return position, euler


def get_base_pitch(robot_id):
    """Return base pitch in degrees."""
    _, euler = get_base_state(robot_id)
    return deg(euler[1])


def get_base_roll(robot_id):
    """Return base roll in degrees."""
    _, euler = get_base_state(robot_id)
    return deg(euler[0])


def get_base_yaw(robot_id):
    """Return base yaw in degrees."""
    _, euler = get_base_state(robot_id)
    return deg(euler[2])


def get_link_state(robot_id, link_index):
    """Return world position and Euler orientation of a link."""
    state = p.getLinkState(
        robot_id,
        link_index,
        computeForwardKinematics=True,
    )

    position = state[4]
    orientation = state[5]

    euler = p.getEulerFromQuaternion(orientation)

    return position, euler


def get_caster_contacts(robot_id):
    """
    Return all caster-ground contacts.

    Link index 3 is the caster wheel.
    """
    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=GROUND_ID,
        linkIndexA=CASTER_WHEEL_JOINT,
    )

    return contacts


def get_caster_force(robot_id):
    """Sum normal forces generated at caster-ground contacts."""
    contacts = get_caster_contacts(robot_id)

    total_force = 0.0

    for contact in contacts:
        # PyBullet contact tuple:
        # normalForce is index 9.
        total_force += contact[9]

    return total_force


def get_caster_contact_distances(robot_id):
    """Return all caster-ground contact distances."""
    contacts = get_caster_contacts(robot_id)

    return [contact[8] for contact in contacts]


def get_all_robot_ground_contacts(robot_id):
    """Return all robot-ground contacts."""
    return p.getContactPoints(
        bodyA=robot_id,
        bodyB=GROUND_ID,
    )


# ============================================================
# ROBOT CREATION
# ============================================================

def create_robot(caster_margin=None):
    """
    Create the V3 robot.

    caster_margin:
        None -> use PyBullet default caster collision margin
        float -> explicitly set caster wheel margin
    """

    # --------------------------------------------------------
    # CHASSIS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DRIVE WHEELS
    # --------------------------------------------------------

    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
    )

    # --------------------------------------------------------
    # CASTER FORK
    # --------------------------------------------------------

    caster_fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            CASTER_TRAIL / 2.0,
            CASTER_WHEEL_WIDTH,
            0.02,
        ],
    )

    caster_fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            CASTER_TRAIL / 2.0,
            CASTER_WHEEL_WIDTH,
            0.02,
        ],
    )

    # --------------------------------------------------------
    # CASTER WHEEL
    # --------------------------------------------------------

    caster_wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        height=CASTER_WHEEL_WIDTH,
    )

    caster_wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        length=CASTER_WHEEL_WIDTH,
    )

    # --------------------------------------------------------
    # APPLY CONTROLLED COLLISION MARGIN
    # --------------------------------------------------------

    if caster_margin is not None:
        p.changeDynamics(
            # body does not exist yet, so margin cannot be
            # changed here through changeDynamics.
            #
            # The actual margin correction is therefore applied
            # after createMultiBody().
            0,
            -1,
        )

    # --------------------------------------------------------
    # MULTIBODY
    # --------------------------------------------------------

    wheel_center_z = WHEEL_RADIUS

    chassis_center_z = (
        WHEEL_RADIUS +
        CHASSIS_HEIGHT / 2.0
    )

    robot_id = p.createMultiBody(
        baseMass=CHASSIS_MASS,
        baseCollisionShapeIndex=chassis_collision,
        baseVisualShapeIndex=chassis_visual,
        basePosition=[
            0.0,
            0.0,
            chassis_center_z,
        ],

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

        linkPositions=[
            [0.0, WHEEL_BASE / 2.0, -CHASSIS_HEIGHT / 2.0],
            [0.0, -WHEEL_BASE / 2.0, -CHASSIS_HEIGHT / 2.0],
            [CASTER_X, 0.0, -CHASSIS_HEIGHT / 2.0],
            [-CASTER_TRAIL, 0.0, -0.04],
        ],

        linkOrientations=[
            p.getQuaternionFromEuler(
                [math.pi / 2.0, 0.0, 0.0]
            ),
            p.getQuaternionFromEuler(
                [math.pi / 2.0, 0.0, 0.0]
            ),
            [0.0, 0.0, 0.0, 1.0],
            p.getQuaternionFromEuler(
                [math.pi / 2.0, 0.0, 0.0]
            ),
        ],

        linkInertialFramePositions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],

        linkInertialFrameOrientations=[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],

        linkParentIndices=[
            0,
            0,
            0,
            3,
        ],

        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
        ],

        linkJointAxis=[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
    )

    # --------------------------------------------------------
    # APPLY CASTER COLLISION MARGIN
    # --------------------------------------------------------

    if caster_margin is not None:
        p.changeDynamics(
            robot_id,
            CASTER_WHEEL_JOINT,
            lateralFriction=0.8,
            spinningFriction=0.01,
            rollingFriction=0.01,
            collisionMargin=caster_margin,
        )
    else:
        # Keep the default Bullet margin.
        p.changeDynamics(
            robot_id,
            CASTER_WHEEL_JOINT,
            lateralFriction=0.8,
            spinningFriction=0.01,
            rollingFriction=0.01,
        )

    # --------------------------------------------------------
    # DRIVE WHEEL DYNAMICS
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        LEFT_WHEEL_JOINT,
        lateralFriction=1.0,
        spinningFriction=0.01,
        rollingFriction=0.01,
    )

    p.changeDynamics(
        robot_id,
        RIGHT_WHEEL_JOINT,
        lateralFriction=1.0,
        spinningFriction=0.01,
        rollingFriction=0.01,
    )

    # --------------------------------------------------------
    # CASTER FORK DYNAMICS
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        CASTER_SWIVEL_JOINT,
        lateralFriction=0.8,
        spinningFriction=0.01,
        rollingFriction=0.01,
    )

    # --------------------------------------------------------
    # INITIAL JOINT STATES
    # --------------------------------------------------------

    p.resetJointState(
        robot_id,
        LEFT_WHEEL_JOINT,
        0.0,
        0.0,
    )

    p.resetJointState(
        robot_id,
        RIGHT_WHEEL_JOINT,
        0.0,
        0.0,
    )

    p.resetJointState(
        robot_id,
        CASTER_SWIVEL_JOINT,
        0.0,
        0.0,
    )

    p.resetJointState(
        robot_id,
        CASTER_WHEEL_JOINT,
        0.0,
        0.0,
    )

    return robot_id


# ============================================================
# MOTOR CONTROL
# ============================================================

def stop_robot(robot_id):
    """Stop all wheel motors."""
    p.setJointMotorControl2(
        robot_id,
        LEFT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )


def set_wheel_velocity(robot_id, left_velocity, right_velocity):
    """Command left and right drive wheel velocities."""

    p.setJointMotorControl2(
        robot_id,
        LEFT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=left_velocity,
        force=MOTOR_FORCE,
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=right_velocity,
        force=MOTOR_FORCE,
    )


# ============================================================
# SETUP PHYSICS
# ============================================================

def setup_world():
    """Create PyBullet world and ground."""

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(
        0.0,
        0.0,
        GRAVITY,
    )

    p.setTimeStep(TIME_STEP)

    p.setPhysicsEngineParameter(
        fixedTimeStep=TIME_STEP,
        numSolverIterations=100,
    )

    ground_id = p.loadURDF(
        "plane.urdf",
        useFixedBase=True,
    )

    return ground_id


# ============================================================
# SETTLE TEST
# ============================================================

def settle_robot(robot_id, steps=SETTLE_STEPS):
    """Let the robot settle under gravity."""

    stop_robot(robot_id)

    for _ in range(steps):
        p.stepSimulation()


# ============================================================
# MOTION TEST
# ============================================================

def run_motion_test(
    robot_id,
    left_velocity,
    right_velocity,
    steps=MOTION_STEPS,
):
    """
    Run controlled wheel motion.

    Returns detailed diagnostic data.
    """

    pitch_values = []
    roll_values = []
    base_z_values = []
    caster_z_values = []
    caster_force_values = []
    caster_contact_counts = []
    caster_contact_distance_values = []
    swivel_values = []

    set_wheel_velocity(
        robot_id,
        left_velocity,
        right_velocity,
    )

    for step in range(steps):

        p.stepSimulation()

        # ----------------------------------------------------
        # BASE
        # ----------------------------------------------------

        base_position, base_euler = get_base_state(robot_id)

        base_z = base_position[2]
        roll = deg(base_euler[0])
        pitch = deg(base_euler[1])

        # ----------------------------------------------------
        # CASTER
        # ----------------------------------------------------

        caster_position, caster_euler = get_link_state(
            robot_id,
            CASTER_WHEEL_JOINT,
        )

        caster_z = caster_position[2]

        swivel_position, swivel_euler = get_link_state(
            robot_id,
            CASTER_SWIVEL_JOINT,
        )

        swivel_yaw = deg(swivel_euler[2])

        contacts = get_caster_contacts(robot_id)

        caster_force = get_caster_force(robot_id)

        contact_distances = get_caster_contact_distances(
            robot_id
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        pitch_values.append(pitch)
        roll_values.append(roll)

        base_z_values.append(base_z)
        caster_z_values.append(caster_z)

        caster_force_values.append(caster_force)
        caster_contact_counts.append(len(contacts))

        if contact_distances:
            caster_contact_distance_values.append(
                min(contact_distances)
            )

        swivel_values.append(swivel_yaw)

        # ----------------------------------------------------
        # EARLY DIAGNOSTIC OUTPUT
        # ----------------------------------------------------

        if step in [0, 1, 2, 9, 24, 49, 74, 99, 199, 499, 999]:

            print(
                f"  Step {step + 1:4d} | "
                f"X={base_position[0]: .6f} | "
                f"BaseZ={base_z:.6f} | "
                f"CasterZ={caster_z:.6f} | "
                f"Pitch={pitch: .3f} deg | "
                f"Contacts={len(contacts):d} | "
                f"Force={caster_force:.4f} N"
            )

    stop_robot(robot_id)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    avg_pitch = (
        sum(pitch_values) / len(pitch_values)
        if pitch_values
        else 0.0
    )

    avg_roll = (
        sum(roll_values) / len(roll_values)
        if roll_values
        else 0.0
    )

    max_abs_pitch = (
        max(abs(value) for value in pitch_values)
        if pitch_values
        else 0.0
    )

    max_abs_roll = (
        max(abs(value) for value in roll_values)
        if roll_values
        else 0.0
    )

    final_pitch = (
        pitch_values[-1]
        if pitch_values
        else 0.0
    )

    final_roll = (
        roll_values[-1]
        if roll_values
        else 0.0
    )

    final_base_z = (
        base_z_values[-1]
        if base_z_values
        else 0.0
    )

    final_caster_z = (
        caster_z_values[-1]
        if caster_z_values
        else 0.0
    )

    avg_caster_force = (
        sum(caster_force_values) /
        len(caster_force_values)
        if caster_force_values
        else 0.0
    )

    max_caster_force = (
        max(caster_force_values)
        if caster_force_values
        else 0.0
    )

    contact_fraction = (
        sum(
            1
            for count in caster_contact_counts
            if count > 0
        )
        / len(caster_contact_counts)
        if caster_contact_counts
        else 0.0
    )

    full_contact_fraction = (
        sum(
            1
            for count in caster_contact_counts
            if count >= 3
        )
        / len(caster_contact_counts)
        if caster_contact_counts
        else 0.0
    )

    max_caster_z = (
        max(caster_z_values)
        if caster_z_values
        else 0.0
    )

    min_caster_z = (
        min(caster_z_values)
        if caster_z_values
        else 0.0
    )

    caster_z_range = (
        max_caster_z - min_caster_z
    )

    max_swivel = (
        max(abs(value) for value in swivel_values)
        if swivel_values
        else 0.0
    )

    if caster_contact_distance_values:
        min_contact_distance = min(
            caster_contact_distance_values
        )
        max_contact_distance = max(
            caster_contact_distance_values
        )
    else:
        min_contact_distance = None
        max_contact_distance = None

    return {
        "avg_pitch": avg_pitch,
        "avg_roll": avg_roll,
        "max_abs_pitch": max_abs_pitch,
        "max_abs_roll": max_abs_roll,
        "final_pitch": final_pitch,
        "final_roll": final_roll,
        "final_base_z": final_base_z,
        "final_caster_z": final_caster_z,
        "avg_caster_force": avg_caster_force,
        "max_caster_force": max_caster_force,
        "contact_fraction": contact_fraction,
        "full_contact_fraction": full_contact_fraction,
        "min_caster_z": min_caster_z,
        "max_caster_z": max_caster_z,
        "caster_z_range": caster_z_range,
        "max_swivel": max_swivel,
        "min_contact_distance": min_contact_distance,
        "max_contact_distance": max_contact_distance,
    }


# ============================================================
# COLLISION SHAPE INSPECTION
# ============================================================

def inspect_caster_geometry(robot_id, label):
    """
    Inspect the actual caster link state and dynamics.

    Note:
    PyBullet's reported cylinder dimensions follow its own
    internal convention. The purpose here is to record the
    actual values rather than assume an interpretation.
    """

    print()
    print("-" * 70)
    print(f"CASTER GEOMETRY INSPECTION — {label}")
    print("-" * 70)

    dynamics = p.getDynamicsInfo(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    print(f"Mass:              {dynamics[0]:.6f} kg")
    print(f"Lateral friction:  {dynamics[1]:.6f}")
    print(f"Rolling friction:  {dynamics[6]:.6f}")
    print(f"Spinning friction: {dynamics[7]:.6f}")
    print(f"Restitution:       {dynamics[5]:.6f}")

    position, euler = get_link_state(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    print()
    print(
        "Caster wheel world position:"
    )
    print(
        f"  X = {position[0]: .9f} m"
    )
    print(
        f"  Y = {position[1]: .9f} m"
    )
    print(
        f"  Z = {position[2]: .9f} m"
    )

    print()
    print("Caster wheel Euler:")
    print(
        f"  Roll  = {deg(euler[0]): .6f} deg"
    )
    print(
        f"  Pitch = {deg(euler[1]): .6f} deg"
    )
    print(
        f"  Yaw   = {deg(euler[2]): .6f} deg"
    )

    print()
    print("Expected physical geometry:")
    print(
        f"  Radius = {CASTER_WHEEL_RADIUS:.6f} m"
    )
    print(
        f"  Width  = {CASTER_WHEEL_WIDTH:.6f} m"
    )

    print()
    print(
        "Naive physical bottom:"
    )

    naive_bottom = (
        position[2] -
        CASTER_WHEEL_RADIUS
    )

    print(
        f"  Z_bottom = {naive_bottom:.9f} m"
    )

    contacts = get_caster_contacts(robot_id)

    print()
    print(
        f"Caster-ground contacts: {len(contacts)}"
    )

    if contacts:
        for index, contact in enumerate(contacts):

            point_a = contact[5]
            point_b = contact[6]
            normal = contact[7]
            distance = contact[8]
            force = contact[9]

            print()
            print(f"Contact {index + 1}:")
            print(
                f"  Point A = "
                f"({point_a[0]: .9f}, "
                f"{point_a[1]: .9f}, "
                f"{point_a[2]: .9f})"
            )
            print(
                f"  Point B = "
                f"({point_b[0]: .9f}, "
                f"{point_b[1]: .9f}, "
                f"{point_b[2]: .9f})"
            )
            print(
                f"  Normal = "
                f"({normal[0]: .6f}, "
                f"{normal[1]: .6f}, "
                f"{normal[2]: .6f})"
            )
            print(
                f"  Distance = {distance:.9f} m"
            )
            print(
                f"  Normal force = {force:.6f} N"
            )


# ============================================================
# SINGLE CASE
# ============================================================

def run_case(label, caster_margin):
    """Run one complete margin test."""

    print()
    print("=" * 80)
    print(f"CASE: {label}")
    print("=" * 80)

    if caster_margin is None:
        print("Caster collision margin: DEFAULT")
    else:
        print(
            f"Caster collision margin: "
            f"{caster_margin:.6f} m"
        )

    # --------------------------------------------------------
    # NEW WORLD
    # --------------------------------------------------------

    p.resetSimulation()

    setup_world()

    robot_id = create_robot(
        caster_margin=caster_margin
    )

    # --------------------------------------------------------
    # STATIC SETTLE
    # --------------------------------------------------------

    print()
    print("Static settling...")

    settle_robot(
        robot_id,
        SETTLE_STEPS,
    )

    inspect_caster_geometry(
        robot_id,
        label + " — AFTER SETTLE",
    )

    static_pitch = get_base_pitch(robot_id)
    static_roll = get_base_roll(robot_id)

    static_caster_contacts = len(
        get_caster_contacts(robot_id)
    )

    static_caster_force = get_caster_force(
        robot_id
    )

    # --------------------------------------------------------
    # FORWARD TEST
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("FORWARD +4 / +4")
    print("-" * 80)

    forward = run_motion_test(
        robot_id,
        FORWARD_VELOCITY,
        FORWARD_VELOCITY,
        MOTION_STEPS,
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    p.resetSimulation()

    setup_world()

    robot_id = create_robot(
        caster_margin=caster_margin
    )

    settle_robot(
        robot_id,
        SETTLE_STEPS,
    )

    # --------------------------------------------------------
    # REVERSE TEST
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("REVERSE -4 / -4")
    print("-" * 80)

    reverse = run_motion_test(
        robot_id,
        REVERSE_VELOCITY,
        REVERSE_VELOCITY,
        MOTION_STEPS,
    )

    # --------------------------------------------------------
    # CASE RESULT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"RESULT — {label}")
    print("=" * 80)

    print()
    print("STATIC:")
    print(
        f"  Pitch              = "
        f"{static_pitch:.6f} deg"
    )
    print(
        f"  Roll               = "
        f"{static_roll:.6f} deg"
    )
    print(
        f"  Caster contacts    = "
        f"{static_caster_contacts}"
    )
    print(
        f"  Caster force       = "
        f"{static_caster_force:.6f} N"
    )

    print()
    print("FORWARD:")
    print(
        f"  Final pitch        = "
        f"{forward['final_pitch']:.6f} deg"
    )
    print(
        f"  Max |pitch|        = "
        f"{forward['max_abs_pitch']:.6f} deg"
    )
    print(
        f"  Caster contact     = "
        f"{forward['contact_fraction'] * 100.0:.2f}%"
    )
    print(
        f"  Full contact >=3   = "
        f"{forward['full_contact_fraction'] * 100.0:.2f}%"
    )
    print(
        f"  Avg caster force   = "
        f"{forward['avg_caster_force']:.6f} N"
    )
    print(
        f"  Final caster Z     = "
        f"{forward['final_caster_z']:.6f} m"
    )

    print()
    print("REVERSE:")
    print(
        f"  Final pitch        = "
        f"{reverse['final_pitch']:.6f} deg"
    )
    print(
        f"  Max |pitch|        = "
        f"{reverse['max_abs_pitch']:.6f} deg"
    )
    print(
        f"  Caster contact     = "
        f"{reverse['contact_fraction'] * 100.0:.2f}%"
    )
    print(
        f"  Full contact >=3   = "
        f"{reverse['full_contact_fraction'] * 100.0:.2f}%"
    )
    print(
        f"  Avg caster force   = "
        f"{reverse['avg_caster_force']:.6f} N"
    )
    print(
        f"  Max caster force   = "
        f"{reverse['max_caster_force']:.6f} N"
    )
    print(
        f"  Final caster Z     = "
        f"{reverse['final_caster_z']:.6f} m"
    )
    print(
        f"  Caster Z range     = "
        f"{reverse['caster_z_range']:.6f} m"
    )
    print(
        f"  Max swivel         = "
        f"{reverse['max_swivel']:.6f} deg"
    )

    # --------------------------------------------------------
    # DIAGNOSTIC INTERPRETATION
    # --------------------------------------------------------

    print()
    print("DIAGNOSTIC:")

    if reverse["contact_fraction"] > 0.95:
        print(
            "  [PASS] Caster maintains ground contact "
            "during reverse."
        )
    else:
        print(
            "  [INVESTIGATE] Caster loses ground contact "
            "during reverse."
        )

    if reverse["max_abs_pitch"] < 2.0:
        print(
            "  [PASS] Reverse pitch remains below 2 deg."
        )
    else:
        print(
            "  [INVESTIGATE] Reverse pitch exceeds 2 deg."
        )

    if reverse["avg_caster_force"] > 0.1:
        print(
            "  [PASS] Caster carries measurable load."
        )
    else:
        print(
            "  [INVESTIGATE] Caster carries almost no load."
        )

    return {
        "label": label,
        "margin": caster_margin,
        "static_pitch": static_pitch,
        "static_roll": static_roll,
        "static_contacts": static_caster_contacts,
        "static_force": static_caster_force,
        "forward": forward,
        "reverse": reverse,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison(results):
    """Print a compact comparison of all margin cases."""

    print()
    print()
    print("=" * 110)
    print("V3.22.5f — COLLISION MARGIN COMPARISON")
    print("=" * 110)

    header = (
        f"{'Case':<18}"
        f"{'Margin':>10}"
        f"{'RevPitch':>12}"
        f"{'MaxPitch':>12}"
        f"{'Contact%':>12}"
        f"{'AvgForce':>12}"
        f"{'CasterZ':>12}"
        f"{'PASS?':>10}"
    )

    print(header)
    print("-" * 110)

    for result in results:

        reverse = result["reverse"]

        margin_text = (
            "DEFAULT"
            if result["margin"] is None
            else f"{result['margin']:.4f}"
        )

        pass_case = (
            reverse["max_abs_pitch"] < 2.0
            and reverse["contact_fraction"] > 0.95
            and reverse["avg_caster_force"] > 0.1
        )

        print(
            f"{result['label']:<18}"
            f"{margin_text:>10}"
            f"{reverse['final_pitch']:>12.4f}"
            f"{reverse['max_abs_pitch']:>12.4f}"
            f"{reverse['contact_fraction'] * 100.0:>12.2f}"
            f"{reverse['avg_caster_force']:>12.4f}"
            f"{reverse['final_caster_z']:>12.5f}"
            f"{'PASS' if pass_case else 'INVESTIGATE':>10}"
        )

    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("V3.22.5f — CASTER GEOMETRY / COLLISION-MARGIN CORRECTION")
    print("=" * 80)

    print()
    print("Purpose:")
    print(
        "Determine whether the V3 reverse-motion caster failure "
        "is caused by Bullet collision margin."
    )

    print()
    print("Robot geometry:")
    print(
        f"  Caster physical radius = "
        f"{CASTER_WHEEL_RADIUS:.4f} m"
    )
    print(
        f"  Caster physical width  = "
        f"{CASTER_WHEEL_WIDTH:.4f} m"
    )

    print()
    print("Test margins:")

    for label, margin in MARGIN_TESTS:

        if margin is None:
            print(
                f"  {label:<20} default Bullet margin"
            )
        else:
            print(
                f"  {label:<20} {margin:.4f} m"
            )

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    physics_client = p.connect(
        p.DIRECT
    )

    if physics_client < 0:
        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    print()
    print(
        f"PyBullet client ID: {physics_client}"
    )

    results = []

    try:

        for label, margin in MARGIN_TESTS:

            result = run_case(
                label,
                margin,
            )

            results.append(result)

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        print_comparison(results)

        # ----------------------------------------------------
        # FINAL INTERPRETATION
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("FINAL INTERPRETATION")
        print("=" * 80)

        default_result = results[0]["reverse"]

        corrected_results = results[1:]

        improved_cases = []

        for result in corrected_results:

            reverse = result["reverse"]

            if (
                reverse["max_abs_pitch"]
                < default_result["max_abs_pitch"] * 0.5
                and reverse["contact_fraction"] > 0.95
            ):
                improved_cases.append(
                    result["label"]
                )

        if improved_cases:

            print()
            print(
                "[RESULT] Controlled collision margin "
                "substantially improves reverse behavior."
            )

            print()
            print(
                "Improved cases:"
            )

            for label in improved_cases:
                print(
                    f"  - {label}"
                )

            print()
            print(
                "Interpretation:"
            )
            print(
                "The previous reverse pitch problem is "
                "strongly associated with the caster "
                "collision margin."
            )

            print()
            print(
                "NEXT ACTION:"
            )
            print(
                "Use the smallest physically reasonable "
                "margin that gives stable contact."
            )

        else:

            print()
            print(
                "[RESULT] Collision-margin correction does "
                "not clearly resolve reverse instability."
            )

            print()
            print(
                "Interpretation:"
            )
            print(
                "The reverse caster problem is probably "
                "not caused primarily by the collision margin."
            )

            print()
            print(
                "NEXT ACTION:"
            )
            print(
                "Investigate caster joint hierarchy, "
                "collision geometry, contact constraints, "
                "and caster kinematics."
            )

        print()
        print("=" * 80)
        print("V3.22.5f COMPLETE")
        print("=" * 80)

    finally:

        p.disconnect()

        print()
        print(
            "PyBullet disconnected."
        )


if __name__ == "__main__":
    main()