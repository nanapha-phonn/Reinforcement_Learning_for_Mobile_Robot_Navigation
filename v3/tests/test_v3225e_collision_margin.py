"""
V3.22.5e - Caster Collision-Margin / Geometry Sensitivity Test

Purpose
-------
Investigate whether the abnormal reverse behavior is caused by the
small caster geometry interacting with Bullet's default convex
collision margin.

IMPORTANT
---------
This is a diagnostic experiment.

Do NOT modify the production robot.py yet.
Do NOT train PPO.

Test cases:
    A. Baseline geometry
    B. Reduced effective collision margin / clearance geometry: 0.005 m
    C. Reduced effective collision margin / clearance geometry: 0.001 m

The experiment compares:
    - caster contact fraction
    - average caster normal force
    - maximum pitch
    - final pitch
    - base Z
    - caster Z
    - forward behavior
    - reverse behavior

The robot hierarchy remains:
    link 0 -> left wheel
    link 1 -> right wheel
    link 2 -> caster fork
    link 3 -> caster wheel
"""

import math
import statistics
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


# ============================================================
# TEST SETTINGS
# ============================================================

SETTLE_STEPS = 240
MOTION_STEPS = 1000

GROUND_ID = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def deg(rad):
    return math.degrees(rad)


def get_base_state(robot_id):
    position, orientation = p.getBasePositionAndOrientation(robot_id)

    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

    return {
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def get_caster_state(robot_id):
    states = p.getLinkStates(
        robot_id,
        [2, 3],
        computeForwardKinematics=True,
    )

    fork_state = states[0]
    wheel_state = states[1]

    fork_pos = fork_state[4]
    fork_orn = fork_state[5]

    wheel_pos = wheel_state[4]
    wheel_orn = wheel_state[5]

    fork_rpy = p.getEulerFromQuaternion(fork_orn)
    wheel_rpy = p.getEulerFromQuaternion(wheel_orn)

    return {
        "fork_x": fork_pos[0],
        "fork_y": fork_pos[1],
        "fork_z": fork_pos[2],
        "fork_roll": fork_rpy[0],
        "fork_pitch": fork_rpy[1],
        "fork_yaw": fork_rpy[2],

        "wheel_x": wheel_pos[0],
        "wheel_y": wheel_pos[1],
        "wheel_z": wheel_pos[2],
        "wheel_roll": wheel_rpy[0],
        "wheel_pitch": wheel_rpy[1],
        "wheel_yaw": wheel_rpy[2],
    }


def get_caster_contacts(robot_id):
    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=GROUND_ID,
        linkIndexA=3,
    )

    total_force = 0.0

    for contact in contacts:
        total_force += contact[9]

    return len(contacts), total_force


def stop_robot(robot_id):
    p.setJointMotorControl2(
        robot_id,
        LEFT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=MOTOR_FORCE,
    )

    p.setJointMotorControl2(
        robot_id,
        RIGHT_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=MOTOR_FORCE,
    )


def set_wheel_velocity(robot_id, left_velocity, right_velocity):
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
# COLLISION GEOMETRY
# ============================================================

def create_caster_wheel_shape(test_margin):
    """
    Create the caster collision shape.

    NOTE:
    PyBullet's normal Python createCollisionShape interface does not
    provide a direct, portable setMargin() argument.

    Therefore this test uses a controlled geometry sensitivity
    experiment rather than claiming to directly alter Bullet's
    internal btConvexShape margin.

    test_margin represents the experimental geometry offset.
    """

    # Baseline physical radius.
    radius = CASTER_WHEEL_RADIUS

    # Controlled diagnostic reduction.
    #
    # This does NOT change the production model.
    #
    # The purpose is to determine how sensitive the contact behavior
    # is to the small caster's effective collision envelope.
    effective_radius = radius - test_margin

    # Never allow an invalid radius.
    effective_radius = max(
        0.005,
        effective_radius,
    )

    return p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=effective_radius,
        height=CASTER_WHEEL_WIDTH,
        collisionFramePosition=[0.0, 0.0, 0.0],
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2.0, 0.0, 0.0]
        ),
    )


# ============================================================
# ROBOT CREATION
# ============================================================

def create_robot(test_margin):
    """
    Create the V3 diagnostic robot.

    The hierarchy is intentionally kept identical to V3.22.3:
        left wheel  -> base
        right wheel -> base
        caster fork -> base
        caster wheel -> caster fork
    """

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

    drive_wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        height=WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2.0, 0.0, 0.0]
        ),
    )

    drive_wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=WHEEL_RADIUS,
        length=WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2.0, 0.0, 0.0]
        ),
    )

    caster_fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            CASTER_TRAIL / 2.0,
            0.025,
            0.02,
        ],
    )

    caster_fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            CASTER_TRAIL / 2.0,
            0.025,
            0.02,
        ],
    )

    caster_wheel_collision = create_caster_wheel_shape(
        test_margin
    )

    caster_wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=CASTER_WHEEL_RADIUS,
        length=CASTER_WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2.0, 0.0, 0.0]
        ),
    )

    chassis_center_z = (
        WHEEL_RADIUS +
        CHASSIS_HEIGHT / 2.0
    )

    link_masses = [
        WHEEL_MASS,
        WHEEL_MASS,
        CASTER_FORK_MASS,
        CASTER_WHEEL_MASS,
    ]

    link_collision_shapes = [
        drive_wheel_collision,
        drive_wheel_collision,
        caster_fork_collision,
        caster_wheel_collision,
    ]

    link_visual_shapes = [
        drive_wheel_visual,
        drive_wheel_visual,
        caster_fork_visual,
        caster_wheel_visual,
    ]

    link_positions = [
        [0.0, +WHEEL_BASE / 2.0, -CHASSIS_HEIGHT / 2.0],
        [0.0, -WHEEL_BASE / 2.0, -CHASSIS_HEIGHT / 2.0],
        [CASTER_X, 0.0, -CHASSIS_HEIGHT / 2.0],
        [-CASTER_TRAIL, 0.0, -0.04],
    ]

    link_orientations = [
        p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
    ]

    # CORRECT V3.22.3 hierarchy.
    link_parent_indices = [
        0,
        0,
        0,
        3,
    ]

    link_joint_types = [
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
        p.JOINT_REVOLUTE,
    ]

    link_joint_axes = [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]

    robot_id = p.createMultiBody(
        baseMass=CHASSIS_MASS,
        baseCollisionShapeIndex=chassis_collision,
        baseVisualShapeIndex=chassis_visual,
        basePosition=[
            0.0,
            0.0,
            chassis_center_z,
        ],
        baseOrientation=[
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shapes,
        linkVisualShapeIndices=link_visual_shapes,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=[
            [0.0, 0.0, 0.0],
        ] * 4,
        linkInertialFrameOrientations=[
            [0.0, 0.0, 0.0, 1.0],
        ] * 4,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
    )

    return robot_id


# ============================================================
# SINGLE TEST
# ============================================================

def run_motion_test(robot_id, left_velocity, right_velocity):
    """
    Run one controlled motion experiment.
    """

    # Allow robot to settle first.
    stop_robot(robot_id)

    for _ in range(SETTLE_STEPS):
        p.stepSimulation()

    initial = get_base_state(robot_id)

    pitch_history = []
    roll_history = []
    base_z_history = []
    caster_z_history = []
    caster_force_history = []
    caster_contact_history = []

    for step in range(MOTION_STEPS):

        set_wheel_velocity(
            robot_id,
            left_velocity,
            right_velocity,
        )

        p.stepSimulation()

        base = get_base_state(robot_id)
        caster = get_caster_state(robot_id)

        contact_count, caster_force = get_caster_contacts(
            robot_id
        )

        pitch_history.append(
            deg(base["pitch"])
        )

        roll_history.append(
            deg(base["roll"])
        )

        base_z_history.append(
            base["z"]
        )

        caster_z_history.append(
            caster["wheel_z"]
        )

        caster_force_history.append(
            caster_force
        )

        caster_contact_history.append(
            contact_count
        )

    stop_robot(robot_id)

    # Final stabilization.
    for _ in range(120):
        p.stepSimulation()

    final = get_base_state(robot_id)
    final_caster = get_caster_state(robot_id)

    contact_fraction = (
        sum(
            1
            for value in caster_contact_history
            if value > 0
        )
        / len(caster_contact_history)
    )

    average_force = (
        statistics.mean(caster_force_history)
    )

    return {
        "initial": initial,
        "final": final,
        "final_caster": final_caster,

        "max_pitch": max(
            abs(value)
            for value in pitch_history
        ),

        "average_pitch": statistics.mean(
            abs(value)
            for value in pitch_history
        ),

        "max_roll": max(
            abs(value)
            for value in roll_history
        ),

        "base_z_min": min(base_z_history),
        "base_z_max": max(base_z_history),

        "caster_z_min": min(caster_z_history),
        "caster_z_max": max(caster_z_history),

        "caster_contact_fraction": contact_fraction,

        "average_caster_force": average_force,

        "final_contact_count": (
            caster_contact_history[-1]
        ),
    }


# ============================================================
# TEST CASE
# ============================================================

def run_case(label, test_margin):
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    print(
        f"Experimental geometry offset: "
        f"{test_margin:.6f} m"
    )

    robot_id = create_robot(test_margin)

    # --------------------------------------------------------
    # STATIC
    # --------------------------------------------------------

    static_result = run_motion_test(
        robot_id,
        0.0,
        0.0,
    )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    forward_result = run_motion_test(
        robot_id,
        +4.0,
        +4.0,
    )

    # --------------------------------------------------------
    # REVERSE
    # --------------------------------------------------------

    reverse_result = run_motion_test(
        robot_id,
        -4.0,
        -4.0,
    )

    print()
    print("STATIC")
    print(
        f"  final pitch: "
        f"{deg(static_result['final']['pitch']):.6f} deg"
    )
    print(
        f"  final roll:  "
        f"{deg(static_result['final']['roll']):.6f} deg"
    )

    print()
    print("FORWARD +4/+4")
    print(
        f"  final pitch: "
        f"{deg(forward_result['final']['pitch']):.6f} deg"
    )
    print(
        f"  max pitch:   "
        f"{forward_result['max_pitch']:.6f} deg"
    )
    print(
        f"  caster contact: "
        f"{forward_result['caster_contact_fraction'] * 100:.2f}%"
    )
    print(
        f"  avg caster force: "
        f"{forward_result['average_caster_force']:.6f} N"
    )

    print()
    print("REVERSE -4/-4")
    print(
        f"  final pitch: "
        f"{deg(reverse_result['final']['pitch']):.6f} deg"
    )
    print(
        f"  max pitch:   "
        f"{reverse_result['max_pitch']:.6f} deg"
    )
    print(
        f"  avg pitch:   "
        f"{reverse_result['average_pitch']:.6f} deg"
    )
    print(
        f"  caster contact: "
        f"{reverse_result['caster_contact_fraction'] * 100:.2f}%"
    )
    print(
        f"  avg caster force: "
        f"{reverse_result['average_caster_force']:.6f} N"
    )
    print(
        f"  final caster contacts: "
        f"{reverse_result['final_contact_count']}"
    )
    print(
        f"  final base Z: "
        f"{reverse_result['final']['z']:.6f} m"
    )
    print(
        f"  final caster Z: "
        f"{reverse_result['final_caster']['wheel_z']:.6f} m"
    )

    p.removeBody(robot_id)

    return {
        "static": static_result,
        "forward": forward_result,
        "reverse": reverse_result,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("V3.22.5e - CASTER COLLISION-MARGIN SENSITIVITY TEST")
    print("=" * 72)

    print()
    print("IMPORTANT:")
    print("This test does NOT modify robot.py.")
    print("PPO is disabled.")
    print("Sensor noise is disabled.")
    print("RL environment is disabled.")
    print()

    physics_client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        GRAVITY,
    )

    p.setTimeStep(TIME_STEP)

    p.loadURDF(
        "plane.urdf",
        [0.0, 0.0, 0.0],
    )

    # --------------------------------------------------------
    # Three diagnostic cases
    # --------------------------------------------------------

    results = {}

    results["DEFAULT"] = run_case(
        "CASE A - DEFAULT / BASELINE",
        0.0,
    )

    results["MARGIN_005"] = run_case(
        "CASE B - 0.005 m EXPERIMENT",
        0.005,
    )

    results["MARGIN_001"] = run_case(
        "CASE C - 0.001 m EXPERIMENT",
        0.001,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("V3.22.5e SUMMARY")
    print("=" * 72)

    print()
    print(
        f"{'CASE':<18}"
        f"{'REV PITCH':>14}"
        f"{'MAX PITCH':>14}"
        f"{'CASTER CONTACT':>18}"
        f"{'AVG FORCE':>14}"
    )

    print("-" * 78)

    for name, data in results.items():

        reverse = data["reverse"]

        print(
            f"{name:<18}"
            f"{deg(reverse['final']['pitch']):>14.3f}"
            f"{reverse['max_pitch']:>14.3f}"
            f"{reverse['caster_contact_fraction'] * 100:>17.2f}%"
            f"{reverse['average_caster_force']:>14.3f}"
        )

    print()
    print("=" * 72)

    baseline_pitch = abs(
        deg(
            results["DEFAULT"]["reverse"]["final"]["pitch"]
        )
    )

    baseline_contact = (
        results["DEFAULT"]["reverse"]
        ["caster_contact_fraction"]
    )

    print("INTERPRETATION")
    print("-" * 72)

    print(
        "This test is diagnostic only."
    )

    print(
        "A strong change in reverse pitch/contact across "
        "the three cases indicates that the caster collision "
        "geometry/envelope is sensitive to the small-scale "
        "collision model."
    )

    print(
        f"Baseline reverse pitch: "
        f"{baseline_pitch:.3f} deg"
    )

    print(
        f"Baseline caster contact: "
        f"{baseline_contact * 100:.2f}%"
    )

    print()
    print(
        "Do NOT modify robot.py or retrain PPO based only "
        "on this test."
    )

    print(
        "If the result changes significantly, the next step "
        "is to inspect the exact production caster collision "
        "shape and choose a physically justified Bullet "
        "collision configuration."
    )

    print()
    print(
        "V3.22.5e COMPLETE"
    )

    p.disconnect()


if __name__ == "__main__":
    main()