import math

import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    set_wheel_velocity,
    stop_robot,
    TIME_STEP,
    GRAVITY,
    MOTOR_FORCE,
)


TEST_TIME = 0.5
TARGET = 4.0


def run_test(name, left_command, right_command):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # Fresh simulation
    # --------------------------------------------------------

    p.connect(p.DIRECT)

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

    p.setRealTimeSimulation(0)

    ground_id = p.loadURDF(
        "plane.urdf"
    )

    robot_id = create_robot()

    # --------------------------------------------------------
    # Disable caster wheel-ground collision
    # --------------------------------------------------------

    p.setCollisionFilterGroupMask(
        robot_id,
        3,
        collisionFilterGroup=0,
        collisionFilterMask=0,
    )

    # --------------------------------------------------------
    # Settle
    # --------------------------------------------------------

    for _ in range(
        int(2.0 / TIME_STEP)
    ):
        p.stepSimulation()

    start_pos, start_quat = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    start_euler = p.getEulerFromQuaternion(
        start_quat
    )

    # --------------------------------------------------------
    # Drive
    # --------------------------------------------------------

    set_wheel_velocity(
        robot_id,
        left_command,
        right_command,
    )

    left_vel = []
    right_vel = []

    for _ in range(
        int(TEST_TIME / TIME_STEP)
    ):

        p.stepSimulation()

        left_state = p.getJointState(
            robot_id,
            0
        )

        right_state = p.getJointState(
            robot_id,
            1
        )

        left_vel.append(
            left_state[1]
        )

        right_vel.append(
            right_state[1]
        )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    end_pos, end_quat = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    end_euler = p.getEulerFromQuaternion(
        end_quat
    )

    # --------------------------------------------------------
    # Contacts
    # --------------------------------------------------------

    left_contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=0,
    )

    right_contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=1,
    )

    caster_contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=3,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    dz = end_pos[2] - start_pos[2]

    roll = math.degrees(
        end_euler[0]
    )

    pitch = math.degrees(
        end_euler[1]
    )

    yaw = math.degrees(
        end_euler[2]
    )

    avg_left = (
        sum(left_vel) /
        len(left_vel)
    )

    avg_right = (
        sum(right_vel) /
        len(right_vel)
    )

    print()
    print("COMMAND")
    print(
        f"Left  = {left_command:.3f} rad/s"
    )
    print(
        f"Right = {right_command:.3f} rad/s"
    )

    print()
    print("WHEEL VELOCITY")
    print(
        f"Average left  = {avg_left:.6f}"
    )
    print(
        f"Average right = {avg_right:.6f}"
    )

    print()
    print("MOTION")
    print(
        f"DX = {dx:.6f} m"
    )
    print(
        f"DY = {dy:.6f} m"
    )
    print(
        f"DZ = {dz:.6f} m"
    )

    print()
    print("ORIENTATION")
    print(
        f"Roll  = {roll:.6f} deg"
    )
    print(
        f"Pitch = {pitch:.6f} deg"
    )
    print(
        f"Yaw   = {yaw:.6f} deg"
    )

    print()
    print("CONTACTS")
    print(
        f"Left wheel  = {len(left_contacts)}"
    )
    print(
        f"Right wheel = {len(right_contacts)}"
    )
    print(
        f"Caster      = {len(caster_contacts)}"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    stop_robot(
        robot_id
    )

    p.disconnect()


# ============================================================
# TEST 1
# ============================================================

run_test(
    "TEST 1 — BOTH POSITIVE",
    +TARGET,
    +TARGET,
)


# ============================================================
# TEST 2
# ============================================================

run_test(
    "TEST 2 — LEFT POSITIVE / RIGHT NEGATIVE",
    +TARGET,
    -TARGET,
)


# ============================================================
# TEST 3
# ============================================================

run_test(
    "TEST 3 — BOTH NEGATIVE",
    -TARGET,
    -TARGET,
)


# ============================================================
# TEST 4
# ============================================================

run_test(
    "TEST 4 — LEFT NEGATIVE / RIGHT POSITIVE",
    -TARGET,
    +TARGET,
)


print()
print("=" * 70)
print("V3.15.3.4 COMPLETE")
print("=" * 70)