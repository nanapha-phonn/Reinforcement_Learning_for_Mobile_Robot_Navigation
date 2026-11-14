import time
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


# ============================================================
# V3.15.3.3
# WHEEL DIRECTION DIAGNOSTIC
# ============================================================

TEST_TIME = 1.0
VELOCITY = 4.0


# ============================================================
# CONNECT
# ============================================================

p.connect(p.GUI)

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


# ============================================================
# GROUND
# ============================================================

ground_id = p.loadURDF(
    "plane.urdf"
)


# ============================================================
# ROBOT
# ============================================================

robot_id = create_robot()

# ============================================================
# TEMPORARY CASTER COLLISION ISOLATION
# ============================================================

# Disable ground collision for caster wheel only.
# Link 3 = caster wheel in V3.15.2.4
p.setCollisionFilterGroupMask(
    robot_id,
    3,
    collisionFilterGroup=0,
    collisionFilterMask=0,
)

print()
print("WARNING: CASTER WHEEL GROUND COLLISION DISABLED")
print("This is a diagnostic test only.")

# ============================================================
# SETTLE
# ============================================================

for _ in range(
    int(2.0 / TIME_STEP)
):
    p.stepSimulation()


# ============================================================
# TEST FUNCTION
# ============================================================

def run_test(
    name,
    left_velocity,
    right_velocity,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Command: "
        f"L={left_velocity:.1f}, "
        f"R={right_velocity:.1f} rad/s"
    )

    # --------------------------------------------------------
    # Start position
    # --------------------------------------------------------

    start_position, start_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    start_yaw = p.getEulerFromQuaternion(
        start_orientation
    )[2]

    # --------------------------------------------------------
    # Drive
    # --------------------------------------------------------

    set_wheel_velocity(
        robot_id,
        4.0,
        4.0,
    )

    left_fx = []
    right_fx = []

    for step in range(
        int(TEST_TIME / TIME_STEP)
    ):

        p.stepSimulation()

        # --------------------------------------------
        # Contacts
        # --------------------------------------------

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

        fx_left = 0.0
        fx_right = 0.0

        for contact in left_contacts:

            fx_left += (
                contact[12]
                * contact[13][0]
            )

        for contact in right_contacts:

            fx_right += (
                contact[12]
                * contact[13][0]
            )

        left_fx.append(
            fx_left
        )

        right_fx.append(
            fx_right
        )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    stop_robot(
        robot_id
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_position, final_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    final_yaw = p.getEulerFromQuaternion(
        final_orientation
    )[2]

    dx = (
        final_position[0]
        - start_position[0]
    )

    dy = (
        final_position[1]
        - start_position[1]
    )

    dyaw = (
        final_yaw
        - start_yaw
    )

    # --------------------------------------------------------
    # Average friction
    # --------------------------------------------------------

    avg_left_fx = (
        sum(left_fx)
        / len(left_fx)
    )

    avg_right_fx = (
        sum(right_fx)
        / len(right_fx)
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print(
        f"X displacement : {dx:.6f} m"
    )

    print(
        f"Y displacement : {dy:.6f} m"
    )

    print(
        f"Yaw change     : "
        f"{math.degrees(dyaw):.6f} deg"
    )

    print(
        f"Average left Fx : "
        f"{avg_left_fx:.6f} N"
    )

    print(
        f"Average right Fx: "
        f"{avg_right_fx:.6f} N"
    )

    # --------------------------------------------------------
    # Small reset
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        robot_id,
        [0, 0, 0.13],
        p.getQuaternionFromEuler(
            [0, 0, 0]
        )
    )

    for joint in [0, 1, 2, 3]:

        p.resetJointState(
            robot_id,
            joint,
            0.0,
            0.0,
        )

    for _ in range(
        int(1.0 / TIME_STEP)
    ):

        p.stepSimulation()


# ============================================================
# TEST 1
# ============================================================

run_test(
    "TEST 1 — BOTH POSITIVE",
    +VELOCITY,
    +VELOCITY,
)


# ============================================================
# TEST 2
# ============================================================

run_test(
    "TEST 2 — LEFT POSITIVE / RIGHT NEGATIVE",
    +VELOCITY,
    -VELOCITY,
)


# ============================================================
# TEST 3
# ============================================================

run_test(
    "TEST 3 — BOTH NEGATIVE",
    -VELOCITY,
    -VELOCITY,
)


# ============================================================
# TEST 4
# ============================================================

run_test(
    "TEST 4 — LEFT NEGATIVE / RIGHT POSITIVE",
    -VELOCITY,
    +VELOCITY,
)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("V3.15.3.3 COMPLETE")
print("=" * 70)

print(
    "Do not modify robot.py yet."
)

print(
    "The four tests determine the correct"
)

print(
    "wheel sign convention."
)

time.sleep(5)

p.disconnect()