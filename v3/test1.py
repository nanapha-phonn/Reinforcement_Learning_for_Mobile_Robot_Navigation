import pybullet as p
import pybullet_data
import math
import robot


# ============================================================
# V3.15.2.5
# FINAL REAL-PHYSICS VALIDATION
#
# TESTS:
#   1. Static stability
#   2. Forward motion
#   3. Pure rotation
#
# Each test uses a FRESH PyBullet simulation.
#
# PPO / Gymnasium are NOT used.
# ============================================================


# ============================================================
# TEST PARAMETERS
# ============================================================

SETTLE_TIME = 2.0

TARGET_WHEEL_VELOCITY = 4.0
MOTOR_FORCE = robot.MOTOR_FORCE

FORWARD_TIME = 3.0
ROTATION_TIME = 3.0

EXPECTED_LINEAR_VELOCITY = (
    robot.WHEEL_RADIUS * TARGET_WHEEL_VELOCITY
)

EXPECTED_FORWARD_DISTANCE = (
    EXPECTED_LINEAR_VELOCITY * FORWARD_TIME
)

EXPECTED_YAW_RATE = (
    robot.WHEEL_RADIUS
    * (2.0 * TARGET_WHEEL_VELOCITY)
    / robot.WHEEL_BASE
)

TOTAL_MASS = (
    robot.CHASSIS_MASS
    + 2.0 * robot.WHEEL_MASS
    + robot.CASTER_FORK_MASS
    + robot.CASTER_WHEEL_MASS
)

EXPECTED_WEIGHT = TOTAL_MASS * abs(robot.GRAVITY)


# ============================================================
# VALIDATION TOLERANCES
# ============================================================

WHEEL_VELOCITY_TOLERANCE = 0.50

FORWARD_DISTANCE_MIN = 0.30
LATERAL_DRIFT_MAX = 0.10

ROTATION_RATIO_MIN = 0.80

ROLL_MAX = 5.0
PITCH_MAX = 5.0

FORCE_ERROR_MAX = 2.0

MIN_CONTACTS = 1
MIN_CASTER_LOAD = 1.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def connect_simulation():

    client = p.connect(p.DIRECT)

    if client < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        robot.GRAVITY
    )

    p.setTimeStep(
        robot.TIME_STEP
    )

    p.setPhysicsEngineParameter(
        numSolverIterations=100
    )

    p.loadURDF(
        "plane.urdf"
    )

    return client


def step_simulation(seconds):

    steps = int(
        seconds / robot.TIME_STEP
    )

    for _ in range(steps):
        p.stepSimulation()


def get_orientation_degrees(body_id):

    _, quaternion = p.getBasePositionAndOrientation(
        body_id
    )

    roll, pitch, yaw = p.getEulerFromQuaternion(
        quaternion
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def get_contacts_and_forces(robot_id):

    contact_data = {
        "left": 0,
        "right": 0,
        "caster": 0
    }

    normal_forces = {
        "left": 0.0,
        "right": 0.0,
        "caster": 0.0
    }

    # --------------------------------------------------------
    # All contacts with the plane.
    #
    # plane body ID = 0
    # robot body ID = robot_id
    # --------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=0
    )

    for contact in contacts:

        link_index = contact[3]
        normal_force = contact[9]

        if link_index == robot.LEFT_WHEEL_JOINT:

            contact_data["left"] += 1
            normal_forces["left"] += normal_force

        elif link_index == robot.RIGHT_WHEEL_JOINT:

            contact_data["right"] += 1
            normal_forces["right"] += normal_force

        elif link_index == robot.CASTER_WHEEL_JOINT:

            contact_data["caster"] += 1
            normal_forces["caster"] += normal_force

    total_force = (
        normal_forces["left"]
        + normal_forces["right"]
        + normal_forces["caster"]
    )

    return (
        contact_data,
        normal_forces,
        total_force
    )


def get_wheel_velocities(robot_id):

    left_state = p.getJointState(
        robot_id,
        robot.LEFT_WHEEL_JOINT
    )

    right_state = p.getJointState(
        robot_id,
        robot.RIGHT_WHEEL_JOINT
    )

    left_velocity = left_state[1]
    right_velocity = right_state[1]

    return (
        left_velocity,
        right_velocity
    )


def print_separator():

    print(
        "-" * 70
    )


# ============================================================
# TEST 1
# STATIC STABILITY
# ============================================================

def test_static():

    print()
    print("=" * 70)
    print("TEST 1 — STATIC STABILITY")
    print("=" * 70)

    client = connect_simulation()

    robot_id = robot.create_robot()

    print()
    print("Settling robot...")

    step_simulation(
        SETTLE_TIME
    )

    position, _ = p.getBasePositionAndOrientation(
        robot_id
    )

    roll, pitch, yaw = get_orientation_degrees(
        robot_id
    )

    contacts, forces, total_force = (
        get_contacts_and_forces(robot_id)
    )

    print()
    print("STATIC STATE")
    print(
        f"Position: "
        f"({position[0]:.6f}, "
        f"{position[1]:.6f}, "
        f"{position[2]:.6f})"
    )

    print(
        f"Roll:  {roll:.4f} deg"
    )

    print(
        f"Pitch: {pitch:.4f} deg"
    )

    print(
        f"Yaw:   {yaw:.4f} deg"
    )

    print()
    print("CONTACTS")

    print(
        f"Left wheel : "
        f"{contacts['left']}"
    )

    print(
        f"Right wheel: "
        f"{contacts['right']}"
    )

    print(
        f"Caster     : "
        f"{contacts['caster']}"
    )

    print()
    print("NORMAL FORCES")

    print(
        f"Left wheel : "
        f"{forces['left']:.6f} N"
    )

    print(
        f"Right wheel: "
        f"{forces['right']:.6f} N"
    )

    print(
        f"Caster     : "
        f"{forces['caster']:.6f} N"
    )

    print(
        f"Total      : "
        f"{total_force:.6f} N"
    )

    print(
        f"Expected   : "
        f"{EXPECTED_WEIGHT:.6f} N"
    )

    force_error = (
        total_force
        - EXPECTED_WEIGHT
    )

    print(
        f"Force error: "
        f"{force_error:.6f} N"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    checks = {}

    checks["Roll"] = (
        abs(roll) <= ROLL_MAX
    )

    checks["Pitch"] = (
        abs(pitch) <= PITCH_MAX
    )

    checks["Yaw"] = (
        abs(yaw) <= 5.0
    )

    checks["Left wheel contact"] = (
        contacts["left"] >= MIN_CONTACTS
    )

    checks["Right wheel contact"] = (
        contacts["right"] >= MIN_CONTACTS
    )

    checks["Caster contact"] = (
        contacts["caster"] >= MIN_CONTACTS
    )

    checks["Caster load"] = (
        forces["caster"] >= MIN_CASTER_LOAD
    )

    checks["Weight balance"] = (
        abs(force_error) <= FORCE_ERROR_MAX
    )

    print()
    print("VALIDATION")

    for name, passed in checks.items():

        print(
            f"{name:<25} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall = all(
        checks.values()
    )

    print()
    print(
        "STATIC RESULT:",
        "PASS" if overall else "FAIL"
    )

    p.disconnect(
        client
    )

    return overall


# ============================================================
# TEST 2
# FORWARD MOTION
# ============================================================

def test_forward():

    print()
    print("=" * 70)
    print("TEST 2 — FORWARD MOTION")
    print("=" * 70)

    client = connect_simulation()

    robot_id = robot.create_robot()

    print()
    print("Settling robot...")

    step_simulation(
        SETTLE_TIME
    )

    start_position, _ = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    start_x = start_position[0]
    start_y = start_position[1]

    print()
    print(
        f"Target wheel velocity: "
        f"{TARGET_WHEEL_VELOCITY:.3f} rad/s"
    )

    print(
        f"Theoretical linear velocity: "
        f"{EXPECTED_LINEAR_VELOCITY:.6f} m/s"
    )

    print(
        f"Theoretical distance: "
        f"{EXPECTED_FORWARD_DISTANCE:.6f} m"
    )

    # --------------------------------------------------------
    # Drive
    # --------------------------------------------------------

    robot.set_wheel_velocity(
        robot_id,
        TARGET_WHEEL_VELOCITY,
        TARGET_WHEEL_VELOCITY
    )

    samples = []

    steps = int(
        FORWARD_TIME
        / robot.TIME_STEP
    )

    print()
    print("DRIVING...")

    for step in range(steps):

        p.stepSimulation()

        left_velocity, right_velocity = (
            get_wheel_velocities(
                robot_id
            )
        )

        position, _ = (
            p.getBasePositionAndOrientation(
                robot_id
            )
        )

        samples.append(
            (
                left_velocity,
                right_velocity,
                position[0],
                position[1]
            )
        )

        if step % 120 == 0:

            elapsed = (
                step
                * robot.TIME_STEP
            )

            print(
                f"t={elapsed:.2f}s | "
                f"L={left_velocity:.3f} | "
                f"R={right_velocity:.3f} | "
                f"X={position[0]:.4f} | "
                f"Y={position[1]:.4f}"
            )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_position, _ = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    final_x = final_position[0]
    final_y = final_position[1]

    dx = final_x - start_x
    dy = final_y - start_y

    left_average = sum(
        sample[0]
        for sample in samples
    ) / len(samples)

    right_average = sum(
        sample[1]
        for sample in samples
    ) / len(samples)

    roll, pitch, yaw = (
        get_orientation_degrees(
            robot_id
        )
    )

    contacts, forces, total_force = (
        get_contacts_and_forces(
            robot_id
        )
    )

    print()
    print("FORWARD RESULTS")

    print(
        f"Start X: "
        f"{start_x:.6f} m"
    )

    print(
        f"Final X: "
        f"{final_x:.6f} m"
    )

    print(
        f"X displacement: "
        f"{dx:.6f} m"
    )

    print(
        f"Y displacement: "
        f"{dy:.6f} m"
    )

    print()
    print(
        f"Average left wheel: "
        f"{left_average:.6f} rad/s"
    )

    print(
        f"Average right wheel: "
        f"{right_average:.6f} rad/s"
    )

    print()
    print(
        f"Roll:  {roll:.6f} deg"
    )

    print(
        f"Pitch: {pitch:.6f} deg"
    )

    print(
        f"Yaw:   {yaw:.6f} deg"
    )

    print()
    print(
        f"Left contacts: "
        f"{contacts['left']}"
    )

    print(
        f"Right contacts: "
        f"{contacts['right']}"
    )

    print(
        f"Caster contacts: "
        f"{contacts['caster']}"
    )

    print()
    print(
        f"Total normal force: "
        f"{total_force:.6f} N"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    checks = {}

    checks["Left wheel velocity"] = (
        abs(
            left_average
            - TARGET_WHEEL_VELOCITY
        )
        <= WHEEL_VELOCITY_TOLERANCE
    )

    checks["Right wheel velocity"] = (
        abs(
            right_average
            - TARGET_WHEEL_VELOCITY
        )
        <= WHEEL_VELOCITY_TOLERANCE
    )

    checks["Forward displacement"] = (
        dx >= FORWARD_DISTANCE_MIN
    )

    checks["Lateral drift"] = (
        abs(dy) <= LATERAL_DRIFT_MAX
    )

    checks["Roll"] = (
        abs(roll) <= ROLL_MAX
    )

    checks["Pitch"] = (
        abs(pitch) <= PITCH_MAX
    )

    checks["Left wheel contact"] = (
        contacts["left"] >= MIN_CONTACTS
    )

    checks["Right wheel contact"] = (
        contacts["right"] >= MIN_CONTACTS
    )

    checks["Caster contact"] = (
        contacts["caster"] >= MIN_CONTACTS
    )

    checks["Caster load"] = (
        forces["caster"] >= MIN_CASTER_LOAD
    )

    checks["Weight balance"] = (
        abs(
            total_force
            - EXPECTED_WEIGHT
        )
        <= FORCE_ERROR_MAX
    )

    print()
    print("VALIDATION")

    for name, passed in checks.items():

        print(
            f"{name:<25} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall = all(
        checks.values()
    )

    print()
    print(
        "FORWARD RESULT:",
        "PASS" if overall else "FAIL"
    )

    # Stop motors before disconnecting.
    robot.stop_robot(
        robot_id
    )

    p.disconnect(
        client
    )

    return overall


# ============================================================
# TEST 3
# PURE ROTATION
# ============================================================

def test_rotation():

    print()
    print("=" * 70)
    print("TEST 3 — PURE ROTATION")
    print("=" * 70)

    client = connect_simulation()

    robot_id = robot.create_robot()

    print()
    print("Settling robot...")

    step_simulation(
        SETTLE_TIME
    )

    start_position, start_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    _, _, start_yaw = (
        p.getEulerFromQuaternion(
            start_orientation
        )
    )

    # --------------------------------------------------------
    # Theoretical yaw rate
    #
    # omega = r(vR - vL) / L
    #
    # For:
    #   vL = -4 rad/s
    #   vR = +4 rad/s
    #
    # omega = r(4 - (-4)) / L
    # --------------------------------------------------------

    theoretical_yaw_rate = (
        EXPECTED_YAW_RATE
    )

    print()
    print(
        f"Theoretical yaw rate: "
        f"{theoretical_yaw_rate:.6f} rad/s"
    )

    print(
        f"Command left: "
        f"{-TARGET_WHEEL_VELOCITY:.3f} rad/s"
    )

    print(
        f"Command right: "
        f"{TARGET_WHEEL_VELOCITY:.3f} rad/s"
    )

    # --------------------------------------------------------
    # Drive
    # --------------------------------------------------------

    robot.set_wheel_velocity(
        robot_id,
        -TARGET_WHEEL_VELOCITY,
        TARGET_WHEEL_VELOCITY
    )

    samples = []

    steps = int(
        ROTATION_TIME
        / robot.TIME_STEP
    )

    print()
    print("ROTATING...")

    for step in range(steps):

        p.stepSimulation()

        left_velocity, right_velocity = (
            get_wheel_velocities(
                robot_id
            )
        )

        position, orientation = (
            p.getBasePositionAndOrientation(
                robot_id
            )
        )

        angular_velocity = (
            p.getBaseVelocity(
                robot_id
            )[1]
        )

        roll, pitch, yaw = (
            p.getEulerFromQuaternion(
                orientation
            )
        )

        samples.append(
            {
                "left": left_velocity,
                "right": right_velocity,
                "x": position[0],
                "y": position[1],
                "yaw": yaw,
                "roll": math.degrees(roll),
                "pitch": math.degrees(pitch),
                "yaw_rate": angular_velocity[2]
            }
        )

        if step % 120 == 0:

            elapsed = (
                step
                * robot.TIME_STEP
            )

            print(
                f"t={elapsed:.2f}s | "
                f"L={left_velocity:.3f} | "
                f"R={right_velocity:.3f} | "
                f"yaw_rate={angular_velocity[2]:.3f} | "
                f"pitch={math.degrees(pitch):.2f}"
            )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_position, final_orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    final_x = final_position[0]
    final_y = final_position[1]

    final_roll, final_pitch, final_yaw = (
        p.getEulerFromQuaternion(
            final_orientation
        )
    )

    final_roll_deg = math.degrees(
        final_roll
    )

    final_pitch_deg = math.degrees(
        final_pitch
    )

    final_yaw_deg = math.degrees(
        final_yaw
    )

    left_average = sum(
        sample["left"]
        for sample in samples
    ) / len(samples)

    right_average = sum(
        sample["right"]
        for sample in samples
    ) / len(samples)

    yaw_rate_average = sum(
        sample["yaw_rate"]
        for sample in samples
    ) / len(samples)

    yaw_rate_final = (
        samples[-1]["yaw_rate"]
    )

    max_pitch = max(
        abs(sample["pitch"])
        for sample in samples
    )

    max_roll = max(
        abs(sample["roll"])
        for sample in samples
    )

    # --------------------------------------------------------
    # Rotation ratio
    # --------------------------------------------------------

    rotation_ratio = (
        abs(yaw_rate_average)
        / theoretical_yaw_rate
    )

    # --------------------------------------------------------
    # Contact / force state
    # --------------------------------------------------------

    contacts, forces, total_force = (
        get_contacts_and_forces(
            robot_id
        )
    )

    force_error = (
        total_force
        - EXPECTED_WEIGHT
    )

    # --------------------------------------------------------
    # Position movement
    # --------------------------------------------------------

    dx = (
        final_x
        - start_position[0]
    )

    dy = (
        final_y
        - start_position[1]
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("ROTATION RESULTS")

    print(
        f"Start position: "
        f"({start_position[0]:.6f}, "
        f"{start_position[1]:.6f})"
    )

    print(
        f"Final position: "
        f"({final_x:.6f}, "
        f"{final_y:.6f})"
    )

    print(
        f"X displacement: "
        f"{dx:.6f} m"
    )

    print(
        f"Y displacement: "
        f"{dy:.6f} m"
    )

    print()
    print(
        f"Average left wheel: "
        f"{left_average:.6f} rad/s"
    )

    print(
        f"Average right wheel: "
        f"{right_average:.6f} rad/s"
    )

    print()
    print(
        f"Theoretical yaw rate: "
        f"{theoretical_yaw_rate:.6f} rad/s"
    )

    print(
        f"Average yaw rate: "
        f"{yaw_rate_average:.6f} rad/s"
    )

    print(
        f"Final yaw rate: "
        f"{yaw_rate_final:.6f} rad/s"
    )

    print(
        f"Rotation ratio: "
        f"{rotation_ratio * 100.0:.2f}%"
    )

    print()
    print(
        f"Final roll: "
        f"{final_roll_deg:.6f} deg"
    )

    print(
        f"Final pitch: "
        f"{final_pitch_deg:.6f} deg"
    )

    print(
        f"Final yaw: "
        f"{final_yaw_deg:.6f} deg"
    )

    print(
        f"Maximum |roll|: "
        f"{max_roll:.6f} deg"
    )

    print(
        f"Maximum |pitch|: "
        f"{max_pitch:.6f} deg"
    )

    print()
    print("CONTACTS")

    print(
        f"Left wheel: "
        f"{contacts['left']}"
    )

    print(
        f"Right wheel: "
        f"{contacts['right']}"
    )

    print(
        f"Caster: "
        f"{contacts['caster']}"
    )

    print()
    print("NORMAL FORCES")

    print(
        f"Left wheel: "
        f"{forces['left']:.6f} N"
    )

    print(
        f"Right wheel: "
        f"{forces['right']:.6f} N"
    )

    print(
        f"Caster: "
        f"{forces['caster']:.6f} N"
    )

    print(
        f"Total: "
        f"{total_force:.6f} N"
    )

    print(
        f"Expected: "
        f"{EXPECTED_WEIGHT:.6f} N"
    )

    print(
        f"Force error: "
        f"{force_error:.6f} N"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    checks = {}

    checks["Left wheel velocity"] = (
        abs(
            left_average
            + TARGET_WHEEL_VELOCITY
        )
        <= WHEEL_VELOCITY_TOLERANCE
    )

    checks["Right wheel velocity"] = (
        abs(
            right_average
            - TARGET_WHEEL_VELOCITY
        )
        <= WHEEL_VELOCITY_TOLERANCE
    )

    checks["Rotation ratio"] = (
        rotation_ratio
        >= ROTATION_RATIO_MIN
    )

    checks["Roll"] = (
        max_roll
        <= ROLL_MAX
    )

    checks["Pitch"] = (
        max_pitch
        <= PITCH_MAX
    )

    checks["Left wheel contact"] = (
        contacts["left"]
        >= MIN_CONTACTS
    )

    checks["Right wheel contact"] = (
        contacts["right"]
        >= MIN_CONTACTS
    )

    checks["Caster contact"] = (
        contacts["caster"]
        >= MIN_CONTACTS
    )

    checks["Weight balance"] = (
        abs(force_error)
        <= FORCE_ERROR_MAX
    )

    print()
    print("VALIDATION")

    for name, passed in checks.items():

        print(
            f"{name:<25} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    overall = all(
        checks.values()
    )

    print()
    print(
        "ROTATION RESULT:",
        "PASS" if overall else "FAIL"
    )

    robot.stop_robot(
        robot_id
    )

    p.disconnect(
        client
    )

    return overall


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V3.15.2.5 REAL-PHYSICS VALIDATION")
    print("=" * 70)

    print()
    print("Robot model:")
    print("  Differential-drive")
    print("  Two powered wheels")
    print("  Passive swivel caster")
    print("  Real PyBullet rigid-body dynamics")

    print()
    print("Expected total mass:")
    print(
        f"  {TOTAL_MASS:.3f} kg"
    )

    print(
        f"Expected weight:"
        f" {EXPECTED_WEIGHT:.6f} N"
    )

    print()
    print("Wheel radius:")
    print(
        f"  {robot.WHEEL_RADIUS:.3f} m"
    )

    print(
        f"Wheel base:"
        f" {robot.WHEEL_BASE:.3f} m"
    )

    print(
        f"Motor force:"
        f" {MOTOR_FORCE:.3f}"
    )

    print_separator()

    static_pass = test_static()

    print_separator()

    forward_pass = test_forward()

    print_separator()

    rotation_pass = test_rotation()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL V3.15.2.5 VALIDATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Static stability : "
        f"{'PASS' if static_pass else 'FAIL'}"
    )

    print(
        f"Forward motion   : "
        f"{'PASS' if forward_pass else 'FAIL'}"
    )

    print(
        f"Pure rotation    : "
        f"{'PASS' if rotation_pass else 'FAIL'}"
    )

    overall = (
        static_pass
        and forward_pass
        and rotation_pass
    )

    print()

    if overall:

        print("=" * 70)
        print("OVERALL RESULT: PASS")
        print("=" * 70)

    else:

        print("=" * 70)
        print("OVERALL RESULT: FAIL")
        print("=" * 70)

        print()
        print(
            "Do NOT connect PPO yet."
        )

        print(
            "Fix the failed physics validation first."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("TEST1 STARTED")

    main()