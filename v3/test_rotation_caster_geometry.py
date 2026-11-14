import pybullet as p
import pybullet_data
import math
import csv

import robot


# ============================================================
# V3.17.4 CASTER GEOMETRY SENSITIVITY
# ============================================================

LEFT_COMMAND = -4.0
RIGHT_COMMAND = 4.0

MOTOR_FORCE = 20.0
ROTATION_TIME = 3.0

SAMPLE_INTERVAL = 0.10

TRAIL_VALUES = [
    0.03,
    0.05,
    0.07,
    0.10,
]

CASTER_X = -0.15

THEORETICAL_YAW_RATE = (
    robot.WHEEL_RADIUS
    * (RIGHT_COMMAND - LEFT_COMMAND)
    / robot.WHEEL_BASE
)


# ============================================================
# ANGLE HELPER
# ============================================================

def wrap_angle_deg(angle):

    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


# ============================================================
# CREATE ROBOT WITH MODIFIED CASTER TRAIL
# ============================================================

def create_robot_with_trail(trail):

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=robot.WHEEL_RADIUS,
        height=robot.WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=robot.WHEEL_RADIUS,
        length=robot.WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    chassis_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            robot.CHASSIS_LENGTH / 2,
            robot.CHASSIS_WIDTH / 2,
            robot.CHASSIS_HEIGHT / 2
        ]
    )

    chassis_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            robot.CHASSIS_LENGTH / 2,
            robot.CHASSIS_WIDTH / 2,
            robot.CHASSIS_HEIGHT / 2
        ]
    )

    caster_fork_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.020
        ]
    )

    caster_fork_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.025,
            0.025,
            0.020
        ]
    )

    caster_wheel_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=robot.CASTER_WHEEL_RADIUS,
        height=robot.CASTER_WHEEL_WIDTH,
        collisionFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    caster_wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=robot.CASTER_WHEEL_RADIUS,
        length=robot.CASTER_WHEEL_WIDTH,
        visualFrameOrientation=p.getQuaternionFromEuler(
            [math.pi / 2, 0, 0]
        )
    )

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base_position = [
        0,
        0,
        robot.WHEEL_RADIUS
        + robot.CHASSIS_HEIGHT / 2
    ]

    # --------------------------------------------------------
    # Link positions
    # --------------------------------------------------------

    left_wheel_position = [
        0,
        robot.WHEEL_BASE / 2,
        -robot.CHASSIS_HEIGHT / 2
    ]

    right_wheel_position = [
        0,
        -robot.WHEEL_BASE / 2,
        -robot.CHASSIS_HEIGHT / 2
    ]

    caster_fork_position = [
        CASTER_X,
        0,
        -robot.CHASSIS_HEIGHT / 2
    ]

    caster_wheel_position = [
        -trail,
        0,
        -0.04
    ]

    # --------------------------------------------------------
    # Multibody
    #
    # IMPORTANT:
    # Keep the same parent-index convention validated in
    # V3.15.2.5.
    # --------------------------------------------------------

    robot_id = p.createMultiBody(

        baseMass=robot.CHASSIS_MASS,

        baseCollisionShapeIndex=chassis_collision,

        baseVisualShapeIndex=chassis_visual,

        basePosition=base_position,

        baseOrientation=[
            0,
            0,
            0,
            1
        ],

        linkMasses=[
            robot.WHEEL_MASS,
            robot.WHEEL_MASS,
            robot.CASTER_FORK_MASS,
            robot.CASTER_WHEEL_MASS
        ],

        linkCollisionShapeIndices=[
            wheel_collision,
            wheel_collision,
            caster_fork_collision,
            caster_wheel_collision
        ],

        linkVisualShapeIndices=[
            wheel_visual,
            wheel_visual,
            caster_fork_visual,
            caster_wheel_visual
        ],

        linkPositions=[
            left_wheel_position,
            right_wheel_position,
            caster_fork_position,
            caster_wheel_position
        ],

        linkOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],

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
        ],

        linkParentIndices=[
            0,
            0,
            0,
            2
        ],

        linkJointTypes=[
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE,
            p.JOINT_REVOLUTE
        ],

        linkJointAxis=[
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=0.05,
        rollingFriction=0,
        spinningFriction=0
    )

    p.changeDynamics(
        robot_id,
        robot.LEFT_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0,
        spinningFriction=0
    )

    p.changeDynamics(
        robot_id,
        robot.RIGHT_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0,
        spinningFriction=0
    )

    p.changeDynamics(
        robot_id,
        robot.CASTER_SWIVEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0,
        spinningFriction=0
    )

    p.changeDynamics(
        robot_id,
        robot.CASTER_WHEEL_JOINT,
        lateralFriction=0.5,
        rollingFriction=0,
        spinningFriction=0
    )

    # --------------------------------------------------------
    # Disable self collisions
    # --------------------------------------------------------

    for i in range(4):

        p.setCollisionFilterPair(
            robot_id,
            robot_id,
            -1,
            i,
            0
        )

    # --------------------------------------------------------
    # Disable caster motors
    # --------------------------------------------------------

    p.setJointMotorControl2(
        robot_id,
        robot.CASTER_SWIVEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    p.setJointMotorControl2(
        robot_id,
        robot.CASTER_WHEEL_JOINT,
        p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0
    )

    return robot_id


# ============================================================
# RUN ONE TRAIL CASE
# ============================================================

def run_case(trail):

    print()
    print("=" * 80)
    print(f"TEST CASE: CASTER_TRAIL = {trail:.3f} m")
    print("=" * 80)

    p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        robot.GRAVITY
    )

    p.setTimeStep(
        robot.TIME_STEP
    )

    plane_id = p.loadURDF(
        "plane.urdf"
    )

    robot_id = create_robot_with_trail(
        trail
    )

    # --------------------------------------------------------
    # Settle
    # --------------------------------------------------------

    settle_time = 2.0

    for _ in range(
        int(
            settle_time
            / robot.TIME_STEP
        )
    ):

        p.stepSimulation()

    # --------------------------------------------------------
    # Sample helper
    # --------------------------------------------------------

    def sample():

        base_pos, base_orn = (
            p.getBasePositionAndOrientation(
                robot_id
            )
        )

        base_roll, base_pitch, base_yaw = (
            p.getEulerFromQuaternion(
                base_orn
            )
        )

        caster_state = p.getLinkState(
            robot_id,
            robot.CASTER_WHEEL_JOINT,
            computeLinkVelocity=1
        )

        caster_pos = caster_state[0]

        caster_orn = caster_state[1]

        caster_yaw = p.getEulerFromQuaternion(
            caster_orn
        )[2]

        swivel_state = p.getJointState(
            robot_id,
            robot.CASTER_SWIVEL_JOINT
        )

        swivel_angle = swivel_state[0]
        swivel_velocity = swivel_state[1]

        wheel_state = p.getJointState(
            robot_id,
            robot.CASTER_WHEEL_JOINT
        )

        wheel_velocity = wheel_state[1]

        dx = (
            caster_pos[0]
            - base_pos[0]
        )

        dy = (
            caster_pos[1]
            - base_pos[1]
        )

        radial_distance = math.sqrt(
            dx * dx + dy * dy
        )

        caster_position_angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        base_yaw_deg = math.degrees(
            base_yaw
        )

        radial_relative = wrap_angle_deg(
            caster_position_angle
            - base_yaw_deg
        )

        trailing_error = wrap_angle_deg(
            radial_relative - 180.0
        )

        swivel_deg = math.degrees(
            swivel_angle
        )

        swivel_relative = wrap_angle_deg(
            swivel_deg
            - base_yaw_deg
        )

        contacts = p.getContactPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            linkIndexA=robot.CASTER_WHEEL_JOINT
        )

        contact_count = len(contacts)

        normal_force = sum(
            c[9]
            for c in contacts
        )

        closest = p.getClosestPoints(
            bodyA=robot_id,
            bodyB=plane_id,
            distance=0.20,
            linkIndexA=robot.CASTER_WHEEL_JOINT
        )

        if closest:

            ground_distance = min(
                c[8]
                for c in closest
            )

        else:

            ground_distance = float("nan")

        return {
            "base_yaw": base_yaw_deg,
            "caster_x": caster_pos[0],
            "caster_y": caster_pos[1],
            "caster_z": caster_pos[2],
            "radial_distance": radial_distance,
            "trailing_error": trailing_error,
            "swivel_angle": swivel_deg,
            "swivel_relative": swivel_relative,
            "swivel_velocity": swivel_velocity,
            "wheel_velocity": wheel_velocity,
            "caster_yaw": math.degrees(
                caster_yaw
            ),
            "contacts": contact_count,
            "normal_force": normal_force,
            "ground_distance": ground_distance
        }

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    total_steps = int(
        ROTATION_TIME
        / robot.TIME_STEP
    )

    sample_steps = max(
        1,
        int(
            SAMPLE_INTERVAL
            / robot.TIME_STEP
        )
    )

    samples = []

    for step in range(
        total_steps
    ):

        robot.set_wheel_velocity(
            robot_id,
            LEFT_COMMAND,
            RIGHT_COMMAND
        )

        p.stepSimulation()

        if (
            step % sample_steps == 0
            or step == total_steps - 1
        ):

            data = sample()

            data["time"] = (
                (step + 1)
                * robot.TIME_STEP
            )

            samples.append(data)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    radial = [
        s["radial_distance"]
        for s in samples
    ]

    trailing = [
        abs(s["trailing_error"])
        for s in samples
    ]

    ground = [
        s["ground_distance"]
        for s in samples
        if not math.isnan(
            s["ground_distance"]
        )
    ]

    forces = [
        s["normal_force"]
        for s in samples
    ]

    contacts = [
        s["contacts"]
        for s in samples
    ]

    swivel_velocity = [
        abs(s["swivel_velocity"])
        for s in samples
    ]

    contact_samples = sum(
        1
        for c in contacts
        if c > 0
    )

    result = {

        "trail": trail,

        "wheel_x":
            CASTER_X - trail,

        "initial_radius":
            radial[0],

        "final_radius":
            radial[-1],

        "min_radius":
            min(radial),

        "max_radius":
            max(radial),

        "radius_variation":
            max(radial) - min(radial),

        "max_trailing_error":
            max(trailing),

        "avg_trailing_error":
            sum(trailing)
            / len(trailing),

        "min_z":
            min(
                s["caster_z"]
                for s in samples
            ),

        "max_z":
            max(
                s["caster_z"]
                for s in samples
            ),

        "max_ground_gap":
            max(ground),

        "contact_percentage":
            100.0
            * contact_samples
            / len(samples),

        "max_normal_force":
            max(forces),

        "max_swivel_velocity":
            max(swivel_velocity),

        "final_swivel":
            samples[-1]["swivel_angle"]
    }

    # --------------------------------------------------------
    # Print case
    # --------------------------------------------------------

    print()
    print(
        f"Wheel X position       : "
        f"{result['wheel_x']:.3f} m"
    )

    print(
        f"Initial radius         : "
        f"{result['initial_radius']:.6f} m"
    )

    print(
        f"Final radius           : "
        f"{result['final_radius']:.6f} m"
    )

    print(
        f"Minimum radius         : "
        f"{result['min_radius']:.6f} m"
    )

    print(
        f"Maximum radius         : "
        f"{result['max_radius']:.6f} m"
    )

    print(
        f"Radius variation       : "
        f"{result['radius_variation']:.6f} m"
    )

    print(
        f"Maximum trailing error : "
        f"{result['max_trailing_error']:.3f} deg"
    )

    print(
        f"Average trailing error : "
        f"{result['avg_trailing_error']:.3f} deg"
    )

    print(
        f"Maximum ground gap     : "
        f"{result['max_ground_gap']:.6f} m"
    )

    print(
        f"Contact percentage     : "
        f"{result['contact_percentage']:.2f}%"
    )

    print(
        f"Maximum normal force   : "
        f"{result['max_normal_force']:.6f} N"
    )

    print(
        f"Maximum swivel velocity: "
        f"{result['max_swivel_velocity']:.6f} rad/s"
    )

    print(
        f"Final swivel angle     : "
        f"{result['final_swivel']:.3f} deg"
    )

    p.disconnect()

    return result


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 80)
print("V3.17.4 CASTER GEOMETRY SENSITIVITY")
print("=" * 80)

print()
print("Only CASTER_TRAIL will change.")
print()
print(f"Fixed CASTER_X = {CASTER_X:.3f} m")
print(
    f"Trail cases = "
    f"{', '.join(f'{x:.2f}' for x in TRAIL_VALUES)} m"
)

print()
print(
    f"Theoretical yaw rate = "
    f"{THEORETICAL_YAW_RATE:.6f} rad/s"
)

results = []

for trail in TRAIL_VALUES:

    result = run_case(trail)

    results.append(result)


# ============================================================
# COMPARISON TABLE
# ============================================================

print()
print("=" * 80)
print("V3.17.4 COMPARISON")
print("=" * 80)

print()
print(
    "Trail | WheelX | "
    "RadVar | MaxTrailErr | "
    "Contact | Gap | MaxForce | MaxSwivVel"
)

print("-" * 100)

for r in results:

    print(
        f"{r['trail']:5.2f} | "
        f"{r['wheel_x']:6.2f} | "
        f"{r['radius_variation']:6.3f} | "
        f"{r['max_trailing_error']:11.2f} | "
        f"{r['contact_percentage']:7.2f}% | "
        f"{r['max_ground_gap']:5.3f} | "
        f"{r['max_normal_force']:8.3f} | "
        f"{r['max_swivel_velocity']:10.3f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

filename = (
    "v3_17_4_caster_geometry_results.csv"
)

with open(
    filename,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "trail",
        "wheel_x",
        "initial_radius",
        "final_radius",
        "min_radius",
        "max_radius",
        "radius_variation",
        "max_trailing_error",
        "avg_trailing_error",
        "min_z",
        "max_z",
        "max_ground_gap",
        "contact_percentage",
        "max_normal_force",
        "max_swivel_velocity",
        "final_swivel"
    ])

    for r in results:

        writer.writerow([
            r["trail"],
            r["wheel_x"],
            r["initial_radius"],
            r["final_radius"],
            r["min_radius"],
            r["max_radius"],
            r["radius_variation"],
            r["max_trailing_error"],
            r["avg_trailing_error"],
            r["min_z"],
            r["max_z"],
            r["max_ground_gap"],
            r["contact_percentage"],
            r["max_normal_force"],
            r["max_swivel_velocity"],
            r["final_swivel"]
        ])


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("CSV SAVED")
print("=" * 80)

print(filename)

print()
print("V3.17.4 complete.")
print()
print("No robot.py parameters were permanently changed.")
print("Each trail value was tested in a fresh simulation.")