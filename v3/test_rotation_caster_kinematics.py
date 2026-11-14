import pybullet as p
import pybullet_data
import math
import csv

import robot


# ============================================================
# V3.17.3 CASTER KINEMATIC / CONTACT ANALYSIS
# ============================================================

LEFT_COMMAND = -4.0
RIGHT_COMMAND = 4.0

MOTOR_FORCE = 20.0
ROTATION_TIME = 3.0

SAMPLE_INTERVAL = 0.10

THEORETICAL_YAW_RATE = (
    robot.WHEEL_RADIUS
    * (RIGHT_COMMAND - LEFT_COMMAND)
    / robot.WHEEL_BASE
)

print("=" * 80)
print("V3.17.3 CASTER KINEMATIC / CONTACT ANALYSIS")
print("=" * 80)

print()
print("Purpose:")
print("Determine whether the passive caster behaves like a")
print("proper trailing caster during pure robot rotation.")
print()
print(f"Left wheel command      : {LEFT_COMMAND:.3f} rad/s")
print(f"Right wheel command     : {RIGHT_COMMAND:.3f} rad/s")
print(f"Motor force             : {MOTOR_FORCE:.3f}")
print(f"Rotation time           : {ROTATION_TIME:.3f} s")
print(f"Theoretical yaw rate    : {THEORETICAL_YAW_RATE:.6f} rad/s")


# ============================================================
# CONNECT
# ============================================================

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

plane_id = p.loadURDF("plane.urdf")

robot_id = robot.create_robot()

print()
print(f"Robot ID: {robot_id}")
print(f"Joint count: {p.getNumJoints(robot_id)}")


# ============================================================
# DISABLE PASSIVE CASTER MOTORS
# ============================================================

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

print()
print("Passive caster motors disabled.")
print("Swivel force = 0")
print("Caster wheel force = 0")


# ============================================================
# SETTLE
# ============================================================

SETTLE_TIME = 2.0

for _ in range(
    int(SETTLE_TIME / robot.TIME_STEP)
):
    p.stepSimulation()


# ============================================================
# HELPER
# ============================================================

def wrap_angle_deg(angle):

    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


def get_data():

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    base_pos, base_orn = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    base_lin_vel, base_ang_vel = (
        p.getBaseVelocity(
            robot_id
        )
    )

    base_roll, base_pitch, base_yaw = (
        p.getEulerFromQuaternion(
            base_orn
        )
    )

    # --------------------------------------------------------
    # CASTER WHEEL LINK
    # --------------------------------------------------------

    link_state = p.getLinkState(
        robot_id,
        robot.CASTER_WHEEL_JOINT,
        computeLinkVelocity=1
    )

    caster_pos = link_state[0]
    caster_orn = link_state[1]

    caster_lin_vel = link_state[6]
    caster_ang_vel = link_state[7]

    caster_roll, caster_pitch, caster_yaw = (
        p.getEulerFromQuaternion(
            caster_orn
        )
    )

    # --------------------------------------------------------
    # SWIVEL JOINT
    # --------------------------------------------------------

    swivel_state = p.getJointState(
        robot_id,
        robot.CASTER_SWIVEL_JOINT
    )

    swivel_angle = swivel_state[0]
    swivel_velocity = swivel_state[1]

    # --------------------------------------------------------
    # CASTER WHEEL JOINT
    # --------------------------------------------------------

    wheel_state = p.getJointState(
        robot_id,
        robot.CASTER_WHEEL_JOINT
    )

    wheel_angle = wheel_state[0]
    wheel_velocity = wheel_state[1]

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=robot.CASTER_WHEEL_JOINT
    )

    contact_count = len(contacts)

    normal_force = sum(
        contact[9]
        for contact in contacts
    )

    # --------------------------------------------------------
    # GROUND DISTANCE
    # --------------------------------------------------------

    closest = p.getClosestPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        distance=0.20,
        linkIndexA=robot.CASTER_WHEEL_JOINT
    )

    if closest:

        ground_distance = min(
            contact[8]
            for contact in closest
        )

    else:

        ground_distance = float("nan")

    # --------------------------------------------------------
    # CASTER RADIAL DISTANCE
    # --------------------------------------------------------

    dx = caster_pos[0] - base_pos[0]
    dy = caster_pos[1] - base_pos[1]

    radial_distance = math.sqrt(
        dx * dx + dy * dy
    )

    # --------------------------------------------------------
    # CASTER POSITION ANGLE FROM ROBOT CENTER
    # --------------------------------------------------------

    caster_position_angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    # --------------------------------------------------------
    # CASTER RADIAL ANGLE RELATIVE TO ROBOT YAW
    # --------------------------------------------------------

    radial_relative_angle = wrap_angle_deg(
        caster_position_angle
        - math.degrees(base_yaw)
    )

    # --------------------------------------------------------
    # SWIVEL ANGLE
    # --------------------------------------------------------

    swivel_deg = math.degrees(
        swivel_angle
    )

    # --------------------------------------------------------
    # SWIVEL RELATIVE TO ROBOT YAW
    # --------------------------------------------------------

    swivel_relative_to_yaw = wrap_angle_deg(
        swivel_deg
        - math.degrees(base_yaw)
    )

    # --------------------------------------------------------
    # CASTER ORIENTATION RELATIVE TO ROBOT
    # --------------------------------------------------------

    caster_orientation_relative = wrap_angle_deg(
        math.degrees(caster_yaw)
        - math.degrees(base_yaw)
    )

    # --------------------------------------------------------
    # TRAILING GEOMETRY ERROR
    # --------------------------------------------------------
    #
    # For a caster at the rear of the robot, the radial
    # direction from robot center to caster is approximately
    # 180 degrees relative to the robot's forward direction.
    #
    # This quantity tells us how far the caster radial
    # direction differs from the ideal rear direction.
    #

    trailing_error = wrap_angle_deg(
        radial_relative_angle - 180.0
    )

    return {

        # Base
        "base_x": base_pos[0],
        "base_y": base_pos[1],
        "base_yaw": math.degrees(base_yaw),
        "base_wz": base_ang_vel[2],

        # Caster position
        "caster_x": caster_pos[0],
        "caster_y": caster_pos[1],
        "caster_z": caster_pos[2],

        "caster_vx": caster_lin_vel[0],
        "caster_vy": caster_lin_vel[1],
        "caster_vz": caster_lin_vel[2],

        # Geometry
        "radial_distance": radial_distance,
        "caster_position_angle": caster_position_angle,
        "radial_relative_angle": radial_relative_angle,
        "trailing_error": trailing_error,

        # Swivel
        "swivel_angle": swivel_deg,
        "swivel_velocity": swivel_velocity,
        "swivel_relative_yaw":
            swivel_relative_to_yaw,

        # Wheel
        "wheel_angle":
            math.degrees(wheel_angle),
        "wheel_velocity":
            wheel_velocity,

        # Caster body orientation
        "caster_yaw":
            math.degrees(caster_yaw),

        "caster_orientation_relative":
            caster_orientation_relative,

        # Contact
        "contacts": contact_count,
        "normal_force": normal_force,
        "ground_distance": ground_distance
    }


# ============================================================
# INITIAL STATE
# ============================================================

initial = get_data()

print()
print("=" * 80)
print("INITIAL CASTER GEOMETRY")
print("=" * 80)

print(
    f"Caster position       : "
    f"({initial['caster_x']:.6f}, "
    f"{initial['caster_y']:.6f}, "
    f"{initial['caster_z']:.6f})"
)

print(
    f"Radial distance       : "
    f"{initial['radial_distance']:.6f} m"
)

print(
    f"Position angle        : "
    f"{initial['caster_position_angle']:.3f} deg"
)

print(
    f"Robot yaw             : "
    f"{initial['base_yaw']:.3f} deg"
)

print(
    f"Radial relative angle : "
    f"{initial['radial_relative_angle']:.3f} deg"
)

print(
    f"Trailing error        : "
    f"{initial['trailing_error']:.3f} deg"
)

print(
    f"Swivel angle          : "
    f"{initial['swivel_angle']:.3f} deg"
)

print(
    f"Swivel relative yaw   : "
    f"{initial['swivel_relative_yaw']:.3f} deg"
)

print(
    f"Ground distance       : "
    f"{initial['ground_distance']:.6f} m"
)

print(
    f"Normal force          : "
    f"{initial['normal_force']:.6f} N"
)


# ============================================================
# ROTATION
# ============================================================

total_steps = int(
    ROTATION_TIME / robot.TIME_STEP
)

sample_steps = max(
    1,
    int(
        SAMPLE_INTERVAL
        / robot.TIME_STEP
    )
)

samples = []

print()
print("=" * 80)
print("ROTATION START")
print("=" * 80)

for step in range(total_steps):

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

        data = get_data()

        data["time"] = (
            (step + 1)
            * robot.TIME_STEP
        )

        samples.append(data)


# ============================================================
# TABLE
# ============================================================

print()
print("=" * 80)
print("CASTER KINEMATIC / CONTACT HISTORY")
print("=" * 80)

print(
    "Time | BaseYaw | Swivel | "
    "SwivRel | RadialDist | "
    "TrailErr | CasterZ | Ground | "
    "Force | Contact"
)

print("-" * 115)

for s in samples:

    print(
        f"{s['time']:4.2f} | "
        f"{s['base_yaw']:8.2f} | "
        f"{s['swivel_angle']:7.2f} | "
        f"{s['swivel_relative_yaw']:8.2f} | "
        f"{s['radial_distance']:10.4f} | "
        f"{s['trailing_error']:8.2f} | "
        f"{s['caster_z']:7.4f} | "
        f"{s['ground_distance']:6.4f} | "
        f"{s['normal_force']:6.2f} | "
        f"{s['contacts']:7d}"
    )


# ============================================================
# ANALYSIS
# ============================================================

radial_distances = [
    s["radial_distance"]
    for s in samples
]

swivel_angles = [
    s["swivel_angle"]
    for s in samples
]

swivel_relative = [
    s["swivel_relative_yaw"]
    for s in samples
]

trailing_errors = [
    abs(s["trailing_error"])
    for s in samples
]

caster_z = [
    s["caster_z"]
    for s in samples
]

ground_distances = [
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


# ------------------------------------------------------------
# Radial distance
# ------------------------------------------------------------

initial_radius = radial_distances[0]
final_radius = radial_distances[-1]

min_radius = min(
    radial_distances
)

max_radius = max(
    radial_distances
)

radius_change = (
    max_radius
    - min_radius
)


# ------------------------------------------------------------
# Swivel
# ------------------------------------------------------------

initial_swivel = swivel_angles[0]
final_swivel = swivel_angles[-1]

swivel_range = (
    max(swivel_angles)
    - min(swivel_angles)
)


# ------------------------------------------------------------
# Relative swivel
# ------------------------------------------------------------

relative_swivel_range = (
    max(swivel_relative)
    - min(swivel_relative)
)


# ------------------------------------------------------------
# Trailing error
# ------------------------------------------------------------

maximum_trailing_error = max(
    trailing_errors
)

average_trailing_error = (
    sum(trailing_errors)
    / len(trailing_errors)
)


# ------------------------------------------------------------
# Height
# ------------------------------------------------------------

min_z = min(caster_z)
max_z = max(caster_z)

maximum_ground_distance = max(
    ground_distances
)


# ------------------------------------------------------------
# Contact
# ------------------------------------------------------------

contact_samples = sum(
    1
    for c in contacts
    if c > 0
)

contact_percentage = (
    100.0
    * contact_samples
    / len(samples)
)

max_force = max(
    forces
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("V3.17.3 KINEMATIC / CONTACT SUMMARY")
print("=" * 80)

print()
print("1. CASTER RADIAL MOTION")
print("-" * 80)

print(
    f"Initial radial distance     : "
    f"{initial_radius:.6f} m"
)

print(
    f"Final radial distance       : "
    f"{final_radius:.6f} m"
)

print(
    f"Minimum radial distance     : "
    f"{min_radius:.6f} m"
)

print(
    f"Maximum radial distance     : "
    f"{max_radius:.6f} m"
)

print(
    f"Radial distance variation   : "
    f"{radius_change:.6f} m"
)


print()
print("2. SWIVEL / ROBOT YAW")
print("-" * 80)

print(
    f"Initial swivel angle        : "
    f"{initial_swivel:.6f} deg"
)

print(
    f"Final swivel angle          : "
    f"{final_swivel:.6f} deg"
)

print(
    f"Swivel angle range          : "
    f"{swivel_range:.6f} deg"
)

print(
    f"Relative swivel range       : "
    f"{relative_swivel_range:.6f} deg"
)


print()
print("3. TRAILING GEOMETRY")
print("-" * 80)

print(
    f"Maximum trailing error      : "
    f"{maximum_trailing_error:.6f} deg"
)

print(
    f"Average trailing error      : "
    f"{average_trailing_error:.6f} deg"
)


print()
print("4. CASTER HEIGHT")
print("-" * 80)

print(
    f"Minimum caster Z            : "
    f"{min_z:.6f} m"
)

print(
    f"Maximum caster Z            : "
    f"{max_z:.6f} m"
)

print(
    f"Maximum ground distance     : "
    f"{maximum_ground_distance:.6f} m"
)


print()
print("5. GROUND CONTACT")
print("-" * 80)

print(
    f"Contact samples             : "
    f"{contact_samples}/{len(samples)}"
)

print(
    f"Contact percentage          : "
    f"{contact_percentage:.2f}%"
)

print(
    f"Maximum normal force        : "
    f"{max_force:.6f} N"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

print()
print("This test does NOT modify the robot.")

print()
print("The results will distinguish between:")

print()
print("A) PROPER TRAILING CASTER")
print(
    "   Caster remains approximately at a fixed radial distance,"
)
print(
    "   swivel follows the robot's changing direction,"
)
print(
    "   and ground contact remains stable."
)

print()
print("B) DYNAMICALLY SWINGING CASTER")
print(
    "   Caster radial distance changes substantially,"
)
print(
    "   swivel oscillates or overshoots,"
)
print(
    "   and ground contact repeatedly disappears."
)

print()
print("C) CONTACT-LIMITED CASTER")
print(
    "   Caster geometry remains reasonable,"
)
print(
    "   but wheel height/contact distance increases,"
)
print(
    "   causing intermittent loss of normal force."
)


# ============================================================
# SAVE CSV
# ============================================================

filename = (
    "v3_17_3_caster_kinematic_history.csv"
)

with open(
    filename,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "time",
        "base_x",
        "base_y",
        "base_yaw",
        "base_wz",
        "caster_x",
        "caster_y",
        "caster_z",
        "caster_vx",
        "caster_vy",
        "caster_vz",
        "radial_distance",
        "caster_position_angle",
        "radial_relative_angle",
        "trailing_error",
        "swivel_angle",
        "swivel_velocity",
        "swivel_relative_yaw",
        "caster_yaw",
        "caster_orientation_relative",
        "wheel_angle",
        "wheel_velocity",
        "contacts",
        "normal_force",
        "ground_distance"
    ])

    for s in samples:

        writer.writerow([
            s["time"],
            s["base_x"],
            s["base_y"],
            s["base_yaw"],
            s["base_wz"],
            s["caster_x"],
            s["caster_y"],
            s["caster_z"],
            s["caster_vx"],
            s["caster_vy"],
            s["caster_vz"],
            s["radial_distance"],
            s["caster_position_angle"],
            s["radial_relative_angle"],
            s["trailing_error"],
            s["swivel_angle"],
            s["swivel_velocity"],
            s["swivel_relative_yaw"],
            s["caster_yaw"],
            s["caster_orientation_relative"],
            s["wheel_angle"],
            s["wheel_velocity"],
            s["contacts"],
            s["normal_force"],
            s["ground_distance"]
        ])


print()
print("=" * 80)
print("CSV SAVED")
print("=" * 80)

print(filename)

p.disconnect()