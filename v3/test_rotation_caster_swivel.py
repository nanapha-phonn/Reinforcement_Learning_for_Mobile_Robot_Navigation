import pybullet as p
import pybullet_data
import math

import robot


# ============================================================
# V3.17.2 CASTER SWIVEL RESPONSE DIAGNOSTIC
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

print("=" * 75)
print("V3.17.2 CASTER SWIVEL RESPONSE DIAGNOSTIC")
print("=" * 75)

print()
print("Purpose:")
print("Determine whether the passive caster swivel is actually free")
print("during pure robot rotation.")
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
# INSPECT JOINTS
# ============================================================

print()
print("=" * 75)
print("CASTER JOINT CONFIGURATION")
print("=" * 75)

swivel_info = p.getJointInfo(
    robot_id,
    robot.CASTER_SWIVEL_JOINT
)

wheel_info = p.getJointInfo(
    robot_id,
    robot.CASTER_WHEEL_JOINT
)

print()
print("CASTER SWIVEL JOINT")
print(f"Joint index       : {robot.CASTER_SWIVEL_JOINT}")
print(f"Joint type        : {swivel_info[2]}")
print(f"Joint axis        : {swivel_info[13]}")
print(f"Parent index      : {swivel_info[16]}")
print(f"Lower limit       : {swivel_info[8]}")
print(f"Upper limit       : {swivel_info[9]}")

print()
print("CASTER WHEEL JOINT")
print(f"Joint index       : {robot.CASTER_WHEEL_JOINT}")
print(f"Joint type        : {wheel_info[2]}")
print(f"Joint axis        : {wheel_info[13]}")
print(f"Parent index      : {wheel_info[16]}")
print(f"Lower limit       : {wheel_info[8]}")
print(f"Upper limit       : {wheel_info[9]}")


# ============================================================
# EXPLICITLY DISABLE PASSIVE JOINT MOTORS
# ============================================================

# Important:
# We do NOT command these joints.
# These calls make sure no residual velocity controller
# is applying torque to the passive joints.

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
print("Passive caster motors explicitly disabled.")
print("Swivel motor force = 0")
print("Caster wheel motor force = 0")


# ============================================================
# SETTLE ROBOT
# ============================================================

SETTLE_TIME = 2.0

for _ in range(
    int(SETTLE_TIME / robot.TIME_STEP)
):
    p.stepSimulation()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_caster_data():

    # --------------------------------------------------------
    # CASTER WHEEL LINK STATE
    # --------------------------------------------------------

    link_state = p.getLinkState(
        robot_id,
        robot.CASTER_WHEEL_JOINT,
        computeLinkVelocity=1
    )

    pos = link_state[0]
    orn = link_state[1]

    linear_velocity = link_state[6]
    angular_velocity = link_state[7]

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(orn)
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

    swivel_reaction_force = swivel_state[2]
    swivel_reaction_torque = swivel_state[3]

    # --------------------------------------------------------
    # CASTER WHEEL JOINT
    # --------------------------------------------------------

    wheel_state = p.getJointState(
        robot_id,
        robot.CASTER_WHEEL_JOINT
    )

    wheel_angle = wheel_state[0]
    wheel_velocity = wheel_state[1]

    wheel_reaction_force = wheel_state[2]
    wheel_reaction_torque = wheel_state[3]

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

    return {
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],

        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),

        "vx": linear_velocity[0],
        "vy": linear_velocity[1],
        "vz": linear_velocity[2],

        "swivel_angle": math.degrees(swivel_angle),
        "swivel_velocity": swivel_velocity,

        # PyBullet returns reaction torque as a scalar
        # in this installed build.
        "swivel_torque": swivel_reaction_torque,

        "wheel_angle": math.degrees(wheel_angle),
        "wheel_velocity": wheel_velocity,

        # PyBullet returns reaction torque as a scalar
        # in this installed build.
        "wheel_torque": wheel_reaction_torque,

        "contacts": contact_count,
        "normal_force": normal_force,

        "ground_distance": ground_distance
    }


# ============================================================
# INITIAL STATE
# ============================================================

initial = get_caster_data()

print()
print("=" * 75)
print("INITIAL CASTER STATE")
print("=" * 75)

print(
    f"Caster position      : "
    f"({initial['x']:.6f}, "
    f"{initial['y']:.6f}, "
    f"{initial['z']:.6f})"
)

print(
    f"Swivel angle         : "
    f"{initial['swivel_angle']:.6f} deg"
)

print(
    f"Swivel velocity      : "
    f"{initial['swivel_velocity']:.6f} rad/s"
)

print(
    f"Wheel angle          : "
    f"{initial['wheel_angle']:.6f} deg"
)

print(
    f"Wheel velocity       : "
    f"{initial['wheel_velocity']:.6f} rad/s"
)

print(
    f"Swivel reaction T    : "
    f"{initial['swivel_torque']:.6f} N*m"
)

print(
    f"Wheel reaction T     : "
    f"{initial['wheel_torque']:.6f} N*m"
)

print(
    f"Contact count        : "
    f"{initial['contacts']}"
)

print(
    f"Normal force         : "
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
    int(SAMPLE_INTERVAL / robot.TIME_STEP)
)

samples = []

print()
print("=" * 75)
print("ROTATION START")
print("=" * 75)

for step in range(total_steps):

    # Drive wheels only.
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

        t = (
            (step + 1)
            * robot.TIME_STEP
        )

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

        caster = get_caster_data()

        sample = {
            "time": t,

            "base_x": base_pos[0],
            "base_y": base_pos[1],

            "base_roll":
                math.degrees(base_roll),

            "base_pitch":
                math.degrees(base_pitch),

            "base_yaw":
                math.degrees(base_yaw),

            "base_wz":
                base_ang_vel[2],

            **caster
        }

        samples.append(sample)


# ============================================================
# TABLE
# ============================================================

print()
print("=" * 75)
print("CASTER SWIVEL TIME HISTORY")
print("=" * 75)

print(
    "Time | Swivel° | SwivelVel | SwivelTq | "
    "Wheel° | WheelVel | Contact | Force | Caster X | Caster Y | Caster Z"
)

print("-" * 130)

for s in samples:

    print(
        f"{s['time']:4.2f} | "
        f"{s['swivel_angle']:8.3f} | "
        f"{s['swivel_velocity']:9.5f} | "
        f"{s['swivel_torque']:9.5f} | "
        f"{s['wheel_angle']:8.3f} | "
        f"{s['wheel_velocity']:8.5f} | "
        f"{s['contacts']:7d} | "
        f"{s['normal_force']:6.2f} | "
        f"{s['x']:8.4f} | "
        f"{s['y']:8.4f} | "
        f"{s['z']:8.4f}"
    )


# ============================================================
# ANALYSIS
# ============================================================

swivel_angles = [
    s["swivel_angle"]
    for s in samples
]

swivel_velocities = [
    abs(s["swivel_velocity"])
    for s in samples
]

wheel_velocities = [
    abs(s["wheel_velocity"])
    for s in samples
]

swivel_torques = [
    abs(s["swivel_torque"])
    for s in samples
]

wheel_torques = [
    abs(s["wheel_torque"])
    for s in samples
]

forces = [
    s["normal_force"]
    for s in samples
]

contacts = [
    s["contacts"]
    for s in samples
]


initial_swivel = swivel_angles[0]
final_swivel = swivel_angles[-1]

swivel_change = (
    final_swivel - initial_swivel
)

max_swivel_change = (
    max(swivel_angles)
    - min(swivel_angles)
)

max_swivel_velocity = max(
    swivel_velocities
)

max_swivel_torque = max(
    swivel_torques
)

max_wheel_velocity = max(
    wheel_velocities
)

max_wheel_torque = max(
    wheel_torques
)

max_force = max(forces)

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


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("V3.17.2 SWIVEL RESPONSE SUMMARY")
print("=" * 75)

print()
print("SWIVEL")
print("-" * 75)

print(
    f"Initial angle              : "
    f"{initial_swivel:.6f} deg"
)

print(
    f"Final angle                : "
    f"{final_swivel:.6f} deg"
)

print(
    f"Net angle change           : "
    f"{swivel_change:.6f} deg"
)

print(
    f"Total angle range          : "
    f"{max_swivel_change:.6f} deg"
)

print(
    f"Maximum angular velocity   : "
    f"{max_swivel_velocity:.6f} rad/s"
)

print(
    f"Maximum reaction torque    : "
    f"{max_swivel_torque:.6f} N*m"
)

print()
print("CASTER WHEEL")
print("-" * 75)

print(
    f"Maximum wheel velocity     : "
    f"{max_wheel_velocity:.6f} rad/s"
)

print(
    f"Maximum wheel reaction T   : "
    f"{max_wheel_torque:.6f} N*m"
)

print()
print("GROUND CONTACT")
print("-" * 75)

print(
    f"Contact samples            : "
    f"{contact_samples}/{len(samples)}"
)

print(
    f"Contact percentage         : "
    f"{contact_percentage:.2f}%"
)

print(
    f"Maximum normal force       : "
    f"{max_force:.6f} N"
)

print()
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

if max_swivel_change > 5.0:

    print(
        "SWIVEL RESPONSE: ACTIVE"
    )

    print(
        "The caster is changing orientation "
        "during rotation."
    )

else:

    print(
        "SWIVEL RESPONSE: VERY SMALL"
    )

    print(
        "The caster swivel changes very little "
        "during rotation."
    )


if max_swivel_velocity > 0.05:

    print(
        "SWIVEL MOTION: DYNAMIC"
    )

else:

    print(
        "SWIVEL MOTION: VERY LOW"
    )


if max_wheel_velocity > 0.05:

    print(
        "CASTER WHEEL: ROTATING"
    )

else:

    print(
        "CASTER WHEEL: VERY LOW ROTATION"
    )


print()
print(
    "No robot geometry or drive parameters "
    "were modified."
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    "v3_17_2_caster_swivel_history.csv",
    "w"
) as f:

    f.write(
        "time,"
        "swivel_angle,"
        "swivel_velocity,"
        "swivel_torque,"
        "wheel_angle,"
        "wheel_velocity,"
        "wheel_torque,"
        "contact,"
        "normal_force,"
        "caster_x,"
        "caster_y,"
        "caster_z,"
        "ground_distance,"
        "base_x,"
        "base_y,"
        "base_pitch,"
        "base_yaw,"
        "base_wz\n"
    )

    for s in samples:

        f.write(
            f"{s['time']:.6f},"
            f"{s['swivel_angle']:.6f},"
            f"{s['swivel_velocity']:.6f},"
            f"{s['swivel_torque']:.6f},"
            f"{s['wheel_angle']:.6f},"
            f"{s['wheel_velocity']:.6f},"
            f"{s['wheel_torque']:.6f},"
            f"{s['contacts']},"
            f"{s['normal_force']:.6f},"
            f"{s['x']:.6f},"
            f"{s['y']:.6f},"
            f"{s['z']:.6f},"
            f"{s['ground_distance']:.6f},"
            f"{s['base_x']:.6f},"
            f"{s['base_y']:.6f},"
            f"{s['base_pitch']:.6f},"
            f"{s['base_yaw']:.6f},"
            f"{s['base_wz']:.6f}\n"
        )


print()
print("=" * 75)
print("CSV SAVED")
print("=" * 75)

print(
    "v3_17_2_caster_swivel_history.csv"
)

p.disconnect()