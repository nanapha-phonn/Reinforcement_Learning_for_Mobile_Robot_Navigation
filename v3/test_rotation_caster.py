import pybullet as p
import pybullet_data
import math

import robot


# ============================================================
# V3.17.1 CASTER ROTATION DIAGNOSTIC
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

print("=" * 70)
print("V3.17.1 CASTER ROTATION DIAGNOSTIC")
print("=" * 70)

print()
print("The robot will rotate for 3 seconds.")
print("Caster behavior will be sampled every 0.10 seconds.")
print()
print(f"Left command       : {LEFT_COMMAND:.3f} rad/s")
print(f"Right command      : {RIGHT_COMMAND:.3f} rad/s")
print(f"Theoretical yaw    : {THEORETICAL_YAW_RATE:.6f} rad/s")


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

def get_caster_data():

    # --------------------------------------------------------
    # CASTER LINK STATE
    # --------------------------------------------------------

    caster_state = p.getLinkState(
        robot_id,
        robot.CASTER_WHEEL_JOINT,
        computeLinkVelocity=1
    )

    caster_pos = caster_state[0]
    caster_orn = caster_state[1]

    caster_lin_vel = caster_state[6]
    caster_ang_vel = caster_state[7]

    caster_roll, caster_pitch, caster_yaw = (
        p.getEulerFromQuaternion(caster_orn)
    )

    # --------------------------------------------------------
    # CASTER SWIVEL JOINT
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
    # CASTER CONTACT
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
    # CLOSEST GROUND DISTANCE
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
    # CASTER FORK CONTACT
    # --------------------------------------------------------

    fork_contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=plane_id,
        linkIndexA=robot.CASTER_SWIVEL_JOINT
    )

    fork_contact_count = len(fork_contacts)

    return {
        "x": caster_pos[0],
        "y": caster_pos[1],
        "z": caster_pos[2],

        "roll": math.degrees(caster_roll),
        "pitch": math.degrees(caster_pitch),
        "yaw": math.degrees(caster_yaw),

        "vx": caster_lin_vel[0],
        "vy": caster_lin_vel[1],
        "vz": caster_lin_vel[2],

        "swivel_angle": math.degrees(swivel_angle),
        "swivel_velocity": swivel_velocity,

        "wheel_angle": math.degrees(wheel_angle),
        "wheel_velocity": wheel_velocity,

        "contacts": contact_count,
        "normal_force": normal_force,

        "ground_distance": ground_distance,

        "fork_contacts": fork_contact_count
    }


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
print("=" * 70)
print("ROTATION START")
print("=" * 70)

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

        roll, pitch, yaw = (
            p.getEulerFromQuaternion(
                base_orn
            )
        )

        caster = get_caster_data()

        sample = {
            "t": t,

            "base_x": base_pos[0],
            "base_y": base_pos[1],

            "base_roll": math.degrees(roll),
            "base_pitch": math.degrees(pitch),
            "base_yaw": math.degrees(yaw),

            "base_vx": base_lin_vel[0],
            "base_vy": base_lin_vel[1],
            "base_vz": base_lin_vel[2],

            "base_wz": base_ang_vel[2],

            **caster
        }

        samples.append(sample)


# ============================================================
# FINAL ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CASTER TIME HISTORY")
print("=" * 70)

print(
    "Time | Caster X | Caster Y | Caster Z | "
    "Swivel° | Wheel° | Contact | Force N | GroundDist"
)

print("-" * 110)

for s in samples:

    print(
        f"{s['t']:4.2f} | "
        f"{s['x']:8.4f} | "
        f"{s['y']:8.4f} | "
        f"{s['z']:8.4f} | "
        f"{s['swivel_angle']:8.3f} | "
        f"{s['wheel_angle']:8.3f} | "
        f"{s['contacts']:7d} | "
        f"{s['normal_force']:8.3f} | "
        f"{s['ground_distance']:10.5f}"
    )


# ============================================================
# FIND CONTACT TRANSITION
# ============================================================

first_no_contact = None
first_contact = None

for s in samples:

    if s["contacts"] == 0 and first_no_contact is None:
        first_no_contact = s

    if s["contacts"] > 0 and first_contact is None:
        first_contact = s


# ============================================================
# SUMMARY STATISTICS
# ============================================================

caster_z_values = [
    s["z"]
    for s in samples
]

caster_force_values = [
    s["normal_force"]
    for s in samples
]

swivel_values = [
    s["swivel_angle"]
    for s in samples
]

ground_distances = [
    s["ground_distance"]
    for s in samples
    if not math.isnan(s["ground_distance"])
]

max_caster_z = max(caster_z_values)
min_caster_z = min(caster_z_values)

max_force = max(caster_force_values)
min_force = min(caster_force_values)

max_swivel = max(
    abs(value)
    for value in swivel_values
)

max_ground_distance = max(
    ground_distances
) if ground_distances else float("nan")


# ============================================================
# FINAL STATE
# ============================================================

final = samples[-1]

print()
print("=" * 70)
print("CASTER DIAGNOSTIC SUMMARY")
print("=" * 70)

print()
print("CASTER POSITION")
print("-" * 70)
print(
    f"Initial Z            : "
    f"{samples[0]['z']:.6f} m"
)

print(
    f"Final Z              : "
    f"{final['z']:.6f} m"
)

print(
    f"Minimum Z            : "
    f"{min_caster_z:.6f} m"
)

print(
    f"Maximum Z            : "
    f"{max_caster_z:.6f} m"
)

print()
print("CASTER SWIVEL")
print("-" * 70)

print(
    f"Final swivel angle   : "
    f"{final['swivel_angle']:.6f} deg"
)

print(
    f"Maximum |swivel|     : "
    f"{max_swivel:.6f} deg"
)

print(
    f"Final swivel speed   : "
    f"{final['swivel_velocity']:.6f} rad/s"
)

print()
print("CASTER WHEEL")
print("-" * 70)

print(
    f"Final wheel angle    : "
    f"{final['wheel_angle']:.6f} deg"
)

print(
    f"Final wheel speed    : "
    f"{final['wheel_velocity']:.6f} rad/s"
)

print()
print("GROUND CONTACT")
print("-" * 70)

print(
    f"Final contact count  : "
    f"{final['contacts']}"
)

print(
    f"Minimum force        : "
    f"{min_force:.6f} N"
)

print(
    f"Maximum force        : "
    f"{max_force:.6f} N"
)

print(
    f"Final force          : "
    f"{final['normal_force']:.6f} N"
)

print(
    f"Maximum ground dist. : "
    f"{max_ground_distance:.6f} m"
)

print(
    f"Final ground dist.   : "
    f"{final['ground_distance']:.6f} m"
)

print()
print("FORK CONTACT")
print("-" * 70)

print(
    f"Final fork contacts  : "
    f"{final['fork_contacts']}"
)

print()
print("=" * 70)
print("CONTACT TRANSITION")
print("=" * 70)

if first_no_contact is not None:

    print(
        "Caster first lost contact at approximately:"
    )

    print(
        f"  t = {first_no_contact['t']:.3f} s"
    )

    print(
        f"  Z = {first_no_contact['z']:.6f} m"
    )

    print(
        f"  swivel = "
        f"{first_no_contact['swivel_angle']:.3f} deg"
    )

    print(
        f"  ground distance = "
        f"{first_no_contact['ground_distance']:.6f} m"
    )

else:

    print("Caster never lost contact.")


# ============================================================
# SAVE CSV
# ============================================================

with open(
    "v3_17_1_caster_history.csv",
    "w"
) as f:

    f.write(
        "time,caster_x,caster_y,caster_z,"
        "swivel_angle,wheel_angle,contact,"
        "normal_force,ground_distance,"
        "fork_contacts,base_pitch,base_yaw\n"
    )

    for s in samples:

        f.write(
            f"{s['t']:.6f},"
            f"{s['x']:.6f},"
            f"{s['y']:.6f},"
            f"{s['z']:.6f},"
            f"{s['swivel_angle']:.6f},"
            f"{s['wheel_angle']:.6f},"
            f"{s['contacts']},"
            f"{s['normal_force']:.6f},"
            f"{s['ground_distance']:.6f},"
            f"{s['fork_contacts']},"
            f"{s['base_pitch']:.6f},"
            f"{s['base_yaw']:.6f}\n"
        )


print()
print("=" * 70)
print("CSV SAVED")
print("=" * 70)
print(
    "v3_17_1_caster_history.csv"
)

p.disconnect()