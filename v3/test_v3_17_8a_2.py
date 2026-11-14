import math
import pybullet as p
import pybullet_data

from robot import (
    create_robot,
    CASTER_WHEEL_JOINT,
    TIME_STEP,
    GRAVITY,
)


# ============================================================
# V3.17.8A.2
# CONTACT / FRICTION TORQUE DIAGNOSIS
#
# Purpose:
# Determine whether ground contact/friction generates torque
# about the passive caster wheel axle.
#
# Conditions:
#   Gravity ON
#   Ground ON
#   Drive motors OFF
#   Robot yaw = stationary
#   Caster swivel passive
#   Caster wheel passive
#
# NO robot.py changes.
# ============================================================


SIMULATION_TIME = 3.0
SETTLE_TIME = 1.0

CASTER_WHEEL_RADIUS = 0.04

SAMPLE_INTERVAL = 0.10


# ============================================================
# VECTOR HELPERS
# ============================================================

def dot(a, b):
    return (
        a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
    )


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vector_add(a, b):
    return [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    ]


def vector_scale(a, s):
    return [
        a[0] * s,
        a[1] * s,
        a[2] * s,
    ]


def magnitude(v):
    return math.sqrt(
        dot(v, v)
    )


def normalize(v):

    mag = magnitude(v)

    if mag < 1e-12:
        return [0.0, 0.0, 0.0]

    return [
        v[0] / mag,
        v[1] / mag,
        v[2] / mag,
    ]


# ============================================================
# ROTATE LOCAL VECTOR INTO WORLD FRAME
# ============================================================

def local_vector_to_world(
    quaternion,
    vector,
):
    """
    Rotate a local vector using a quaternion.
    """

    matrix = p.getMatrixFromQuaternion(
        quaternion
    )

    # PyBullet matrix is row-major.
    #
    # world_vector = R * local_vector

    x = (
        matrix[0] * vector[0]
        + matrix[1] * vector[1]
        + matrix[2] * vector[2]
    )

    y = (
        matrix[3] * vector[0]
        + matrix[4] * vector[1]
        + matrix[5] * vector[2]
    )

    z = (
        matrix[6] * vector[0]
        + matrix[7] * vector[1]
        + matrix[8] * vector[2]
    )

    return [x, y, z]


# ============================================================
# CONNECT
# ============================================================

physics_client = p.connect(
    p.DIRECT
)

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
# DISABLE DRIVE MOTORS
# ============================================================

p.setJointMotorControl2(
    robot_id,
    0,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)

p.setJointMotorControl2(
    robot_id,
    1,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# PASSIVE CASTER SWIVEL
# ============================================================

p.setJointMotorControl2(
    robot_id,
    2,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# PASSIVE CASTER WHEEL
# ============================================================

p.setJointMotorControl2(
    robot_id,
    CASTER_WHEEL_JOINT,
    p.VELOCITY_CONTROL,
    targetVelocity=0,
    force=0,
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("V3.17.8A.2 — CONTACT / FRICTION TORQUE DIAGNOSIS")
print("=" * 80)

print()

print("Gravity       : ON")
print("Ground        : ON")
print("Drive motors  : OFF")
print("Caster swivel : PASSIVE")
print("Caster wheel  : PASSIVE")
print("Robot yaw     : STATIONARY")

print()

print("Objective:")
print(
    "Measure caster-ground contact forces and calculate "
    "their torque about the caster wheel axle."
)

print()

print("=" * 80)


# ============================================================
# SETTLE
# ============================================================

print()
print("Settling robot...")

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):
    p.stepSimulation()

print("Settlement complete.")


# ============================================================
# STORAGE
# ============================================================

history = []

normal_forces = []
friction_forces = []
total_forces = []

normal_torques = []
friction_torques = []
total_torques = []

axle_torques = []
normal_axle_torques = []
friction_axle_torques = []

wheel_velocities = []
wheel_angles = []

contact_count = 0
total_samples = 0


# ============================================================
# SIMULATION
# ============================================================

simulation_steps = int(
    SIMULATION_TIME / TIME_STEP
)

sample_every = max(
    1,
    int(
        SAMPLE_INTERVAL
        / TIME_STEP
    )
)


for step in range(
    simulation_steps
):

    p.stepSimulation()

    if step % sample_every != 0:
        continue


    # ========================================================
    # CASTER WHEEL LINK STATE
    # ========================================================

    wheel_state = p.getLinkState(
        robot_id,
        CASTER_WHEEL_JOINT,
        computeForwardKinematics=True,
    )

    wheel_position = wheel_state[4]
    wheel_quaternion = wheel_state[5]


    # ========================================================
    # CASTER WHEEL AXLE
    # ========================================================
    #
    # Caster wheel joint axis is local Y.
    #
    # Convert local Y axis into world coordinates.
    # ========================================================

    axle_world = local_vector_to_world(
        wheel_quaternion,
        [0.0, 1.0, 0.0],
    )

    axle_world = normalize(
        axle_world
    )


    # ========================================================
    # CONTACT POINTS
    # ========================================================

    contacts = p.getContactPoints(
        bodyA=robot_id,
        bodyB=ground_id,
        linkIndexA=CASTER_WHEEL_JOINT,
    )


    total_normal_force = 0.0

    total_friction_force_vector = [
        0.0,
        0.0,
        0.0,
    ]

    total_force_vector = [
        0.0,
        0.0,
        0.0,
    ]


    normal_torque_vector = [
        0.0,
        0.0,
        0.0,
    ]

    friction_torque_vector = [
        0.0,
        0.0,
        0.0,
    ]

    total_torque_vector = [
        0.0,
        0.0,
        0.0,
    ]


    # ========================================================
    # CONTACT FORCE PROCESSING
    # ========================================================

    for contact in contacts:

        # ----------------------------------------------------
        # PyBullet contact fields
        # ----------------------------------------------------

        contact_position_A = contact[5]

        contact_position_B = contact[6]

        normal_on_B = contact[7]

        normal_force = contact[9]

        friction_dir_1 = contact[11]

        friction_force_1 = contact[10]

        friction_dir_2 = contact[13]

        friction_force_2 = contact[12]


        # ----------------------------------------------------
        # Normal force
        # ----------------------------------------------------

        normal_force_vector = vector_scale(
            normal_on_B,
            normal_force,
        )


        # ----------------------------------------------------
        # Friction forces
        # ----------------------------------------------------

        friction_vector_1 = vector_scale(
            friction_dir_1,
            friction_force_1,
        )

        friction_vector_2 = vector_scale(
            friction_dir_2,
            friction_force_2,
        )

        friction_vector = vector_add(
            friction_vector_1,
            friction_vector_2,
        )


        # ----------------------------------------------------
        # Total contact force
        # ----------------------------------------------------

        contact_force = vector_add(
            normal_force_vector,
            friction_vector,
        )


        # ----------------------------------------------------
        # Moment arm
        #
        # From caster wheel center to contact point.
        # ----------------------------------------------------

        r = [
            contact_position_A[0]
            - wheel_position[0],

            contact_position_A[1]
            - wheel_position[1],

            contact_position_A[2]
            - wheel_position[2],
        ]


        # ----------------------------------------------------
        # Torque from NORMAL force
        # ----------------------------------------------------

        normal_torque = cross(
            r,
            normal_force_vector,
        )


        # ----------------------------------------------------
        # Torque from FRICTION
        # ----------------------------------------------------

        friction_torque = cross(
            r,
            friction_vector,
        )


        # ----------------------------------------------------
        # Total torque
        # ----------------------------------------------------

        total_torque = cross(
            r,
            contact_force,
        )


        # ----------------------------------------------------
        # Accumulate forces
        # ----------------------------------------------------

        total_normal_force += (
            normal_force
        )

        total_friction_force_vector = (
            vector_add(
                total_friction_force_vector,
                friction_vector,
            )
        )

        total_force_vector = (
            vector_add(
                total_force_vector,
                contact_force,
            )
        )


        # ----------------------------------------------------
        # Accumulate torques
        # ----------------------------------------------------

        normal_torque_vector = (
            vector_add(
                normal_torque_vector,
                normal_torque,
            )
        )

        friction_torque_vector = (
            vector_add(
                friction_torque_vector,
                friction_torque,
            )
        )

        total_torque_vector = (
            vector_add(
                total_torque_vector,
                total_torque,
            )
        )


    # ========================================================
    # AXLE TORQUE PROJECTION
    # ========================================================

    normal_axle_torque = dot(
        normal_torque_vector,
        axle_world,
    )

    friction_axle_torque = dot(
        friction_torque_vector,
        axle_world,
    )

    total_axle_torque = dot(
        total_torque_vector,
        axle_world,
    )


    # ========================================================
    # FORCE MAGNITUDES
    # ========================================================

    friction_force_magnitude = magnitude(
        total_friction_force_vector
    )

    total_force_magnitude = magnitude(
        total_force_vector
    )


    # ========================================================
    # TORQUE MAGNITUDES
    # ========================================================

    normal_torque_magnitude = magnitude(
        normal_torque_vector
    )

    friction_torque_magnitude = magnitude(
        friction_torque_vector
    )

    total_torque_magnitude = magnitude(
        total_torque_vector
    )


    # ========================================================
    # WHEEL JOINT STATE
    # ========================================================

    wheel_joint_state = p.getJointState(
        robot_id,
        CASTER_WHEEL_JOINT,
    )

    wheel_angle = math.degrees(
        wheel_joint_state[0]
    )

    wheel_velocity = (
        wheel_joint_state[1]
    )


    # ========================================================
    # STORAGE
    # ========================================================

    normal_forces.append(
        total_normal_force
    )

    friction_forces.append(
        friction_force_magnitude
    )

    total_forces.append(
        total_force_magnitude
    )

    normal_torques.append(
        normal_torque_magnitude
    )

    friction_torques.append(
        friction_torque_magnitude
    )

    total_torques.append(
        total_torque_magnitude
    )

    axle_torques.append(
        total_axle_torque
    )

    normal_axle_torques.append(
        normal_axle_torque
    )

    friction_axle_torques.append(
        friction_axle_torque
    )

    wheel_angles.append(
        wheel_angle
    )

    wheel_velocities.append(
        wheel_velocity
    )


    if len(contacts) > 0:
        contact_count += 1

    total_samples += 1


    # ========================================================
    # HISTORY
    # ========================================================

    history.append(
        (
            step * TIME_STEP,
            len(contacts),
            total_normal_force,
            friction_force_magnitude,
            total_force_magnitude,
            normal_torque_magnitude,
            friction_torque_magnitude,
            total_torque_magnitude,
            normal_axle_torque,
            friction_axle_torque,
            total_axle_torque,
            wheel_angle,
            wheel_velocity,
            axle_world,
        )
    )


# ============================================================
# RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("V3.17.8A.2 RESULTS")
print("=" * 80)

print()

print(
    f"Contact samples          : "
    f"{contact_count}/{total_samples}"
)

print(
    f"Contact percentage       : "
    f"{100.0 * contact_count / total_samples:.2f}%"
)

print()

print(
    f"Maximum normal force     : "
    f"{max(normal_forces):.6f} N"
)

print(
    f"Average normal force     : "
    f"{sum(normal_forces) / len(normal_forces):.6f} N"
)

print()

print(
    f"Maximum friction force   : "
    f"{max(friction_forces):.6f} N"
)

print(
    f"Average friction force   : "
    f"{sum(friction_forces) / len(friction_forces):.6f} N"
)

print()

print(
    f"Maximum total force      : "
    f"{max(total_forces):.6f} N"
)

print(
    f"Average total force      : "
    f"{sum(total_forces) / len(total_forces):.6f} N"
)

print()

print(
    f"Maximum normal torque    : "
    f"{max(normal_torques):.9f} N*m"
)

print(
    f"Maximum friction torque  : "
    f"{max(friction_torques):.9f} N*m"
)

print(
    f"Maximum total torque     : "
    f"{max(total_torques):.9f} N*m"
)

print()

print(
    f"Maximum normal axle "
    f"torque                  : "
    f"{max(abs(x) for x in normal_axle_torques):.9f} N*m"
)

print(
    f"Maximum friction axle "
    f"torque                  : "
    f"{max(abs(x) for x in friction_axle_torques):.9f} N*m"
)

print(
    f"Maximum total axle "
    f"torque                  : "
    f"{max(abs(x) for x in axle_torques):.9f} N*m"
)

print()

print(
    f"Average normal axle "
    f"torque                  : "
    f"{sum(normal_axle_torques) / len(normal_axle_torques):.9f} N*m"
)

print(
    f"Average friction axle "
    f"torque                  : "
    f"{sum(friction_axle_torques) / len(friction_axle_torques):.9f} N*m"
)

print(
    f"Average total axle "
    f"torque                  : "
    f"{sum(axle_torques) / len(axle_torques):.9f} N*m"
)

print()

print(
    f"Initial wheel angle      : "
    f"{wheel_angles[0]:.6f} deg"
)

print(
    f"Final wheel angle        : "
    f"{wheel_angles[-1]:.6f} deg"
)

print(
    f"Wheel angle drift        : "
    f"{wheel_angles[-1] - wheel_angles[0]:.6f} deg"
)

print(
    f"Maximum wheel velocity   : "
    f"{max(abs(v) for v in wheel_velocities):.6f} rad/s"
)


# ============================================================
# TIME HISTORY
# ============================================================

print()
print("=" * 80)
print("TIME HISTORY")
print("=" * 80)

print()

print(
    " Time | Contacts | NormalF | FrictionF | "
    "NormalAxle | FrictionAxle | TotalAxle | WheelAngle"
)

print("-" * 110)


for record in history:

    (
        time,
        number_contacts,
        normal_force,
        friction_force,
        total_force,
        normal_torque,
        friction_torque,
        total_torque,
        normal_axle,
        friction_axle,
        total_axle,
        wheel_angle,
        wheel_velocity,
        axle_world,
    ) = record


    print(
        f"{time:5.2f} | "
        f"{number_contacts:8d} | "
        f"{normal_force:8.3f} | "
        f"{friction_force:9.3f} | "
        f"{normal_axle:10.6f} | "
        f"{friction_axle:12.6f} | "
        f"{total_axle:9.6f} | "
        f"{wheel_angle:10.3f}°"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

max_friction_axle_torque = max(
    abs(x)
    for x in friction_axle_torques
)

max_normal_axle_torque = max(
    abs(x)
    for x in normal_axle_torques
)

max_total_axle_torque = max(
    abs(x)
    for x in axle_torques
)


print()
print("=" * 80)
print("DIAGNOSTIC")
print("=" * 80)

print()

if max_total_axle_torque < 1e-4:

    print(
        "PASS — measured contact torque about the "
        "caster axle is very small."
    )

else:

    print(
        "CONTACT TORQUE DETECTED — ground contact "
        "produces measurable torque about the caster axle."
    )


print()

if max_friction_axle_torque > (
    10.0 * max_normal_axle_torque
):

    print(
        "FRICTION-DOMINATED — axle torque is primarily "
        "associated with friction forces."
    )

elif max_normal_axle_torque > (
    10.0 * max_friction_axle_torque
):

    print(
        "NORMAL-FORCE-DOMINATED — axle torque is primarily "
        "associated with the normal contact force."
    )

else:

    print(
        "MIXED — both normal and friction components "
        "contribute to axle torque."
    )


print()

print(
    "IMPORTANT:"
)

print(
    "A non-zero contact torque does not by itself prove "
    "that the caster model is incorrect."
)

print(
    "The torque must be compared with the observed "
    "caster-wheel acceleration and joint behavior."
)

print()

print("=" * 80)
print("END V3.17.8A.2")
print("=" * 80)


# ============================================================
# DISCONNECT
# ============================================================

p.disconnect()