import pybullet as p
import pybullet_data
import time
import math


# ============================================================
# V3 CONTACT-GEOMETRY + JOINT-STATE DIAGNOSTIC
#
# Purpose:
#   1. Verify robot link hierarchy
#   2. Verify wheel/caster geometry
#   3. Verify ground contacts
#   4. Verify normal-force balance
#   5. Inspect joint states after gravity settling
#
# Robot is NOT driven.
# Only gravity settling is performed.
#
# No Gymnasium
# No PPO
# ============================================================


# ============================================================
# IMPORT ROBOT
# ============================================================

from robot import create_robot


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

TIME_STEP = 1.0 / 240.0
SETTLE_TIME = 2.0

GRAVITY = -9.81


# ============================================================
# JOINT NAMES
# ============================================================

JOINT_NAMES = {
    0: "LEFT WHEEL",
    1: "RIGHT WHEEL",
    2: "CASTER SWIVEL",
    3: "CASTER WHEEL"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_to_euler_degrees(quaternion):

    roll, pitch, yaw = p.getEulerFromQuaternion(
        quaternion
    )

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


def print_separator():

    print("-" * 70)


# ============================================================
# START SIMULATION
# ============================================================

print("CONTACT DIAGNOSTIC STARTED")

physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(
    0.0,
    0.0,
    GRAVITY
)

p.setTimeStep(
    TIME_STEP
)

p.setPhysicsEngineParameter(
    fixedTimeStep=TIME_STEP,
    numSolverIterations=100
)


# ============================================================
# GROUND
# ============================================================

ground_collision = p.createCollisionShape(
    p.GEOM_PLANE
)

ground_id = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=ground_collision
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("V3 CONTACT-GEOMETRY + JOINT-STATE DIAGNOSTIC")
print("=" * 70)

print()
print("The robot will NOT be driven.")
print("Only gravity settling will be performed.")


# ============================================================
# EXPECTED GEOMETRY
# ============================================================

print()
print("=" * 70)
print("EXPECTED GEOMETRY")
print("=" * 70)

print()
print("Drive wheel axle positions:")
print("  Left wheel  : X=0.000, Y=0.150")
print("  Right wheel : X=0.000, Y=-0.150")

print()
print("Caster geometry:")
print("  Fork X      : -0.150 m")
print("  Wheel X     : -0.200 m")
print("  Wheel radius: 0.040 m")

print()
print("Expected ground plane: Z = 0.000 m")


# ============================================================
# CREATE ROBOT
# ============================================================

robot_id = create_robot()

print()
print("Robot created.")
print("Robot ID:", robot_id)
print("Joint count:", p.getNumJoints(robot_id))


# ============================================================
# INITIAL ROBOT STATE
# ============================================================

initial_position, initial_orientation = (
    p.getBasePositionAndOrientation(robot_id)
)

initial_roll, initial_pitch, initial_yaw = (
    quaternion_to_euler_degrees(
        initial_orientation
    )
)


# ============================================================
# GRAVITY SETTLING
# ============================================================

print()
print("Settling for %.2f seconds..." % SETTLE_TIME)

settle_steps = int(
    SETTLE_TIME / TIME_STEP
)

for _ in range(settle_steps):

    p.stepSimulation()


# ============================================================
# BASE STATE
# ============================================================

base_position, base_orientation = (
    p.getBasePositionAndOrientation(robot_id)
)

base_roll, base_pitch, base_yaw = (
    quaternion_to_euler_degrees(
        base_orientation
    )
)


print()
print("=" * 70)
print("BASE STATE")
print("=" * 70)

print("Position:")
print("  X = %.9f m" % base_position[0])
print("  Y = %.9f m" % base_position[1])
print("  Z = %.9f m" % base_position[2])

print("Orientation:")
print("  Roll  = %.6f deg" % base_roll)
print("  Pitch = %.6f deg" % base_pitch)
print("  Yaw   = %.6f deg" % base_yaw)


# ============================================================
# JOINT / LINK GEOMETRY
# ============================================================

print()
print("=" * 70)
print("JOINT / LINK GEOMETRY")
print("=" * 70)

joint_count = p.getNumJoints(robot_id)

print("Joint count:", joint_count)


for joint_index in range(joint_count):

    joint_info = p.getJointInfo(
        robot_id,
        joint_index
    )

    joint_type = joint_info[2]

    joint_axis = joint_info[13]

    parent_index = joint_info[16]

    parent_frame_position = joint_info[14]

    parent_frame_orientation = joint_info[15]

    joint_name = JOINT_NAMES.get(
        joint_index,
        "UNKNOWN"
    )

    link_state = p.getLinkState(
        robot_id,
        joint_index,
        computeForwardKinematics=True
    )

    world_position = link_state[4]

    world_orientation = link_state[5]

    roll, pitch, yaw = (
        quaternion_to_euler_degrees(
            world_orientation
        )
    )

    print()
    print(
        "JOINT %d — %s"
        % (
            joint_index,
            joint_name
        )
    )

    print("  Type:", joint_type)

    print(
        "  Parent index:",
        parent_index
    )

    print(
        "  Joint axis: (%.1f, %.1f, %.1f)"
        % (
            joint_axis[0],
            joint_axis[1],
            joint_axis[2]
        )
    )

    print(
        "  Parent frame position: "
        "(%.6f, %.6f, %.6f)"
        % (
            parent_frame_position[0],
            parent_frame_position[1],
            parent_frame_position[2]
        )
    )

    print(
        "  Parent frame orientation: "
        "(%.6f, %.6f, %.6f, %.6f)"
        % (
            parent_frame_orientation[0],
            parent_frame_orientation[1],
            parent_frame_orientation[2],
            parent_frame_orientation[3]
        )
    )

    print("  World position:")

    print(
        "    X = %.9f"
        % world_position[0]
    )

    print(
        "    Y = %.9f"
        % world_position[1]
    )

    print(
        "    Z = %.9f"
        % world_position[2]
    )

    print("  World orientation:")

    print(
        "    Roll  = %.6f deg"
        % roll
    )

    print(
        "    Pitch = %.6f deg"
        % pitch
    )

    print(
        "    Yaw   = %.6f deg"
        % yaw
    )


# ============================================================
# JOINT STATE DIAGNOSTIC
# ============================================================

print()
print("=" * 70)
print("JOINT STATE DIAGNOSTIC")
print("=" * 70)

print()
print("These values show the actual joint motion after")
print("the robot has settled under gravity.")

for joint_index in range(joint_count):

    joint_state = p.getJointState(
        robot_id,
        joint_index
    )

    joint_position = joint_state[0]
    joint_velocity = joint_state[1]
    reaction_forces = joint_state[2]
    applied_torque = joint_state[3]

    joint_name = JOINT_NAMES.get(
        joint_index,
        "UNKNOWN"
    )

    print()
    print(
        "JOINT %d — %s"
        % (
            joint_index,
            joint_name
        )
    )

    print(
        "  Position = %.9f rad"
        % joint_position
    )

    print(
        "  Position = %.6f deg"
        % math.degrees(joint_position)
    )

    print(
        "  Velocity = %.9f rad/s"
        % joint_velocity
    )

    print(
        "  Applied torque = %.9f N*m"
        % applied_torque
    )

    print(
        "  Reaction force/torque = %s"
        % str(reaction_forces)
    )


# ============================================================
# BASE LINEAR / ANGULAR VELOCITY
# ============================================================

base_velocity_linear, base_velocity_angular = (
    p.getBaseVelocity(robot_id)
)

print()
print("=" * 70)
print("BASE VELOCITY AFTER SETTLING")
print("=" * 70)

print(
    "Linear velocity:"
    " X = %.9f"
    " Y = %.9f"
    " Z = %.9f"
    % (
        base_velocity_linear[0],
        base_velocity_linear[1],
        base_velocity_linear[2]
    )
)

print(
    "Angular velocity:"
    " X = %.9f"
    " Y = %.9f"
    " Z = %.9f"
    % (
        base_velocity_angular[0],
        base_velocity_angular[1],
        base_velocity_angular[2]
    )
)


# ============================================================
# GROUND CONTACT GEOMETRY
# ============================================================

print()
print("=" * 70)
print("GROUND CONTACT GEOMETRY")
print("=" * 70)

contact_points = p.getContactPoints(
    bodyA=robot_id,
    bodyB=ground_id
)

print(
    "Total contact points:",
    len(contact_points)
)


# ============================================================
# CONTACT FORCE STORAGE
# ============================================================

contact_force = {
    -1: 0.0,
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0
}

contact_count = {
    -1: 0,
    0: 0,
    1: 0,
    2: 0,
    3: 0
}


# ============================================================
# CONTACT NAME
# ============================================================

def get_part_name(link_index):

    if link_index == -1:
        return "CHASSIS"

    if link_index == 0:
        return "LEFT WHEEL"

    if link_index == 1:
        return "RIGHT WHEEL"

    if link_index == 2:
        return "CASTER FORK"

    if link_index == 3:
        return "CASTER WHEEL"

    return "UNKNOWN"


# ============================================================
# PRINT CONTACTS
# ============================================================

for contact_number, contact in enumerate(
    contact_points,
    start=1
):

    link_index = contact[3]

    robot_contact_position = contact[5]

    ground_contact_position = contact[6]

    contact_normal = contact[7]

    contact_distance = contact[8]

    normal_force = contact[9]

    friction_1 = contact[10]

    friction_direction_1 = contact[11]

    friction_2 = contact[12]

    friction_direction_2 = contact[13]

    part_name = get_part_name(
        link_index
    )

    contact_count[link_index] += 1

    contact_force[link_index] += (
        normal_force
    )

    print()
    print("-" * 70)

    print(
        "CONTACT %d"
        % contact_number
    )

    print(
        "Body part:",
        part_name
    )

    print(
        "Link index:",
        link_index
    )

    print("Robot contact point:")

    print(
        "  X = %.9f"
        % robot_contact_position[0]
    )

    print(
        "  Y = %.9f"
        % robot_contact_position[1]
    )

    print(
        "  Z = %.9f"
        % robot_contact_position[2]
    )

    print("Ground contact point:")

    print(
        "  X = %.9f"
        % ground_contact_position[0]
    )

    print(
        "  Y = %.9f"
        % ground_contact_position[1]
    )

    print(
        "  Z = %.9f"
        % ground_contact_position[2]
    )

    print(
        "Contact normal: "
        "(%s, %s, %s)"
        % (
            contact_normal[0],
            contact_normal[1],
            contact_normal[2]
        )
    )

    print(
        "Contact distance: %.9f m"
        % contact_distance
    )

    print(
        "Normal force: %.9f N"
        % normal_force
    )

    print(
        "Friction 1: %.9f N"
        % friction_1
    )

    print(
        "Friction direction 1: %s"
        % str(friction_direction_1)
    )

    print(
        "Friction 2: %.9f N"
        % friction_2
    )

    print(
        "Friction direction 2: %s"
        % str(friction_direction_2)
    )


# ============================================================
# CONTACT SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONTACT SUMMARY")
print("=" * 70)

print(
    "LEFT WHEEL      contacts=%d   force=%.6f N"
    % (
        contact_count[0],
        contact_force[0]
    )
)

print(
    "RIGHT WHEEL     contacts=%d   force=%.6f N"
    % (
        contact_count[1],
        contact_force[1]
    )
)

print(
    "CASTER FORK     contacts=%d   force=%.6f N"
    % (
        contact_count[2],
        contact_force[2]
    )
)

print(
    "CASTER WHEEL    contacts=%d   force=%.6f N"
    % (
        contact_count[3],
        contact_force[3]
    )
)

print(
    "CHASSIS         contacts=%d   force=%.6f N"
    % (
        contact_count[-1],
        contact_force[-1]
    )
)


# ============================================================
# TOTAL NORMAL FORCE
# ============================================================

total_normal_force = sum(
    contact_force.values()
)

total_mass = (
    5.0
    + 0.5
    + 0.5
    + 0.05
    + 0.15
)

expected_weight = (
    total_mass
    * abs(GRAVITY)
)

force_error = (
    total_normal_force
    - expected_weight
)


print()
print(
    "TOTAL NORMAL FORCE = %.6f N"
    % total_normal_force
)

print(
    "EXPECTED WEIGHT    = %.6f N"
    % expected_weight
)

print(
    "FORCE ERROR         = %.6f N"
    % force_error
)


# ============================================================
# STATIC VALIDATION
# ============================================================

print()
print("=" * 70)
print("STATIC GEOMETRY VALIDATION")
print("=" * 70)


# Roll criterion
roll_pass = abs(base_roll) < 2.0

# Pitch criterion
pitch_pass = abs(base_pitch) < 2.0

# Yaw criterion
yaw_pass = abs(base_yaw) < 2.0

# Contact criteria
left_contact_pass = (
    contact_count[0] > 0
)

right_contact_pass = (
    contact_count[1] > 0
)

caster_wheel_contact_pass = (
    contact_count[3] > 0
)

caster_fork_clear_pass = (
    contact_count[2] == 0
)

chassis_clear_pass = (
    contact_count[-1] == 0
)

# Force balance
force_balance_pass = (
    abs(force_error) < 0.1
)


def print_pass_fail(
    name,
    condition
):

    result = "PASS" if condition else "FAIL"

    print(
        "%-30s %s"
        % (
            name,
            result
        )
    )


print_pass_fail(
    "Roll",
    roll_pass
)

print_pass_fail(
    "Pitch",
    pitch_pass
)

print_pass_fail(
    "Yaw",
    yaw_pass
)

print_pass_fail(
    "Left wheel contact",
    left_contact_pass
)

print_pass_fail(
    "Right wheel contact",
    right_contact_pass
)

print_pass_fail(
    "Caster wheel contact",
    caster_wheel_contact_pass
)

print_pass_fail(
    "Caster fork clear",
    caster_fork_clear_pass
)

print_pass_fail(
    "Chassis clear",
    chassis_clear_pass
)

print_pass_fail(
    "Weight balance",
    force_balance_pass
)


# ============================================================
# JOINT STATE VALIDATION
# ============================================================

print()
print("=" * 70)
print("JOINT STATE VALIDATION")
print("=" * 70)

for joint_index in range(joint_count):

    joint_state = p.getJointState(
        robot_id,
        joint_index
    )

    joint_position = joint_state[0]

    joint_velocity = joint_state[1]

    joint_name = JOINT_NAMES.get(
        joint_index,
        "UNKNOWN"
    )

    print()
    print(
        "%s:"
        % joint_name
    )

    print(
        "  Position = %.9f rad"
        % joint_position
    )

    print(
        "  Position = %.6f deg"
        % math.degrees(joint_position)
    )

    print(
        "  Velocity = %.9f rad/s"
        % joint_velocity
    )


# ============================================================
# FINAL RESULT
# ============================================================

overall_pass = (
    roll_pass
    and pitch_pass
    and yaw_pass
    and left_contact_pass
    and right_contact_pass
    and caster_wheel_contact_pass
    and caster_fork_clear_pass
    and chassis_clear_pass
    and force_balance_pass
)


print()
print("=" * 70)

if overall_pass:
    print("OVERALL STATIC GEOMETRY: PASS")
else:
    print("OVERALL STATIC GEOMETRY: FAIL")

print("=" * 70)


# ============================================================
# END
# ============================================================

p.disconnect()

print()
print("DIAGNOSTIC COMPLETE")