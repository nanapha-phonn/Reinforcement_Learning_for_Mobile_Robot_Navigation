import pybullet as p
import pybullet_data


# ============================================================
# CONNECT
# ============================================================

p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(0, 0, -9.81)


# ============================================================
# SIMPLE BOX SHAPE
# ============================================================

box = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.05]
)


# ============================================================
# FOUR LINKS
#
# Link 0 = LEFT
# Link 1 = RIGHT
# Link 2 = CASTER FORK
# Link 3 = CASTER WHEEL
# ============================================================

linkMasses = [
    1.0,
    1.0,
    1.0,
    1.0
]


linkCollisionShapeIndices = [
    box,
    box,
    box,
    box
]


linkVisualShapeIndices = [
    -1,
    -1,
    -1,
    -1
]


# Each link has a unique obvious position.
#
# LEFT       = ( 0.0, +1.0, 1.0)
# RIGHT      = ( 0.0, -1.0, 1.0)
# FORK       = (-2.0,  0.0, 1.0)
# WHEEL      = (-3.0,  0.0, 1.0)

linkPositions = [
    [0.0,  1.0, 1.0],
    [0.0, -1.0, 1.0],
    [-2.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0]
]


linkOrientations = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
]


linkInertialFramePositions = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]


linkInertialFrameOrientations = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
]


# ============================================================
# IMPORTANT TEST
# ============================================================

linkParentIndices = [
    0,
    0,
    0,
    3
]


linkJointTypes = [
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE
]


linkJointAxis = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0]
]


# ============================================================
# PRINT INPUT
# ============================================================

print()
print("=" * 70)
print("MULTIBODY HIERARCHY ISOLATION TEST")
print("=" * 70)

print()
print("Parent indices:")
print(linkParentIndices)

print()
print("Expected conceptual hierarchy:")
print("  Link 0 -> BASE")
print("  Link 1 -> BASE")
print("  Link 2 -> BASE")
print("  Link 3 -> LINK 2")


# ============================================================
# CREATE
# ============================================================

robot_id = p.createMultiBody(
    baseMass=5.0,
    baseCollisionShapeIndex=box,
    baseVisualShapeIndex=-1,

    basePosition=[0, 0, 2],
    baseOrientation=[0, 0, 0, 1],

    linkMasses=linkMasses,
    linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,

    linkPositions=linkPositions,
    linkOrientations=linkOrientations,

    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrientations,

    linkParentIndices=linkParentIndices,

    linkJointTypes=linkJointTypes,

    linkJointAxis=linkJointAxis
)


# ============================================================
# JOINT COUNT
# ============================================================

print()
print("Robot ID:", robot_id)
print("Joint count:", p.getNumJoints(robot_id))


# ============================================================
# PYBULLET'S ACTUAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("ACTUAL PYBULLET JOINT STRUCTURE")
print("=" * 70)


for i in range(p.getNumJoints(robot_id)):

    info = p.getJointInfo(
        robot_id,
        i
    )

    joint_type = info[2]

    joint_axis = info[13]

    parent_index = info[16]

    parent_frame = info[14]

    state = p.getLinkState(
        robot_id,
        i,
        computeForwardKinematics=True
    )

    world_position = state[4]

    print()
    print("JOINT", i)

    print(
        "  Parent index:",
        parent_index
    )

    print(
        "  Axis:",
        joint_axis
    )

    print(
        "  Parent frame:",
        parent_frame
    )

    print(
        "  World position:",
        world_position
    )


# ============================================================
# SETTLE
# ============================================================

for _ in range(240):

    p.stepSimulation()


# ============================================================
# FINAL LINK POSITIONS
# ============================================================

print()
print("=" * 70)
print("FINAL LINK POSITIONS")
print("=" * 70)


for i in range(p.getNumJoints(robot_id)):

    state = p.getLinkState(
        robot_id,
        i,
        computeForwardKinematics=True
    )

    position = state[4]

    print(
        "Link %d -> (%.6f, %.6f, %.6f)"
        % (
            i,
            position[0],
            position[1],
            position[2]
        )
    )


# ============================================================
# END
# ============================================================

p.disconnect()

print()
print("TEST COMPLETE")