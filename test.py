import pybullet as p
import pybullet_data
import time

from v3.robot import create_robot


# ============================================
# CONNECT
# ============================================

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")


# ============================================
# CREATE ROBOT
# ============================================

robot_id = create_robot()


# ============================================
# SETTLE
# ============================================

for _ in range(720):
    p.stepSimulation()
    time.sleep(1 / 240)


# ============================================
# ROBOT GEOMETRY
# ============================================

print()
print("V3.15.x GEOMETRY DIAGNOSTIC")
print("=" * 60)

# Base
pos, orn = p.getBasePositionAndOrientation(robot_id)

print(
    f"BASE: "
    f"x={pos[0]:.6f}, "
    f"y={pos[1]:.6f}, "
    f"z={pos[2]:.6f}"
)

# Robot links
print()
print("ROBOT LINKS")
print("-" * 60)

num_joints = p.getNumJoints(robot_id)

print(f"Number of joints: {num_joints}")

for link in range(num_joints):

    state = p.getLinkState(
        robot_id,
        link,
        computeForwardKinematics=True
    )

    pos = state[0]

    print(
        f"LINK {link}: "
        f"x={pos[0]:.6f}, "
        f"y={pos[1]:.6f}, "
        f"z={pos[2]:.6f}"
    )


# ============================================
# ALL PYBULLET BODIES
# ============================================

print()
print("PYBULLET BODIES")
print("-" * 60)

num_bodies = p.getNumBodies()

print(f"Number of bodies: {num_bodies}")

for body in range(num_bodies):

    body_pos, body_orn = p.getBasePositionAndOrientation(body)

    print(
        f"BODY {body}: "
        f"x={body_pos[0]:.6f}, "
        f"y={body_pos[1]:.6f}, "
        f"z={body_pos[2]:.6f}, "
        f"joints={p.getNumJoints(body)}"
    )


# ============================================
# CONTACTS
# ============================================

print()
print("GROUND CONTACTS")
print("-" * 60)

contacts = p.getContactPoints(
    bodyA=robot_id
)

print(f"Number of contacts: {len(contacts)}")

for contact in contacts:

    link_a = contact[3]
    force = contact[9]

    print(
        f"Link {link_a}: "
        f"normal force={force:.6f} N"
    )


print("=" * 60)

input("\nPress ENTER to close...")

p.disconnect()