"""
V3.22.5i — Final Production Caster Collision-Shape Audit

Purpose:
    Inspect the ACTUAL collision shape of every robot link.

Important:
    PyBullet may return only the base shape when using:
        getCollisionShapeData(body, -1)

Therefore this test explicitly queries:
    base  = -1
    link0 = 0
    link1 = 1
    link2 = 2
    link3 = 3

Expected caster:
    Link 3
    GEOM_CYLINDER
    radius = 0.040 m
    height = 0.025 m

This test:
    - does NOT modify robot.py
    - does NOT train PPO
"""

from pathlib import Path
import ast
import importlib.util

import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_FILE = PROJECT_ROOT / "robot.py"

EXPECTED_RADIUS = 0.040
EXPECTED_WIDTH = 0.025


def read_constants():

    source = ROBOT_FILE.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(ROBOT_FILE)
    )

    values = {}

    wanted = {
        "CASTER_WHEEL_RADIUS",
        "CASTER_WHEEL_WIDTH",
    }

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id in wanted
                    and isinstance(node.value, ast.Constant)
                ):

                    values[target.id] = node.value.value

        elif isinstance(node, ast.AnnAssign):

            target = node.target

            if (
                isinstance(target, ast.Name)
                and target.id in wanted
                and node.value is not None
                and isinstance(node.value, ast.Constant)
            ):

                values[target.id] = node.value.value

    return values


def load_robot():

    spec = importlib.util.spec_from_file_location(
        "robot_v3225i",
        ROBOT_FILE
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load robot.py"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def inspect_link(body_id, link_index):

    data = p.getCollisionShapeData(
        body_id,
        link_index
    )

    if not data:

        return None

    return data[0]


def main():

    print("=" * 80)
    print("V3.22.5i — FINAL PRODUCTION CASTER SHAPE AUDIT")
    print("=" * 80)

    # ================================================================
    # 1. FILE
    # ================================================================

    print()
    print("Production file:")
    print(f"  {ROBOT_FILE}")

    if not ROBOT_FILE.exists():

        print()
        print("[FAIL] robot.py not found.")
        return

    # ================================================================
    # 2. SOURCE CONSTANTS
    # ================================================================

    print()
    print("-" * 80)
    print("1. SOURCE CONSTANT AUDIT")
    print("-" * 80)

    constants = read_constants()

    source_radius = constants.get(
        "CASTER_WHEEL_RADIUS"
    )

    source_width = constants.get(
        "CASTER_WHEEL_WIDTH"
    )

    print(
        f"CASTER_WHEEL_RADIUS = {source_radius}"
    )

    print(
        f"CASTER_WHEEL_WIDTH  = {source_width}"
    )

    source_pass = (
        source_radius == EXPECTED_RADIUS
        and
        source_width == EXPECTED_WIDTH
    )

    print(
        "[PASS] Source constants are correct."
        if source_pass
        else
        "[FAIL] Source constants are incorrect."
    )

    # ================================================================
    # 3. LOAD ROBOT
    # ================================================================

    print()
    print("-" * 80)
    print("2. PRODUCTION MODULE")
    print("-" * 80)

    try:

        robot = load_robot()

        print(
            "[PASS] robot.py imported successfully."
        )

    except Exception as error:

        print(
            "[FAIL] robot.py import failed."
        )

        print(
            f"Error: {error}"
        )

        return

    # ================================================================
    # 4. RUNTIME CONSTANTS
    # ================================================================

    print()
    print("-" * 80)
    print("3. RUNTIME CONSTANT AUDIT")
    print("-" * 80)

    runtime_radius = getattr(
        robot,
        "CASTER_WHEEL_RADIUS",
        None
    )

    runtime_width = getattr(
        robot,
        "CASTER_WHEEL_WIDTH",
        None
    )

    print(
        f"Runtime CASTER_WHEEL_RADIUS = "
        f"{runtime_radius}"
    )

    print(
        f"Runtime CASTER_WHEEL_WIDTH  = "
        f"{runtime_width}"
    )

    runtime_pass = (
        runtime_radius == EXPECTED_RADIUS
        and
        runtime_width == EXPECTED_WIDTH
    )

    print(
        "[PASS] Runtime constants are correct."
        if runtime_pass
        else
        "[FAIL] Runtime constants are incorrect."
    )

    # ================================================================
    # 5. PYBULLET
    # ================================================================

    print()
    print("-" * 80)
    print("4. CREATE PRODUCTION ROBOT")
    print("-" * 80)

    client = p.connect(
        p.DIRECT
    )

    if client < 0:

        print(
            "[FAIL] PyBullet connection failed."
        )

        return

    try:

        p.setGravity(
            0,
            0,
            -9.81
        )

        robot_id = robot.create_robot()

        print(
            "[PASS] Production robot created."
        )

        print(
            f"Robot ID = {robot_id}"
        )

        num_joints = p.getNumJoints(
            robot_id
        )

        print(
            f"Number of joints = {num_joints}"
        )

        # ============================================================
        # 6. INSPECT EVERY LINK
        # ============================================================

        print()
        print("-" * 80)
        print("5. COLLISION SHAPE BY LINK")
        print("-" * 80)

        link_data = {}

        for link_index in range(
            -1,
            num_joints
        ):

            data = inspect_link(
                robot_id,
                link_index
            )

            if data is None:

                print()
                print(
                    f"Link {link_index}: "
                    "[NO COLLISION SHAPE RETURNED]"
                )

                continue

            link_data[link_index] = data

            print()
            print(
                f"Link {link_index}"
            )

            print(
                f"  Shape type : {data[2]}"
            )

            print(
                f"  Dimensions : {data[3]}"
            )

            print(
                f"  Local pos  : {data[5]}"
            )

            print(
                f"  Local orn  : {data[6]}"
            )

        # ============================================================
        # 7. CASTER
        # ============================================================

        print()
        print("-" * 80)
        print("6. CASTER LINK 3")
        print("-" * 80)

        caster = link_data.get(
            3
        )

        if caster is None:

            print(
                "[FAIL] No collision shape returned for link 3."
            )

            print()
            print(
                "This means the production robot is not exposing "
                "a collision shape for the caster link."
            )

            return

        shape_type = caster[2]
        dimensions = caster[3]
        local_position = caster[5]
        local_orientation = caster[6]

        print(
            f"Shape type = {shape_type}"
        )

        print(
            f"Dimensions = {dimensions}"
        )

        print(
            f"Local pos  = {local_position}"
        )

        print(
            f"Local orn  = {local_orientation}"
        )

        # ============================================================
        # 8. TYPE
        # ============================================================

        print()
        print("-" * 80)
        print("7. SHAPE TYPE")
        print("-" * 80)

        cylinder_pass = (
            shape_type == p.GEOM_CYLINDER
        )

        print(
            f"Expected = {p.GEOM_CYLINDER}"
        )

        print(
            f"Actual   = {shape_type}"
        )

        print(
            "[PASS] Caster is a cylinder."
            if cylinder_pass
            else
            "[FAIL] Caster is not a cylinder."
        )

        # ============================================================
        # 9. DIMENSIONS
        # ============================================================

        print()
        print("-" * 80)
        print("8. CASTER DIMENSIONS")
        print("-" * 80)

        actual_radius = dimensions[0]
        actual_height = dimensions[1]

        print(
            f"Expected radius = {EXPECTED_RADIUS:.6f}"
        )

        print(
            f"Actual radius   = {actual_radius:.6f}"
        )

        print()

        print(
            f"Expected width  = {EXPECTED_WIDTH:.6f}"
        )

        print(
            f"Actual height   = {actual_height:.6f}"
        )

        radius_pass = (
            abs(
                actual_radius
                - EXPECTED_RADIUS
            )
            < 1e-9
        )

        width_pass = (
            abs(
                actual_height
                - EXPECTED_WIDTH
            )
            < 1e-9
        )

        print()

        print(
            "[PASS] Radius is correct."
            if radius_pass
            else
            "[FAIL] Radius is incorrect."
        )

        print(
            "[PASS] Width is correct."
            if width_pass
            else
            "[FAIL] Width is incorrect."
        )

        # ============================================================
        # 10. REVERSED
        # ============================================================

        print()
        print("-" * 80)
        print("9. REVERSED GEOMETRY CHECK")
        print("-" * 80)

        reversed_geometry = (
            abs(
                actual_radius
                - EXPECTED_WIDTH
            )
            < 1e-9
            and
            abs(
                actual_height
                - EXPECTED_RADIUS
            )
            < 1e-9
        )

        if reversed_geometry:

            print(
                "[CONFIRMED] Runtime dimensions are reversed."
            )

        else:

            print(
                "[PASS] Runtime dimensions are not reversed."
            )

        # ============================================================
        # 11. ORIENTATION
        # ============================================================

        print()
        print("-" * 80)
        print("10. COLLISION ORIENTATION")
        print("-" * 80)

        expected_orientation = (
            p.getQuaternionFromEuler(
                [
                    1.5707963267948966,
                    0.0,
                    0.0
                ]
            )
        )

        print(
            f"Expected = {expected_orientation}"
        )

        print(
            f"Actual   = {local_orientation}"
        )

        orientation_pass = all(
            abs(
                local_orientation[i]
                - expected_orientation[i]
            )
            < 1e-6
            for i in range(4)
        )

        print(
            "[PASS] Collision orientation is correct."
            if orientation_pass
            else
            "[INVESTIGATE] Collision orientation differs."
        )

        # ============================================================
        # 12. HIERARCHY
        # ============================================================

        print()
        print("-" * 80)
        print("11. CASTER HIERARCHY")
        print("-" * 80)

        joint3 = p.getJointInfo(
            robot_id,
            3
        )

        parent_index = joint3[16]
        joint_axis = joint3[13]
        joint_position = joint3[14]

        print(
            f"Parent index = {parent_index}"
        )

        print(
            f"Joint axis   = {joint_axis}"
        )

        print(
            f"Local pos    = {joint_position}"
        )

        hierarchy_pass = (
            parent_index == 2
            and
            joint_axis == (
                0.0,
                1.0,
                0.0
            )
        )

        print(
            "[PASS] Caster hierarchy is correct."
            if hierarchy_pass
            else
            "[FAIL] Caster hierarchy is incorrect."
        )

        # ============================================================
        # 13. FINAL
        # ============================================================

        print()
        print("-" * 80)
        print("12. FINAL DIAGNOSIS")
        print("-" * 80)

        all_pass = (
            source_pass
            and
            runtime_pass
            and
            cylinder_pass
            and
            radius_pass
            and
            width_pass
            and
            orientation_pass
            and
            hierarchy_pass
        )

        if all_pass:

            print(
                "[PASS] V3.22.5i"
            )

            print(
                "Production caster collision geometry "
                "is correct."
            )

            print()
            print(
                "The earlier runtime dimension mismatch "
                "is NOT reproduced."
            )

        elif reversed_geometry:

            print(
                "[FAIL] V3.22.5i"
            )

            print(
                "Production caster dimensions are reversed."
            )

        else:

            print(
                "[INVESTIGATE] V3.22.5i"
            )

            print(
                "Production caster geometry still "
                "does not match the expected model."
            )

        print()
        print(
            "robot.py was NOT modified."
        )

        print(
            "PPO was NOT trained."
        )

        print()
        print(
            "V3.22.5i COMPLETE"
        )

        print("=" * 80)

    finally:

        p.disconnect()


if __name__ == "__main__":
    main()