"""
V3.22.5i — Production Caster Collision-Shape Audit

Does NOT modify robot.py.
Does NOT train PPO.

Checks:
1. Production constants
2. Runtime constants
3. Actual PyBullet caster collision dimensions
4. Collision orientation
5. Caster hierarchy
6. Final diagnosis
"""

from pathlib import Path
import ast
import importlib.util

import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_FILE = PROJECT_ROOT / "robot.py"

EXPECTED_RADIUS = 0.040
EXPECTED_WIDTH = 0.025


def read_source_constants():

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
        "production_robot_v3225i",
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


def main():

    print("=" * 80)
    print("V3.22.5i — PRODUCTION CASTER COLLISION-SHAPE AUDIT")
    print("=" * 80)

    print()
    print("Production file:")
    print(f"  {ROBOT_FILE}")

    if not ROBOT_FILE.exists():

        print()
        print("[FAIL] robot.py not found.")
        return

    # ================================================================
    # 1. SOURCE CONSTANTS
    # ================================================================

    print()
    print("-" * 80)
    print("1. SOURCE CONSTANT AUDIT")
    print("-" * 80)

    constants = read_source_constants()

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
    # 2. LOAD ROBOT
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
    # 3. RUNTIME CONSTANTS
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
    # 4. CONNECT
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

        # IMPORTANT:
        # Production create_robot() takes NO arguments.
        robot_id = robot.create_robot()

        print(
            "[PASS] Production robot created."
        )

        print(
            f"Robot ID = {robot_id}"
        )

        # ============================================================
        # 5. RUNTIME COLLISION SHAPES
        # ============================================================

        print()
        print("-" * 80)
        print("5. RUNTIME COLLISION SHAPES")
        print("-" * 80)

        shapes = p.getCollisionShapeData(
            robot_id,
            -1
        )

        if not shapes:

            print(
                "[FAIL] No collision shapes returned."
            )

            return

        caster_shape = None

        for item in shapes:

            link_index = item[1]
            shape_type = item[2]
            dimensions = item[3]
            local_position = item[5]
            local_orientation = item[6]

            print()
            print(
                f"Link {link_index}"
            )

            print(
                f"  Shape type : {shape_type}"
            )

            print(
                f"  Dimensions : {dimensions}"
            )

            print(
                f"  Local pos  : {local_position}"
            )

            print(
                f"  Local orn  : {local_orientation}"
            )

            if link_index == 3:

                caster_shape = item

        # ============================================================
        # 6. CASTER SHAPE
        # ============================================================

        print()
        print("-" * 80)
        print("6. CASTER COLLISION SHAPE")
        print("-" * 80)

        if caster_shape is None:

            print(
                "[FAIL] Link 3 caster collision shape not found."
            )

            return

        shape_type = caster_shape[2]
        dimensions = caster_shape[3]
        local_position = caster_shape[5]
        local_orientation = caster_shape[6]

        actual_radius = dimensions[0]
        actual_height = dimensions[1]

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
        # 7. DIMENSION CHECK
        # ============================================================

        print()
        print("-" * 80)
        print("7. DIMENSION CHECK")
        print("-" * 80)

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
            abs(actual_radius - EXPECTED_RADIUS)
            < 1e-9
        )

        width_pass = (
            abs(actual_height - EXPECTED_WIDTH)
            < 1e-9
        )

        print()

        print(
            "[PASS] Runtime radius is correct."
            if radius_pass
            else
            "[FAIL] Runtime radius is WRONG."
        )

        print(
            "[PASS] Runtime width is correct."
            if width_pass
            else
            "[FAIL] Runtime width is WRONG."
        )

        # ============================================================
        # 8. REVERSED CHECK
        # ============================================================

        print()
        print("-" * 80)
        print("8. REVERSED-DIMENSION CHECK")
        print("-" * 80)

        reversed_geometry = (
            abs(actual_radius - EXPECTED_WIDTH)
            < 1e-9
            and
            abs(actual_height - EXPECTED_RADIUS)
            < 1e-9
        )

        if reversed_geometry:

            print(
                "[CONFIRMED] Runtime dimensions are reversed."
            )

        else:

            print(
                "[PASS] Runtime dimensions are not simply reversed."
            )

        # ============================================================
        # 9. COLLISION ORIENTATION
        # ============================================================

        print()
        print("-" * 80)
        print("9. COLLISION ORIENTATION")
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
            ) < 1e-6
            for i in range(4)
        )

        print(
            "[PASS] Collision orientation is correct."
            if orientation_pass
            else
            "[INVESTIGATE] Collision orientation differs."
        )

        # ============================================================
        # 10. HIERARCHY
        # ============================================================

        print()
        print("-" * 80)
        print("10. CASTER JOINT HIERARCHY")
        print("-" * 80)

        num_joints = p.getNumJoints(
            robot_id
        )

        print(
            f"Number of joints = {num_joints}"
        )

        hierarchy_pass = False

        if num_joints >= 4:

            info = p.getJointInfo(
                robot_id,
                3
            )

            parent = info[16]
            axis = info[13]

            print(
                f"Joint 3 parent = {parent}"
            )

            print(
                f"Joint 3 axis   = {axis}"
            )

            hierarchy_pass = (
                parent == 2
                and
                axis == (
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

        else:

            print(
                "[FAIL] Less than 4 joints."
            )

        # ============================================================
        # 11. FINAL
        # ============================================================

        print()
        print("-" * 80)
        print("11. FINAL DIAGNOSIS")
        print("-" * 80)

        if (
            source_pass
            and
            runtime_pass
            and
            radius_pass
            and
            width_pass
            and
            orientation_pass
            and
            hierarchy_pass
        ):

            print(
                "[PASS] Production caster geometry is CORRECT."
            )

            print(
                "The previous reported dimension mismatch "
                "was not reproduced."
            )

        elif (
            source_pass
            and
            runtime_pass
            and
            reversed_geometry
        ):

            print(
                "[CONFIRMED] Constants are correct."
            )

            print(
                "[CONFIRMED] Runtime collision dimensions "
                "are reversed."
            )

            print(
                "Investigate the production collision-shape "
                "construction code next."
            )

        else:

            print(
                "[INVESTIGATE] Production geometry mismatch remains."
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