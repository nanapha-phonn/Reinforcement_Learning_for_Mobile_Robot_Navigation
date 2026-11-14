"""
V3.22.5h — Production Caster Constant Audit

Purpose:
    Automatically inspect robot.py and determine the actual values
    assigned to:

        CASTER_WHEEL_RADIUS
        CASTER_WHEEL_WIDTH

This test does NOT modify robot.py.
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_FILE = PROJECT_ROOT / "robot.py"

EXPECTED_RADIUS = 0.040
EXPECTED_WIDTH = 0.025


def main():
    print("=" * 80)
    print("V3.22.5h — PRODUCTION CASTER CONSTANT AUDIT")
    print("=" * 80)

    print()
    print(f"Production file:")
    print(f"  {ROBOT_FILE}")

    if not ROBOT_FILE.exists():
        print()
        print("[FAIL] robot.py not found.")
        return

    source = ROBOT_FILE.read_text(
        encoding="utf-8"
    )

    try:
        tree = ast.parse(
            source,
            filename=str(ROBOT_FILE)
        )
    except SyntaxError as error:
        print()
        print("[FAIL] robot.py contains a syntax error.")
        print(f"Line: {error.lineno}")
        print(f"Error: {error.msg}")
        return

    constants = {}

    target_names = {
        "CASTER_WHEEL_RADIUS",
        "CASTER_WHEEL_WIDTH",
    }

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id in target_names
                ):

                    value = node.value

                    if isinstance(
                        value,
                        ast.Constant
                    ):
                        constants[target.id] = value.value

                    else:
                        try:
                            constants[target.id] = ast.unparse(
                                value
                            )
                        except Exception:
                            constants[target.id] = "<expression>"

        elif isinstance(node, ast.AnnAssign):

            target = node.target

            if (
                isinstance(target, ast.Name)
                and target.id in target_names
            ):

                value = node.value

                if value is not None:

                    if isinstance(
                        value,
                        ast.Constant
                    ):
                        constants[target.id] = value.value

                    else:
                        try:
                            constants[target.id] = ast.unparse(
                                value
                            )
                        except Exception:
                            constants[target.id] = "<expression>"

    print()
    print("-" * 80)
    print("PRODUCTION CONSTANTS")
    print("-" * 80)

    for name in sorted(target_names):

        if name in constants:
            print(
                f"{name} = {constants[name]}"
            )
        else:
            print(
                f"{name} = NOT FOUND"
            )

    radius = constants.get(
        "CASTER_WHEEL_RADIUS"
    )

    width = constants.get(
        "CASTER_WHEEL_WIDTH"
    )

    print()
    print("-" * 80)
    print("EXPECTED VALUES")
    print("-" * 80)

    print(
        f"CASTER_WHEEL_RADIUS = "
        f"{EXPECTED_RADIUS}"
    )

    print(
        f"CASTER_WHEEL_WIDTH  = "
        f"{EXPECTED_WIDTH}"
    )

    print()
    print("-" * 80)
    print("DIAGNOSIS")
    print("-" * 80)

    if radius == EXPECTED_RADIUS and width == EXPECTED_WIDTH:

        print("[PASS] Constants are correct.")

    elif radius == EXPECTED_WIDTH and width == EXPECTED_RADIUS:

        print("[FAIL] Constants are REVERSED.")

        print()
        print("Current:")
        print(
            f"  CASTER_WHEEL_RADIUS = {radius}"
        )
        print(
            f"  CASTER_WHEEL_WIDTH  = {width}"
        )

        print()
        print("Required:")
        print(
            f"  CASTER_WHEEL_RADIUS = {EXPECTED_RADIUS}"
        )
        print(
            f"  CASTER_WHEEL_WIDTH  = {EXPECTED_WIDTH}"
        )

    else:

        print("[FAIL] Constants do not match expected geometry.")

        print()
        print(
            f"Production radius = {radius}"
        )

        print(
            f"Production width  = {width}"
        )

    print()
    print("-" * 80)
    print("IMPORTANT")
    print("-" * 80)

    print(
        "This test does NOT modify robot.py."
    )

    print(
        "This test does NOT retrain PPO."
    )

    print()
    print("V3.22.5h COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
