"""
V3.22.5j — Production Collision-Shape Call Trace

Purpose:
    Intercept PyBullet's createCollisionShape() while production
    robot.py creates the robot.

    This tells us EXACTLY what robot.py passes to PyBullet.

Expected caster call:

    GEOM_CYLINDER
    radius = 0.040
    height = 0.025

If the intercepted call shows:

    radius = 0.025
    height = 0.040

then the production construction path is explicitly reversed.

This test:
    - does NOT modify robot.py
    - does NOT train PPO
"""

from pathlib import Path
import importlib.util
import pybullet as p


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_FILE = PROJECT_ROOT / "robot.py"

EXPECTED_RADIUS = 0.040
EXPECTED_WIDTH = 0.025


# ----------------------------------------------------------------------
# Save original PyBullet function
# ----------------------------------------------------------------------

original_create_collision_shape = p.createCollisionShape


captured_calls = []


# ----------------------------------------------------------------------
# Interceptor
# ----------------------------------------------------------------------

def traced_create_collision_shape(*args, **kwargs):

    record = {
        "args": args,
        "kwargs": dict(kwargs),
    }

    captured_calls.append(record)

    return original_create_collision_shape(
        *args,
        **kwargs
    )


# ----------------------------------------------------------------------
# Load production robot
# ----------------------------------------------------------------------

def load_robot():

    spec = importlib.util.spec_from_file_location(
        "production_robot_v3225j",
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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    print("=" * 80)
    print("V3.22.5j — PRODUCTION COLLISION-SHAPE CALL TRACE")
    print("=" * 80)

    print()
    print("Production file:")
    print(f"  {ROBOT_FILE}")

    if not ROBOT_FILE.exists():

        print()
        print("[FAIL] robot.py not found.")
        return

    # ------------------------------------------------------------------
    # Install interceptor BEFORE importing/creating robot
    # ------------------------------------------------------------------

    p.createCollisionShape = traced_create_collision_shape

    client = -1

    try:

        # --------------------------------------------------------------
        # Connect
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("1. PYBULLET CONNECTION")
        print("-" * 80)

        client = p.connect(
            p.DIRECT
        )

        if client < 0:

            print(
                "[FAIL] PyBullet connection failed."
            )

            return

        print(
            "[PASS] PyBullet connected."
        )

        # --------------------------------------------------------------
        # Load robot
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("2. LOAD PRODUCTION ROBOT")
        print("-" * 80)

        robot = load_robot()

        print(
            "[PASS] robot.py imported."
        )

        # --------------------------------------------------------------
        # Runtime constants
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("3. RUNTIME CONSTANTS")
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
            f"CASTER_WHEEL_RADIUS = "
            f"{runtime_radius}"
        )

        print(
            f"CASTER_WHEEL_WIDTH  = "
            f"{runtime_width}"
        )

        # --------------------------------------------------------------
        # Create robot
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("4. CREATE PRODUCTION ROBOT")
        print("-" * 80)

        robot_id = robot.create_robot()

        print(
            "[PASS] Production robot created."
        )

        print(
            f"Robot ID = {robot_id}"
        )

        # --------------------------------------------------------------
        # Captured calls
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("5. CAPTURED createCollisionShape() CALLS")
        print("-" * 80)

        print(
            f"Total calls = {len(captured_calls)}"
        )

        for index, call in enumerate(
            captured_calls,
            start=1
        ):

            args = call["args"]
            kwargs = call["kwargs"]

            print()
            print(
                f"CALL {index}"
            )

            print(
                f"  positional args = {args}"
            )

            print(
                f"  keyword args    = {kwargs}"
            )

            if kwargs:

                if "shapeType" in kwargs:

                    print(
                        f"  shapeType = "
                        f"{kwargs['shapeType']}"
                    )

                if "radius" in kwargs:

                    print(
                        f"  radius = "
                        f"{kwargs['radius']}"
                    )

                if "height" in kwargs:

                    print(
                        f"  height = "
                        f"{kwargs['height']}"
                    )

        # --------------------------------------------------------------
        # Find cylinder calls
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("6. CYLINDER CALL ANALYSIS")
        print("-" * 80)

        cylinder_calls = []

        for index, call in enumerate(
            captured_calls,
            start=1
        ):

            args = call["args"]
            kwargs = call["kwargs"]

            shape_type = None

            if "shapeType" in kwargs:

                shape_type = kwargs["shapeType"]

            elif len(args) > 0:

                shape_type = args[0]

            if shape_type == p.GEOM_CYLINDER:

                cylinder_calls.append(
                    (
                        index,
                        call
                    )
                )

        print(
            f"Cylinder calls = "
            f"{len(cylinder_calls)}"
        )

        for index, call in cylinder_calls:

            kwargs = call["kwargs"]

            print()
            print(
                f"CYLINDER CALL {index}"
            )

            print(
                f"  radius = "
                f"{kwargs.get('radius', '<not supplied>')}"
            )

            print(
                f"  height = "
                f"{kwargs.get('height', '<not supplied>')}"
            )

            print(
                f"  orientation = "
                f"{kwargs.get('collisionFrameOrientation', '<not supplied>')}"
            )

        # --------------------------------------------------------------
        # Match expected caster construction
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("7. EXPECTED CASTER CONSTRUCTION")
        print("-" * 80)

        print(
            f"Expected radius = "
            f"{EXPECTED_RADIUS}"
        )

        print(
            f"Expected height = "
            f"{EXPECTED_WIDTH}"
        )

        # --------------------------------------------------------------
        # Diagnose
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("8. DIAGNOSIS")
        print("-" * 80)

        reversed_found = False
        correct_found = False

        for _, call in cylinder_calls:

            kwargs = call["kwargs"]

            radius = kwargs.get(
                "radius"
            )

            height = kwargs.get(
                "height"
            )

            if (
                radius == EXPECTED_RADIUS
                and
                height == EXPECTED_WIDTH
            ):

                correct_found = True

            if (
                radius == EXPECTED_WIDTH
                and
                height == EXPECTED_RADIUS
            ):

                reversed_found = True

        if reversed_found:

            print(
                "[CONFIRMED] Production PyBullet call "
                "passes the caster dimensions REVERSED."
            )

            print()
            print(
                "The problem is NOT:"
            )

            print(
                "  - CASTER_WHEEL_RADIUS constant"
            )

            print(
                "  - CASTER_WHEEL_WIDTH constant"
            )

            print(
                "  - joint hierarchy"
            )

            print(
                "  - collision orientation"
            )

            print()
            print(
                "The problem is in the actual "
                "collision-shape construction path."
            )

        elif correct_found:

            print(
                "[CONFIRMED] Production call passes "
                "the correct caster dimensions."
            )

            print()
            print(
                "If PyBullet still reports reversed "
                "dimensions, the next issue is PyBullet "
                "shape ownership or shape reuse."
            )

        else:

            print(
                "[INVESTIGATE] No clear caster "
                "construction call was identified."
            )

        # --------------------------------------------------------------
        # Final
        # --------------------------------------------------------------

        print()
        print("-" * 80)
        print("FINAL STATUS")
        print("-" * 80)

        print(
            "robot.py was NOT modified."
        )

        print(
            "PPO was NOT trained."
        )

        print()
        print(
            "V3.22.5j COMPLETE"
        )

        print("=" * 80)

    finally:

        # Restore PyBullet function
        p.createCollisionShape = (
            original_create_collision_shape
        )

        if client >= 0:

            p.disconnect()


if __name__ == "__main__":
    main()