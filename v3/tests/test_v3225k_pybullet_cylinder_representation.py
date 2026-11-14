"""
V3.22.5k — PyBullet Cylinder Representation Test

Purpose:
    Determine whether PyBullet's getCollisionShapeData()
    reports cylinder dimensions according to its internal
    representation rather than the radius/height arguments.

We create several cylinders with known radius/height values
and inspect the returned dimensions.

No robot.py modification.
No PPO training.
"""

import pybullet as p


TEST_CASES = [
    ("A", 0.040, 0.025, "rotated"),
    ("B", 0.025, 0.040, "rotated"),
    ("C", 0.040, 0.025, "unrotated"),
    ("D", 0.025, 0.040, "unrotated"),
]


def make_orientation(mode):

    if mode == "rotated":

        return p.getQuaternionFromEuler(
            [
                1.5707963267948966,
                0.0,
                0.0
            ]
        )

    return (
        0.0,
        0.0,
        0.0,
        1.0
    )


def main():

    print("=" * 80)
    print("V3.22.5k — PYBULLET CYLINDER REPRESENTATION TEST")
    print("=" * 80)

    client = p.connect(
        p.DIRECT
    )

    if client < 0:

        print(
            "[FAIL] Could not connect to PyBullet."
        )

        return

    try:

        for name, radius, height, mode in TEST_CASES:

            print()
            print("-" * 80)
            print(
                f"TEST {name} — "
                f"radius={radius}, "
                f"height={height}, "
                f"{mode}"
            )
            print("-" * 80)

            orientation = make_orientation(
                mode
            )

            shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                collisionFrameOrientation=orientation,
            )

            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=shape_id,
                basePosition=[
                    0.0,
                    0.0,
                    1.0
                ],
            )

            data = p.getCollisionShapeData(
                body_id,
                -1
            )[0]

            print(
                f"Input radius     = {radius}"
            )

            print(
                f"Input height     = {height}"
            )

            print(
                f"Input orientation= {orientation}"
            )

            print()

            print(
                f"Returned type   = {data[2]}"
            )

            print(
                f"Returned dims   = {data[3]}"
            )

            print(
                f"Returned pos    = {data[5]}"
            )

            print(
                f"Returned orient = {data[6]}"
            )

            print()

            returned_radius = data[3][0]
            returned_height = data[3][1]

            if (
                abs(returned_radius - radius)
                < 1e-9
                and
                abs(returned_height - height)
                < 1e-9
            ):

                print(
                    "[PASS] Returned dimensions match input."
                )

            elif (
                abs(returned_radius - height)
                < 1e-9
                and
                abs(returned_height - radius)
                < 1e-9
            ):

                print(
                    "[INFO] Returned dimensions are "
                    "swapped relative to input."
                )

            else:

                print(
                    "[INVESTIGATE] Returned dimensions "
                    "do not match either ordering."
                )

            # Remove test body before next case.
            p.removeBody(
                body_id
            )

    finally:

        p.disconnect()

    print()
    print("=" * 80)
    print("V3.22.5k COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()