import math


HEADING_WEIGHT = 0.02
HEADING_DISTANCE = 1.0


def clip(value, lower, upper):
    return max(lower, min(value, upper))


def heading_reward(distance, bearing_deg):
    weight = clip(
        1.0 - distance / HEADING_DISTANCE,
        0.0,
        1.0,
    )

    bearing_rad = math.radians(bearing_deg)

    return HEADING_WEIGHT * weight * math.cos(bearing_rad)


def test_heading_reward():

    test_cases = [
        (2.0, 0.0, 0.000),
        (1.0, 0.0, 0.000),
        (0.75, 0.0, 0.005),
        (0.50, 0.0, 0.010),
        (0.40, 0.0, 0.012),
        (0.20, 0.0, 0.016),
        (0.50, 90.0, 0.000),
        (0.50, 180.0, -0.010),
        (0.20, 180.0, -0.016),
    ]

    tolerance = 1e-9

    for distance, bearing, expected in test_cases:

        actual = heading_reward(distance, bearing)

        print(
            f"D={distance:.2f} m | "
            f"bearing={bearing:6.1f} deg | "
            f"heading reward={actual:+.6f} | "
            f"expected={expected:+.6f}"
        )

        assert abs(actual - expected) < tolerance


def test_heading_reward_properties():

    # Far from goal: no heading shaping
    assert heading_reward(2.0, 0.0) == 0.0

    # Exactly at the boundary: no heading shaping
    assert heading_reward(1.0, 0.0) == 0.0

    # Facing goal: positive
    assert heading_reward(0.5, 0.0) > 0.0

    # Perpendicular: zero
    assert abs(heading_reward(0.5, 90.0)) < 1e-12

    # Facing away: negative
    assert heading_reward(0.5, 180.0) < 0.0

    # Closer to goal increases maximum possible heading reward
    assert heading_reward(0.2, 0.0) > heading_reward(0.5, 0.0)

    print("All mathematical properties: PASS")


if __name__ == "__main__":
    test_heading_reward()
    test_heading_reward_properties()

    print()
    print("V3.21.7 REWARD MATHEMATICS TEST: PASS")