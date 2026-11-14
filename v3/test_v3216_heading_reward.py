import math


def heading_reward(previous_bearing, current_bearing):

    heading_progress = (
        previous_bearing -
        abs(current_bearing)
    )

    reward = 0.01 * heading_progress

    return heading_progress, reward


def main():

    print("=== V3.21.6 HEADING REWARD TEST ===")

    # Better alignment
    progress, reward = heading_reward(
        0.8,
        0.4
    )

    print("\nCase 1: Better alignment")
    print("Heading progress:", progress)
    print("Heading reward:", reward)

    assert reward > 0

    # Worse alignment
    progress, reward = heading_reward(
        0.4,
        0.8
    )

    print("\nCase 2: Worse alignment")
    print("Heading progress:", progress)
    print("Heading reward:", reward)

    assert reward < 0

    # No change
    progress, reward = heading_reward(
        0.5,
        0.5
    )

    print("\nCase 3: No change")
    print("Heading progress:", progress)
    print("Heading reward:", reward)

    assert math.isclose(
        reward,
        0.0,
        abs_tol=1e-12
    )

    print("\nV3.21.6 REWARD VALIDATION: PASS")


if __name__ == "__main__":
    main()