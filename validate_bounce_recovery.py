"""Validate that missed-bounce recovery uses observations, not tracker holds."""

from analyze_tennis_events import observed_ball_rows


def main():
    rows = [
        {"frame": 1, "ball": {"center": [10, 10]}},
        {"frame": 2, "ball": {"center": [11, 11], "interpolated": True}},
        {"frame": 3, "ball": {"center": [12, 12], "motion_gate": "coast"}},
        {"frame": 4, "ball": {"center": [13, 13]}},
    ]
    observed = observed_ball_rows(rows)
    if [row["frame"] for row in observed] != [1, 4]:
        print("bounce recovery must exclude interpolated and coasted rows")
        return 1
    print("bounce recovery observation filter passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
