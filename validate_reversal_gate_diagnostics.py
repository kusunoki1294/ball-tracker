"""Validate the review-only reversal diagnostics context shape."""

from export_reversal_gate_diagnostics import scene_context


def main():
    item = {
        "ball": {
            "conf": 0.42,
            "bbox": [98, 98, 104, 104],
            "interpolated": True,
            "motion_gate": "coast",
        },
        "player_near": {"bbox": [0, 0, 20, 20], "conf": 0.9},
        "player_far": {"bbox": [90, 90, 110, 110], "conf": 0.8},
        "scene": [{"class_name": "tennis racket", "bbox": [105, 100, 115, 110]}],
    }
    context = scene_context(item, 101.0, 101.0)
    errors = []
    if context["ball_interpolated"] is not True:
        errors.append("ball interpolation provenance must be preserved")
    if context["ball_motion_gate"] != "coast":
        errors.append("ball motion gate must be preserved")
    if context["nearest_player"]["far"]["distance_px"] != 0.0:
        errors.append("nearest-player distance must use the full bbox")
    if context["nearest_racket_distance_px"] != 4.0:
        errors.append("nearest-racket distance must be reported")
    if errors:
        for error in errors:
            print(error)
        return 1
    print("reversal diagnostics validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
