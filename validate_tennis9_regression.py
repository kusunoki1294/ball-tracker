import argparse
import csv
import json
import os
import sys


EXPECTED_POINTS = [
    {
        "index": 1,
        "serve_state": "played_out_after_geometric_fault",
        "serve_confidence": "low",
        "point_end_reason": "out",
        "point_end_type": "unforced_error_out",
        "point_end_confidence": "low",
        "winner": "near",
        "winner_source": "manual_low_confidence_auto_fallback",
        "score_after": "0-15",
        "game_score_after": "0-0",
        "set_score_after": "0-0",
        "review_flags": {
            "serve_geometry_disagrees_with_play_continuation",
            "low_confidence_terminal_out",
            "final_ball_out_of_frame",
        },
    },
    {
        "index": 2,
        "serve_state": "second_serve_in",
        "serve_confidence": "high",
        "point_end_reason": "out",
        "point_end_type": "unforced_error_out",
        "point_end_confidence": "medium",
        "winner": "near",
        "winner_source": "auto",
        "score_after": "0-30",
        "game_score_after": "0-0",
        "set_score_after": "0-0",
        "review_flags": set(),
    },
    {
        "index": 3,
        "serve_state": "double_fault",
        "serve_confidence": "high",
        "point_end_reason": "double_fault",
        "point_end_type": "double_fault",
        "point_end_confidence": "high",
        "winner": "near",
        "winner_source": "auto",
        "score_after": "0-40",
        "game_score_after": "0-0",
        "set_score_after": "0-0",
        "review_flags": set(),
    },
    {
        "index": 4,
        "serve_state": "double_fault",
        "serve_confidence": "high",
        "point_end_reason": "double_fault",
        "point_end_type": "double_fault",
        "point_end_confidence": "high",
        "winner": "near",
        "winner_source": "auto",
        "score_after": "0-0",
        "game_score_after": "0-1",
        "set_score_after": "0-0",
        "review_flags": set(),
    },
]

REQUIRED_AUDIT_COLUMNS = {
    "serve_state",
    "serve_confidence",
    "serve_reasons",
    "point_end_type",
    "point_end_confidence",
    "point_end_reasons",
    "point_review_flags",
}

EXPECTED_MISSED_BOUNCE_CANDIDATES = []


def parse_args():
    parser = argparse.ArgumentParser(description="Validate known-good tennis9 analysis outputs.")
    parser.add_argument(
        "--analysis",
        default="yoloVids/outputs/tennis9/play_segments/ai9.5.analysis.json",
        help="Analysis JSON produced by run_tennis_pipeline.py.",
    )
    parser.add_argument(
        "--audit-csv",
        default="yoloVids/outputs/tennis9/play_segments/ai9.5.audit.csv",
        help="Audit CSV produced by export_tennis_audit.py.",
    )
    parser.add_argument(
        "--point-debug-dir",
        default="yoloVids/outputs/tennis9/play_segments/point_debug",
        help="Directory of per-point debug PNGs produced by export_tennis_audit.py.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_equal(errors, label, actual, expected):
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def validate_points(analysis):
    errors = []
    points = analysis.get("points") or []
    check_equal(errors, "point count", len(points), len(EXPECTED_POINTS))
    points_by_index = {point.get("index"): point for point in points}
    for expected in EXPECTED_POINTS:
        point = points_by_index.get(expected["index"])
        if not point:
            errors.append(f"point {expected['index']}: missing")
            continue
        for key, expected_value in expected.items():
            if key == "review_flags":
                flags = set(point.get("point_review_flags") or [])
                missing = expected_value - flags
                unexpected = flags - expected_value
                if missing:
                    errors.append(f"point {expected['index']} review flags: missing {sorted(missing)}")
                if unexpected:
                    errors.append(f"point {expected['index']} review flags: unexpected {sorted(unexpected)}")
                continue
            check_equal(errors, f"point {expected['index']} {key}", point.get(key), expected_value)

    summary = analysis.get("summary") or {}
    check_equal(errors, "summary final_point_score", summary.get("final_point_score"), "0-0")
    check_equal(errors, "summary final_game_score", summary.get("final_game_score"), "0-1")
    check_equal(errors, "summary final_set_score", summary.get("final_set_score"), "0-0")
    candidates = analysis.get("missed_bounce_candidates") or []
    check_equal(errors, "missed bounce candidate count", len(candidates), len(EXPECTED_MISSED_BOUNCE_CANDIDATES))
    for expected, actual in zip(EXPECTED_MISSED_BOUNCE_CANDIDATES, candidates):
        for key, value in expected.items():
            check_equal(errors, f"missed bounce candidate {expected['frame']} {key}", actual.get(key), value)
    return errors


def validate_audit_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_AUDIT_COLUMNS - columns)
    if missing:
        return [f"audit CSV missing columns: {', '.join(missing)}"]
    return []


def validate_point_debug_images(directory):
    errors = []
    for point in EXPECTED_POINTS:
        path = os.path.join(directory, f"point_{point['index']:02d}.png")
        if not os.path.exists(path):
            errors.append(f"missing point debug image: {path}")
        elif os.path.getsize(path) <= 0:
            errors.append(f"empty point debug image: {path}")
    return errors


def main():
    args = parse_args()
    errors = []
    analysis = load_json(args.analysis)
    errors.extend(validate_points(analysis))
    errors.extend(validate_audit_csv(args.audit_csv))
    errors.extend(validate_point_debug_images(args.point_debug_dir))
    if errors:
        print("tennis9 regression validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("tennis9 regression validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
