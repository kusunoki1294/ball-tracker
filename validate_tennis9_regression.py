import argparse
import csv
import json
import os
import sys


EXPECTED_POINTS = [
    {
        # Re-baselined 2026-08. This point's first detected bounce is at world
        # y=19.1, on the FAR player's own side, and far is the server - so it
        # cannot be that serve, and the serve bounce was simply missed. The
        # analyzer used to judge that rally bounce as a serve attempt, which is
        # what produced the old "played_out_after_geometric_fault" state (the
        # geometry said double fault while the players carried on). It now
        # reports serve_unobserved, and declines to call the point end rather
        # than defaulting to "out" when the ball track is lost.
        "index": 1,
        "serve_state": "serve_unobserved",
        "serve_confidence": "low",
        "point_end_reason": None,
        "point_end_type": "unknown_end",
        "point_end_confidence": "low",
        "winner": "near",
        "winner_source": "manual",
        "score_after": "0-15",
        "game_score_after": "0-0",
        "set_score_after": "0-0",
        "review_flags": {"unknown_point_end"},
    },
    {
        "index": 2,
        "serve_state": "first_serve_in",
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

EXPECTED_SHOT_TYPES = {
    # Point 1's serve was never detected (see EXPECTED_POINTS[0]), so its opening
    # shots are no longer labelled as serves. The old shot_001 "first serve by
    # far" was right about the player only because it inherited the server from a
    # serve attempt built out of a mid-rally bounce.
    "shot_001": {"type": "groundstroke", "player": "near", "serve_attempt": None},
    "shot_002": {"type": "groundstroke", "player": "far", "serve_attempt": None},
    "shot_012": {"type": "first_serve", "player": "far", "serve_attempt": 1},
    "shot_013": {"type": "return", "player": "near", "serve_attempt": None},
    "shot_018": {"type": "first_serve", "player": "far", "serve_attempt": 1},
    "shot_019": {"type": "second_serve", "player": "far", "serve_attempt": 2},
}


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
    parser.add_argument(
        "--report",
        default="yoloVids/outputs/tennis9/play_segments/match_report.html",
        help="Static HTML report produced by export_match_report.py.",
    )
    parser.add_argument(
        "--report-data",
        default="yoloVids/outputs/tennis9/play_segments/match_report_data.json",
        help="Compact report JSON produced by export_match_report.py.",
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
    shots_by_id = {shot.get("id"): shot for shot in analysis.get("shots") or []}
    for shot_id, expected in EXPECTED_SHOT_TYPES.items():
        shot = shots_by_id.get(shot_id)
        if not shot:
            errors.append(f"{shot_id}: missing")
            continue
        for key, value in expected.items():
            check_equal(errors, f"{shot_id} {key}", shot.get(key), value)
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


def validate_report(path):
    if not os.path.exists(path):
        return [f"missing match report: {path}"]
    if os.path.getsize(path) <= 0:
        return [f"empty match report: {path}"]
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    required = ["Point By Point", "Serve States", "Point Endings", "Shots", "Bounces"]
    missing = [text for text in required if text not in content]
    if missing:
        return [f"match report missing sections: {', '.join(missing)}"]
    return []


def validate_report_data(path):
    if not os.path.exists(path):
        return [f"missing match report data: {path}"]
    payload = load_json(path)
    errors = []
    check_equal(errors, "report data final_game_score", (payload.get("summary") or {}).get("final_game_score"), "0-1")
    check_equal(errors, "report data point count", len(payload.get("points") or []), len(EXPECTED_POINTS))
    for key in ["serve_states", "point_endings", "points_won", "trusted_stroke_sides"]:
        if key not in (payload.get("stats") or {}):
            errors.append(f"report data missing stats.{key}")
    return errors


def main():
    args = parse_args()
    errors = []
    analysis = load_json(args.analysis)
    errors.extend(validate_points(analysis))
    errors.extend(validate_audit_csv(args.audit_csv))
    errors.extend(validate_point_debug_images(args.point_debug_dir))
    errors.extend(validate_report(args.report))
    errors.extend(validate_report_data(args.report_data))
    if errors:
        print("tennis9 regression validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("tennis9 regression validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
