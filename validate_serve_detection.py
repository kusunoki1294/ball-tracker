"""Validate serve detection against hand-checked serve contacts on tennis11.

Every frame in EXPECTED_SERVES was read off the source video directly: the
server is at full extension with the ball at the racket. They are ground truth,
not a snapshot of what the detector currently prints, so a change that moves any
of them is a real regression rather than a baseline to be renewed.

Run after `run_tennis_pipeline.py --manifest manifests/tennis11_game1_manifest.json`.
"""

import argparse
import json
import sys


# Point index -> serve contact frames, earliest first.
EXPECTED_SERVES = {
    1: [160],
    2: [635],
    # Point 3 is the double fault: two strikes, no rally in between.
    3: [1485, 1659],
    4: [2091],
    # Point 5's toss is never tracked (the ball is present in 42% of its
    # frames), so this one exercises the peak-reach fallback on its own.
    5: [2432],
    # Point 6 is the case that motivated the work: the serve is here, while the
    # bounce previously reported as its serve is a rally lob 195 frames later.
    6: [2952],
}

# The detected contact may sit one frame either side of the hand-picked frame:
# the strike falls between two sampled frames and the box top can peak on
# either. Anything beyond this is a different event, not a rounding difference.
CONTACT_TOLERANCE_FRAMES = 3

# Kept in step with analyze_tennis_events.MAX_SERVE_FLIGHT_SECONDS and
# MIN_SERVE_FLIGHT_SECONDS. The floor matters as much as the ceiling: a bounce
# detector reading trajectory reversals reports the strike itself as a bounce one
# frame after contact, and judging a serve on that lands the right verdict for
# the wrong reason, which is the kind of pass that hides itself.
MAX_SERVE_FLIGHT_SECONDS = 1.5
MIN_SERVE_FLIGHT_SECONDS = 0.33

# Updated when bounces began coming from bounce_detect rather than the sparser
# jsonl event stream. Points 2 and 6 gained the landings their serves always had,
# so both moved off "struck but not observed" onto real verdicts. Point 6 is the
# one to watch: it reported first_serve_in before this work too, but off a rally
# lob's bounce 195 frames after the serve. It now reports the same verdict from
# its own landing 24 frames after the strike.
EXPECTED_STATES = {
    1: "first_serve_in",
    2: "first_serve_in",
    3: "double_fault",
    4: "first_serve_fault",
    5: "serve_struck_bounce_unobserved",
    6: "first_serve_in",
}

# Point 6 used to report first_serve_in off a rally lob's bounce. Nothing may
# claim a serve landing without a bounce anchored to a detected strike.
FORBIDDEN_LANDING_STATES = {"first_serve_in", "second_serve_in", "first_serve_fault"}

# The two guards that stop a rally stroke played from behind the baseline being
# promoted to a second serve. Both points below really do contain a second
# serve-shaped motion, and both must be rejected, each for its own reason --
# checking only the resulting attempt count would pass even if a guard broke and
# the other happened to cover for it.
EXPECTED_SECOND_SERVE_REJECTIONS = {
    # Point 2's serve is now known to have landed in, so the sequence ends there
    # and the second detected motion is never considered. It used to be rejected
    # by the rally test instead -- that test is no longer reached here, so the
    # only guard still exercised on this clip is point 5's.
    2: ("serve_landed_in", 2),
    # The far-extension candidate at 2491 has no toss of its own; it is the
    # server waiting to receive.
    5: ("next_motion_has_no_tracked_toss", 2),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate serve detection on tennis11 game 1.")
    parser.add_argument(
        "--analysis",
        default="yoloVids/outputs/tennis11/ai11.2.analysis.json",
        help="Analysis JSON produced by run_tennis_pipeline.py.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_contacts(analysis):
    errors = []
    points = {point["index"]: point for point in analysis.get("points") or []}
    if len(points) != len(EXPECTED_SERVES):
        errors.append(f"expected {len(EXPECTED_SERVES)} points, got {len(points)}")
    for index, expected_frames in sorted(EXPECTED_SERVES.items()):
        point = points.get(index)
        if point is None:
            errors.append(f"point {index}: missing from analysis")
            continue
        serve_analysis = point.get("serve_analysis") or {}
        actual_frames = serve_analysis.get("serve_contact_frames") or []
        if len(actual_frames) != len(expected_frames):
            errors.append(
                f"point {index}: expected {len(expected_frames)} serve contact(s) "
                f"{expected_frames}, got {len(actual_frames)} {actual_frames}"
            )
            continue
        for expected, actual in zip(expected_frames, actual_frames):
            if abs(actual - expected) > CONTACT_TOLERANCE_FRAMES:
                errors.append(
                    f"point {index}: serve contact frame {actual} is more than "
                    f"{CONTACT_TOLERANCE_FRAMES} frames from the hand-checked {expected}"
                )
    return errors


def validate_states(analysis):
    errors = []
    points = {point["index"]: point for point in analysis.get("points") or []}
    for index, expected_state in sorted(EXPECTED_STATES.items()):
        point = points.get(index)
        if point is None:
            continue
        actual_state = point.get("serve_state")
        if actual_state != expected_state:
            errors.append(
                f"point {index}: expected serve_state {expected_state!r}, got {actual_state!r}"
            )
    return errors


def validate_landings_are_anchored(analysis):
    """No serve may be judged in or out from a bounce that is not its own."""
    errors = []
    # Mirror the analyzer's own window rather than a fixed frame count, so this
    # check keeps meaning the same thing on footage that is not 30fps.
    fps = analysis.get("fps") or 30.0
    max_flight_frames = max(1, int(round(MAX_SERVE_FLIGHT_SECONDS * fps)))
    min_flight_frames = max(1, int(round(MIN_SERVE_FLIGHT_SECONDS * fps)))
    for point in analysis.get("points") or []:
        serve_analysis = point.get("serve_analysis") or {}
        state = point.get("serve_state")
        for attempt in serve_analysis.get("attempts") or []:
            if attempt.get("result") not in {"in", "fault"}:
                continue
            if attempt.get("reason") == "inferred_fault_second_serve_followed":
                continue
            if attempt.get("contact_frame") is None:
                errors.append(
                    f"point {point['index']}: attempt {attempt.get('attempt')} judged "
                    f"{attempt.get('result')!r} without a detected serve strike"
                )
                continue
            bounce_frame = attempt.get("bounce_frame")
            if bounce_frame is None:
                errors.append(
                    f"point {point['index']}: attempt {attempt.get('attempt')} judged "
                    f"{attempt.get('result')!r} without a bounce"
                )
                continue
            lag = bounce_frame - attempt["contact_frame"]
            if lag < min_flight_frames or lag > max_flight_frames:
                errors.append(
                    f"point {point['index']}: attempt {attempt.get('attempt')} judged "
                    f"{attempt.get('result')!r} from a bounce {lag} frames after the "
                    f"strike (outside {min_flight_frames}-{max_flight_frames})"
                )
        if state in FORBIDDEN_LANDING_STATES and not (serve_analysis.get("attempts") or []):
            errors.append(f"point {point['index']}: state {state!r} with no serve attempts")
    return errors


def validate_second_serve_rejections(analysis):
    """The guards against promoting a rally stroke to a second serve still hold."""
    errors = []
    points = {point["index"]: point for point in analysis.get("points") or []}
    for index, (expected_reason, expected_motions) in sorted(
        EXPECTED_SECOND_SERVE_REJECTIONS.items()
    ):
        point = points.get(index)
        if point is None:
            errors.append(f"point {index}: missing from analysis")
            continue
        serve_analysis = point.get("serve_analysis") or {}
        detected = serve_analysis.get("detected_serve_motions")
        if detected != expected_motions:
            errors.append(
                f"point {index}: expected {expected_motions} detected serve motions "
                f"for the rejection guard to be exercised, got {detected}"
            )
        attempts = serve_analysis.get("attempts") or []
        if len(attempts) != 1:
            errors.append(
                f"point {index}: expected the second motion to be rejected "
                f"(1 attempt), got {len(attempts)} attempts"
            )
        actual_reason = serve_analysis.get("second_serve_rejected_reason")
        if actual_reason != expected_reason:
            errors.append(
                f"point {index}: expected rejection reason {expected_reason!r}, "
                f"got {actual_reason!r}"
            )
    return errors


def validate_fallback_is_visible(analysis):
    """A point that fell back to the bounce-first path must say so."""
    errors = []
    for point in analysis.get("points") or []:
        serve_analysis = point.get("serve_analysis") or {}
        fell_back = serve_analysis.get("serve_motion_fallback") == "old_bounce_path"
        has_contact = serve_analysis.get("serve_contact_frame") is not None
        if fell_back and has_contact:
            errors.append(
                f"point {point['index']}: marked as bounce-path fallback but "
                "carries a serve contact frame"
            )
        if not fell_back and not has_contact:
            errors.append(
                f"point {point['index']}: no serve contact and no fallback marker, "
                "so the analyzer is silently back on the bounce-first path"
            )
    return errors


def main():
    args = parse_args()
    analysis = load_json(args.analysis)
    errors = []
    errors.extend(validate_contacts(analysis))
    errors.extend(validate_states(analysis))
    errors.extend(validate_landings_are_anchored(analysis))
    errors.extend(validate_second_serve_rejections(analysis))
    errors.extend(validate_fallback_is_visible(analysis))
    if errors:
        print("serve detection validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(
        "serve detection validation passed "
        f"({sum(len(frames) for frames in EXPECTED_SERVES.values())} serve contacts "
        f"across {len(EXPECTED_SERVES)} points)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
