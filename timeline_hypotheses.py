import argparse
import json
import os

from court_geometry import build_inverse_court_homography, order_court_corners
from serve_detect import (
    MAX_SECOND_SERVE_GAP_SECONDS,
    RALLY_BALL_RETURN_FRACTION,
    RALLY_MIN_TRACKED_SECONDS,
    REACH_SEARCH_SECONDS,
    ball_return_fraction,
    detect_serve_motions_for_point,
    frame_window,
)


FPS_DEFAULT = 30.0
COURT_NET_Y_FT = 39.0
SINGLES_LEFT_FT = 4.5
SINGLES_RIGHT_FT = 31.5
FAR_SERVICE_Y_MIN_FT = 18.0
FAR_SERVICE_Y_MAX_FT = 39.0
NEAR_SERVICE_Y_MIN_FT = 39.0
NEAR_SERVICE_Y_MAX_FT = 60.0
MAX_SERVE_FLIGHT_SECONDS = 1.5
MIN_SERVE_FLIGHT_SECONDS = 0.33
NET_LINE_CONTACT_BAND_FT = 2.0
SECOND_SERVE_MIN_TRACKED_SECONDS = 2.0
RALLY_CONTINUATION_SUPPRESS_SECONDS = 15.0
WEAK_REACH_CONTINUATION_SUPPRESS_SECONDS = 3.0
LONG_SUPPRESSION_REVIEW_SECONDS = 10.0
SINGLE_SERVER_MIN_VOTE_MARGIN = 5.0
SINGLE_SERVER_MIN_COUNT_MARGIN = 2


def read_tracking_log(path):
    rows = []
    by_frame = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["frame"])
            rows.append(row)
            by_frame[frame] = row
    rows.sort(key=lambda row: row["frame"])
    return rows, by_frame


def parse_point_ranges(raw):
    ranges = []
    if not raw:
        return ranges
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            frame = int(token)
            ranges.append({"start_frame": frame, "end_frame": frame})
            continue
        start, end = token.split("-", 1)
        ranges.append({"start_frame": int(start), "end_frame": int(end)})
    return ranges


def side_opponent(side):
    return "near" if side == "far" else "far"


def service_box_for_receiver(receiver):
    if receiver == "near":
        return {
            "y_min": NEAR_SERVICE_Y_MIN_FT,
            "y_max": NEAR_SERVICE_Y_MAX_FT,
        }
    return {
        "y_min": FAR_SERVICE_Y_MIN_FT,
        "y_max": FAR_SERVICE_Y_MAX_FT,
    }


def world_point_in_receiver_service_box(world_point, receiver, margin=0.35):
    if not world_point:
        return False
    xw, yw = world_point
    box = service_box_for_receiver(receiver)
    return (
        SINGLES_LEFT_FT - margin <= xw <= SINGLES_RIGHT_FT + margin
        and box["y_min"] - margin <= yw <= box["y_max"] + margin
    )


def bounce_on_net_line(bounce):
    world_point = bounce.get("world_point")
    if not world_point:
        return False
    return abs(world_point[1] - COURT_NET_Y_FT) <= NET_LINE_CONTACT_BAND_FT


def bounce_after_serve_contact(point_bounces, contact_frame, used_bounce_ids, fps):
    skipped_net_line = []
    for bounce in point_bounces:
        if bounce["frame"] <= contact_frame or bounce["id"] in used_bounce_ids:
            continue
        lag = bounce["frame"] - contact_frame
        if lag > frame_window(MAX_SERVE_FLIGHT_SECONDS, fps):
            return None, skipped_net_line
        if lag < frame_window(MIN_SERVE_FLIGHT_SECONDS, fps):
            continue
        if bounce_on_net_line(bounce):
            skipped_net_line.append(bounce["id"])
            continue
        return bounce, skipped_net_line
    return None, skipped_net_line


def load_court_calibration(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    points = data.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None
    try:
        court_points = order_court_corners([(float(x), float(y)) for x, y in points])
    except Exception:
        return None
    return {"points": court_points}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Infer point-like timeline hypotheses from a tracking log. This does "
            "not replace a manifest; it emits uncertain candidates with reasons."
        )
    )
    parser.add_argument("--jsonl", required=True, help="Input track_ball_yolo JSONL.")
    parser.add_argument("--court-calib-file", required=True, help="Court calibration JSON.")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--out", help="Optional output JSON path.")
    parser.add_argument(
        "--manifest",
        help="Optional manifest whose point_frames are used only for evaluation.",
    )
    parser.add_argument(
        "--expected-contact-frames",
        default="",
        help="Optional comma-separated verified serve-contact frames for contact-level evaluation.",
    )
    parser.add_argument(
        "--contact-tolerance-frames",
        type=int,
        default=3,
        help="Tolerance for --expected-contact-frames evaluation.",
    )
    parser.add_argument(
        "--activity-gap-seconds",
        type=float,
        default=2.0,
        help="Ball-observation gap that splits local activity spans.",
    )
    parser.add_argument(
        "--span-pad-seconds",
        type=float,
        default=1.5,
        help="Pad activity spans before running serve-motion detection.",
    )
    parser.add_argument(
        "--scan-window-seconds",
        type=float,
        default=15.0,
        help="Sliding player-motion window used as a backstop when ball activity is missing.",
    )
    parser.add_argument(
        "--scan-step-seconds",
        type=float,
        default=5.0,
        help="Step for the sliding serve-motion backstop.",
    )
    parser.add_argument(
        "--single-server",
        action="store_true",
        help="Treat the window as one game and suppress motions from the opposite server side.",
    )
    return parser.parse_args()


def ball_center(row):
    ball = row.get("ball") if row else None
    center = ball.get("center") if ball else None
    return center if center and len(center) == 2 else None


def ball_frame_stats(rows):
    frames = []
    coasted = 0
    repeats = 0
    previous_center = None
    for row in rows:
        center = ball_center(row)
        if not center:
            previous_center = None
            continue
        frames.append(int(row["frame"]))
        ball = row["ball"]
        if ball.get("interpolated") or ball.get("motion_gate") == "coast":
            coasted += 1
        if previous_center == center:
            repeats += 1
        previous_center = center
    return {
        "frames_with_ball": len(frames),
        "coasted_or_interpolated": coasted,
        "exact_repeats": repeats,
        "distinct_real_observations": max(0, len(frames) - repeats),
    }


def activity_spans(rows, fps, gap_seconds):
    gap_frames = frame_window(gap_seconds, fps)
    frames = [int(row["frame"]) for row in rows if ball_center(row)]
    if not frames:
        return []
    spans = []
    start = frames[0]
    previous = frames[0]
    for frame in frames[1:]:
        if frame - previous > gap_frames:
            spans.append({"start_frame": start, "end_frame": previous})
            start = frame
        previous = frame
    spans.append({"start_frame": start, "end_frame": previous})
    return spans


def expand_span(span, first_frame, last_frame, fps, seconds):
    pad = frame_window(seconds, fps)
    return {
        "start_frame": max(first_frame, span["start_frame"] - pad),
        "end_frame": min(last_frame, span["end_frame"] + pad),
    }


def collect_serve_motions(
    by_frame,
    spans,
    first_frame,
    last_frame,
    inv_homography,
    fps,
    pad_seconds,
    scan_window_seconds,
    scan_step_seconds,
):
    motions = []

    def scan_window(start_frame, end_frame, source_window, span_index=None):
        detection_end_frame = min(
            last_frame,
            end_frame + frame_window(REACH_SEARCH_SECONDS, fps),
        )
        for side in ("near", "far"):
            for motion in detect_serve_motions_for_point(
                by_frame,
                start_frame,
                detection_end_frame,
                side,
                inv_homography,
                fps,
            ):
                contact_frame = int(motion["contact_frame"])
                if contact_frame < start_frame or contact_frame > end_frame:
                    continue
                item = dict(motion)
                item["server"] = side
                item["receiver"] = side_opponent(side)
                item["activity_span_index"] = span_index
                item["window_start_frame"] = start_frame
                item["window_end_frame"] = end_frame
                item["detection_window_end_frame"] = detection_end_frame
                item["timeline_window_source"] = source_window
                motions.append(item)

    for span_index, span in enumerate(spans, start=1):
        window = expand_span(span, first_frame, last_frame, fps, pad_seconds)
        scan_window(window["start_frame"], window["end_frame"], "activity_span", span_index)

    scan_window_frames = frame_window(scan_window_seconds, fps)
    scan_step_frames = max(1, frame_window(scan_step_seconds, fps))
    frame = first_frame
    while frame <= last_frame:
        scan_window(
            frame,
            min(last_frame, frame + scan_window_frames - 1),
            "sliding_window",
        )
        frame += scan_step_frames
    return dedupe_motions(motions, fps)


def motion_rank(motion):
    source_score = {"ball_toss": 2, "peak_reach": 1}.get(motion.get("source"), 0)
    confidence_score = {"high": 3, "medium": 2, "low": 1}.get(motion.get("confidence"), 0)
    return (confidence_score, source_score, int(motion["contact_frame"]))


def server_vote_weight(attempt):
    confidence_score = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(
        attempt.get("confidence"),
        0.0,
    )
    source_score = {"ball_toss": 1.0, "peak_reach": 0.4}.get(attempt.get("source"), 0.0)
    landing = attempt.get("landing") or {}
    landing_score = 1.0 if landing.get("bounce_id") else 0.0
    if landing.get("result") == "in":
        landing_score += 0.5
    return confidence_score + source_score + landing_score


def resolve_single_server_vote(points):
    votes = {"near": 0.0, "far": 0.0}
    counts = {"near": 0, "far": 0}
    for point in points:
        first = point["attempts"][0]
        server = first.get("server")
        if server not in votes:
            continue
        votes[server] += server_vote_weight(first)
        counts[server] += 1
    if not any(counts.values()):
        return None, {"votes": votes, "counts": counts, "margin": 0.0}
    winner = max(votes, key=lambda side: votes[side])
    loser = "near" if winner == "far" else "far"
    margin = round(votes[winner] - votes[loser], 3)
    count_margin = counts[winner] - counts[loser]
    contested = (
        counts[loser] > 0
        and (
            margin < SINGLE_SERVER_MIN_VOTE_MARGIN
            or count_margin < SINGLE_SERVER_MIN_COUNT_MARGIN
        )
    )
    resolved = None if contested else winner
    return resolved, {
        "votes": {side: round(value, 3) for side, value in votes.items()},
        "counts": counts,
        "margin": margin,
        "count_margin": count_margin,
        "min_margin": SINGLE_SERVER_MIN_VOTE_MARGIN,
        "min_count_margin": SINGLE_SERVER_MIN_COUNT_MARGIN,
        "contested": contested,
    }


def dedupe_motions(motions, fps):
    kept = []
    window = frame_window(0.75, fps)
    for motion in sorted(motions, key=lambda item: int(item["contact_frame"])):
        clash_index = None
        for index, kept_motion in enumerate(kept):
            if abs(int(motion["contact_frame"]) - int(kept_motion["contact_frame"])) <= window:
                clash_index = index
                break
        if clash_index is None:
            kept.append(motion)
        elif motion_rank(motion) > motion_rank(kept[clash_index]):
            kept[clash_index] = motion
    return sorted(kept, key=lambda item: int(item["contact_frame"]))


def detector_bounces(rows, court_calib_file):
    from bounce_detect import detect_bounces

    with open(court_calib_file, "r", encoding="utf-8") as handle:
        calib_points = json.load(handle)["points"]
    bounces = detect_bounces(rows, calib_points)
    for index, bounce in enumerate(bounces, start=1):
        bounce["id"] = f"bounce_{index:03d}"
        bounce["frame"] = int(bounce["frame"])
    return bounces


def serve_landing_after_contact(bounces, contact_frame, server, fps):
    bounce, skipped_net_line_ids = bounce_after_serve_contact(bounces, contact_frame, set(), fps)
    if bounce:
        lag = int(bounce["frame"]) - int(contact_frame)
        world_point = bounce.get("world_point")
        receiver = side_opponent(server)
        result = "unknown"
        reason = "serve_bounce_not_adjudicable"
        if world_point_in_receiver_service_box(world_point, receiver):
            result = "in"
            reason = "receiver_service_box"
        elif world_point:
            result = "fault"
            reason = "outside_receiver_service_box"
        return {
            "bounce_id": bounce.get("id"),
            "bounce_frame": bounce["frame"],
            "world_point": world_point,
            "result": result,
            "reason": reason,
            "lag_frames": lag,
            "skipped_net_line_bounce_ids": skipped_net_line_ids,
        }
    if skipped_net_line_ids:
        return {
            "bounce_id": None,
            "bounce_frame": None,
            "world_point": None,
            "result": "unknown",
            "reason": "serve_bounce_net_line_contact",
            "lag_frames": None,
            "skipped_net_line_bounce_ids": skipped_net_line_ids,
        }
    return {
        "bounce_id": None,
        "bounce_frame": None,
        "world_point": None,
        "result": "unknown",
        "reason": "serve_bounce_not_detected",
        "lag_frames": None,
        "skipped_net_line_bounce_ids": [],
    }


def isolation_before(ball_frames, contact_frame, fps):
    previous = [frame for frame in ball_frames if frame < contact_frame]
    if not previous:
        return None
    gap = contact_frame - previous[-1]
    return {
        "dead_frames_before": gap,
        "isolated_by_deadtime": gap >= frame_window(1.5, fps),
    }


def local_fragmentation(spans, contact_frame, fps):
    radius = frame_window(20.0, fps)
    nearby = [
        span
        for span in spans
        if span["start_frame"] <= contact_frame + radius and span["end_frame"] >= contact_frame - radius
    ]
    return {
        "nearby_activity_spans": len(nearby),
        "spans_per_minute": round(len(nearby) / ((2 * radius + 1) / fps / 60.0), 2),
    }


def confidence_for_point(attempts, isolation, fragmentation):
    score = 0.35
    reasons = []
    first = attempts[0]
    if first["source"] == "ball_toss":
        score += 0.18
        reasons.append("serve_has_tracked_toss")
    else:
        reasons.append("serve_from_reach_fallback")
    if first["landing"]["bounce_id"]:
        score += 0.2
        reasons.append("serve_corroborated_by_landing")
    else:
        reasons.append(first["landing"]["reason"])
    if first["landing"]["result"] == "in":
        score += 0.08
        reasons.append("landing_in_box")
    if isolation and isolation.get("isolated_by_deadtime"):
        score += 0.12
        reasons.append("isolated_by_deadtime")
    else:
        reasons.append("not_isolated_by_deadtime")
    if fragmentation["nearby_activity_spans"] >= 5:
        score -= 0.15
        reasons.append("high_local_fragmentation")
    elif fragmentation["nearby_activity_spans"] >= 3:
        score -= 0.07
        reasons.append("moderate_local_fragmentation")
    if len(attempts) == 2:
        reasons.append("second_serve_grouped_after_fault_or_unknown")
    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.75 else "uncertain"
    return round(score, 3), label, reasons


def boundary_status(isolation):
    if isolation and isolation.get("isolated_by_deadtime"):
        return "point_start_hypothesis_deadtime_isolated"
    return "point_start_hypothesis_no_deadtime_evidence"


def second_serve_grouping_evidence(
    by_frame,
    inv_homography,
    previous_attempt,
    attempt,
    fps,
):
    fraction, tracked = ball_return_fraction(
        by_frame,
        previous_attempt["contact_frame"],
        attempt["contact_frame"],
        previous_attempt["server"],
        inv_homography,
        fps,
    )
    min_tracked = frame_window(
        max(RALLY_MIN_TRACKED_SECONDS, SECOND_SERVE_MIN_TRACKED_SECONDS),
        fps,
    )
    enough_track = tracked >= min_tracked
    return {
        "gap_frames": attempt["contact_frame"] - previous_attempt["contact_frame"],
        "ball_return_fraction": round(fraction, 3) if fraction is not None else None,
        "ball_tracked_frames": tracked,
        "min_ball_tracked_frames": min_tracked,
        "min_ball_tracked_seconds": max(
            RALLY_MIN_TRACKED_SECONDS,
            SECOND_SERVE_MIN_TRACKED_SECONDS,
        ),
        "enough_ball_track": enough_track,
        "rally_return_fraction_threshold": RALLY_BALL_RETURN_FRACTION,
        "rally_between_serves": None if fraction is None else fraction >= RALLY_BALL_RETURN_FRACTION,
    }


def mark_suppressed_motion(attempt, previous, evidence, reason, fps):
    attempt.setdefault("review_reasons", []).append(reason)
    if evidence["gap_frames"] >= frame_window(LONG_SUPPRESSION_REVIEW_SECONDS, fps):
        attempt["review_reasons"].append("suppression_may_hide_point_boundary")
    attempt["previous_motion_evidence"] = evidence
    previous.setdefault("suppressed_rally_motions", []).append(attempt)


def build_hypotheses(
    rows,
    by_frame,
    court_calib_file,
    fps,
    activity_gap_seconds,
    span_pad_seconds,
    scan_window_seconds,
    scan_step_seconds,
    single_server=False,
):
    if not rows:
        return {"hypotheses": [], "activity_spans": [], "summary": {}}
    first_frame = int(rows[0]["frame"])
    last_frame = int(rows[-1]["frame"])
    inv_homography = build_inverse_court_homography(load_court_calibration(court_calib_file))
    spans = activity_spans(rows, fps, activity_gap_seconds)
    ball_frames = [int(row["frame"]) for row in rows if ball_center(row)]
    motions = collect_serve_motions(
        by_frame,
        spans,
        first_frame,
        last_frame,
        inv_homography,
        fps,
        span_pad_seconds,
        scan_window_seconds,
        scan_step_seconds,
    )
    bounces = detector_bounces(rows, court_calib_file)

    raw_points = []
    for motion in motions:
        landing = serve_landing_after_contact(bounces, motion["contact_frame"], motion["server"], fps)
        attempt = {
            "attempt": 1,
            "contact_frame": int(motion["contact_frame"]),
            "server": motion["server"],
            "receiver": motion["receiver"],
            "source": motion.get("source"),
            "confidence": motion.get("confidence"),
            "reasons": motion.get("reasons", []),
            "landing": landing,
        }
        raw_points.append({"attempts": [attempt]})

    grouped = []
    max_second_gap = frame_window(MAX_SECOND_SERVE_GAP_SECONDS, fps)
    max_rally_continuation_gap = frame_window(RALLY_CONTINUATION_SUPPRESS_SECONDS, fps)
    for point in raw_points:
        attempt = point["attempts"][0]
        if grouped:
            previous = grouped[-1]
            previous_attempt = previous["attempts"][-1]
            gap = attempt["contact_frame"] - previous_attempt["contact_frame"]
            same_server = attempt["server"] == previous_attempt["server"]
            can_have_second = len(previous["attempts"]) == 1 and previous_attempt["landing"]["result"] != "in"
            second_serve_evidence = second_serve_grouping_evidence(
                by_frame,
                inv_homography,
                previous_attempt,
                attempt,
                fps,
            )
            enough_track = second_serve_evidence["enough_ball_track"]
            rally_between = second_serve_evidence["rally_between_serves"]
            has_tracked_toss = attempt.get("source") == "ball_toss"
            previous_can_still_be_rally = (
                len(previous["attempts"]) == 1
                and previous_attempt["landing"]["result"] in {"in", "unknown"}
            )
            weak_reach_inside_previous_point = (
                previous_can_still_be_rally
                and attempt.get("source") == "peak_reach"
                and gap <= frame_window(WEAK_REACH_CONTINUATION_SUPPRESS_SECONDS, fps)
                and not enough_track
            )
            if weak_reach_inside_previous_point:
                mark_suppressed_motion(
                    attempt,
                    previous,
                    second_serve_evidence,
                    "suppressed_weak_reach_motion_inside_previous_point",
                    fps,
                )
                continue
            if (
                previous_can_still_be_rally
                and gap <= max_rally_continuation_gap
                and enough_track
                and rally_between is True
            ):
                mark_suppressed_motion(
                    attempt,
                    previous,
                    second_serve_evidence,
                    "suppressed_rally_motion_not_point_start",
                    fps,
                )
                continue
            if (
                same_server
                and can_have_second
                and gap <= max_second_gap
                and enough_track
                and has_tracked_toss
                and rally_between is False
            ):
                attempt["attempt"] = 2
                attempt["second_serve_evidence"] = second_serve_evidence
                attempt["ball_return_fraction_since_previous"] = second_serve_evidence[
                    "ball_return_fraction"
                ]
                attempt["ball_tracked_frames_since_previous"] = second_serve_evidence[
                    "ball_tracked_frames"
                ]
                if second_serve_evidence["ball_return_fraction"] >= max(
                    0.0,
                    RALLY_BALL_RETURN_FRACTION - 0.05,
                ):
                    attempt.setdefault("review_reasons", []).append(
                        "second_serve_grouping_contested"
                    )
                previous["attempts"].append(attempt)
                continue
            if same_server and can_have_second and gap <= max_second_gap and not enough_track:
                attempt.setdefault("review_reasons", []).append(
                    "second_serve_grouping_insufficient_ball_track"
                )
            if same_server and can_have_second and gap <= max_second_gap and rally_between is True:
                attempt.setdefault("review_reasons", []).append(
                    "second_serve_grouping_rally_between_serves"
                )
            if same_server and can_have_second and gap <= max_second_gap and not has_tracked_toss:
                attempt.setdefault("review_reasons", []).append(
                    "second_serve_grouping_requires_tracked_toss"
                )
            attempt["previous_motion_evidence"] = second_serve_evidence
        grouped.append(point)

    resolved_single_server = None
    single_server_vote = None
    if single_server:
        resolved_single_server, single_server_vote = resolve_single_server_vote(grouped)
        if resolved_single_server is None and single_server_vote and single_server_vote["contested"]:
            for point in grouped:
                point["attempts"][0].setdefault("review_reasons", []).append(
                    "single_server_vote_contested"
                )
        filtered = []
        previous_kept = None
        for point in grouped:
            attempt = point["attempts"][0]
            if resolved_single_server and attempt["server"] != resolved_single_server:
                if previous_kept:
                    previous_attempt = previous_kept["attempts"][-1]
                    evidence = second_serve_grouping_evidence(
                        by_frame,
                        inv_homography,
                        previous_attempt,
                        attempt,
                        fps,
                    )
                    mark_suppressed_motion(
                        attempt,
                        previous_kept,
                        evidence,
                        "suppressed_opposite_server_in_single_game",
                        fps,
                    )
                continue
            filtered.append(point)
            previous_kept = point
        grouped = filtered

    hypotheses = []
    for index, point in enumerate(grouped, start=1):
        first_attempt = point["attempts"][0]
        contact_frame = first_attempt["contact_frame"]
        next_contact = grouped[index]["attempts"][0]["contact_frame"] if index < len(grouped) else None
        start_frame = max(first_frame, contact_frame - frame_window(1.5, fps))
        if next_contact is None:
            end_frame = last_frame
        else:
            observed = [frame for frame in ball_frames if contact_frame <= frame < next_contact]
            end_frame = observed[-1] if observed else next_contact - 1
        isolation = isolation_before(ball_frames, contact_frame, fps)
        fragmentation = local_fragmentation(spans, contact_frame, fps)
        score, confidence, reasons = confidence_for_point(point["attempts"], isolation, fragmentation)
        status = boundary_status(isolation)
        review_reasons = sorted(
            {
                reason
                for attempt in (
                    point["attempts"] + point.get("suppressed_rally_motions", [])
                )
                for reason in attempt.get("review_reasons", [])
            }
        )
        hypotheses.append(
            {
                "id": f"point_hypothesis_{index:03d}",
                "display_id": f"serve_motion_hypothesis_{index:03d}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_source": "serve_contact_minus_1.5s",
                "end_source": "last_ball_activity_before_next_hypothesis" if next_contact else "end_of_log",
                "confidence_score": score,
                "confidence": confidence,
                "boundary_status": status,
                "reasons": reasons,
                "review_reasons": review_reasons,
                "ends_have_no_truth": True,
                "serve_count": len(point["attempts"]),
                "suppressed_rally_motion_count": len(point.get("suppressed_rally_motions", [])),
                "serve_corroborated": any(attempt["landing"]["bounce_id"] for attempt in point["attempts"]),
                "landing_in_box": any(attempt["landing"]["result"] == "in" for attempt in point["attempts"]),
                "isolation": isolation,
                "local_fragmentation": fragmentation,
                "attempts": point["attempts"],
                "suppressed_rally_motions": point.get("suppressed_rally_motions", []),
            }
        )

    stats = ball_frame_stats(rows)
    total_frames = max(1, last_frame - first_frame + 1)
    stats["total_frames"] = total_frames
    stats["frames_with_ball_pct"] = round(100.0 * stats["frames_with_ball"] / total_frames, 1)
    stats["distinct_real_observations_pct"] = round(
        100.0 * stats["distinct_real_observations"] / total_frames, 1
    )
    return {
        "source_jsonl": None,
        "court_calib_file": court_calib_file,
        "fps": fps,
        "frame_range": {"start_frame": first_frame, "end_frame": last_frame},
        "summary": {
            **stats,
            "activity_spans": len(spans),
            "serve_motions": len(motions),
            "suppressed_rally_motions": sum(
                len(item.get("suppressed_rally_motions", [])) for item in hypotheses
            ),
            "point_hypotheses": len(hypotheses),
            "isolated_point_start_candidates": sum(
                1
                for item in hypotheses
                if item.get("boundary_status") == "point_start_hypothesis_deadtime_isolated"
            ),
            "high_confidence_hypotheses": sum(1 for item in hypotheses if item["confidence"] == "high"),
            "uncertain_hypotheses": sum(1 for item in hypotheses if item["confidence"] == "uncertain"),
            "confidence_caveat": (
                "Hypothesis confidence is clip-relative and experimental. "
                "It is not a scoring contract; serve_count and point boundaries "
                "remain hypotheses until independently verified."
            ),
            "single_server": bool(single_server),
            "resolved_single_server": resolved_single_server,
            "single_server_vote": single_server_vote,
        },
        "activity_spans": spans,
        "serve_motions": motions,
        "hypotheses": hypotheses,
    }


def overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def evaluate_against_manifest(hypotheses, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    truth = parse_point_ranges(",".join(
        f"{item['start_frame']}-{item['end_frame']}" for item in manifest.get("point_frames", [])
    ))
    if not truth:
        return None
    matched_truth = set()
    ious = []
    for hypothesis in hypotheses:
        best = None
        for t_index, point_range in enumerate(truth):
            shared = overlap(
                hypothesis["start_frame"],
                hypothesis["end_frame"],
                point_range["start_frame"],
                point_range["end_frame"],
            )
            if not shared:
                continue
            union = (
                max(hypothesis["end_frame"], point_range["end_frame"])
                - min(hypothesis["start_frame"], point_range["start_frame"])
                + 1
            )
            iou = shared / float(union)
            if best is None or iou > best[1]:
                best = (t_index, iou)
        if best:
            matched_truth.add(best[0])
            ious.append(best[1])
    return {
        "manifest": manifest_path,
        "truth_points": len(truth),
        "truth_points_with_any_overlap": len(matched_truth),
        "hypotheses": len(hypotheses),
        "mean_iou_agreement_not_accuracy": round(sum(ious) / len(ious), 3) if ious else 0.0,
        "caveat": (
            "Manifest ranges tile the clip and are not independent point-end ground truth; "
            "truth overlap and IoU are agreement diagnostics, not precision or recall."
        ),
    }


def parse_contact_frames(raw):
    if not raw:
        return []
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def evaluate_contacts(motions, expected_frames, tolerance):
    if not expected_frames:
        return None
    matched_expected = set()
    matched_motions = set()
    for motion_index, motion in enumerate(motions):
        frame = int(motion["contact_frame"])
        for expected_index, expected in enumerate(expected_frames):
            if abs(frame - expected) <= tolerance:
                matched_expected.add(expected_index)
                matched_motions.add(motion_index)
                break
    return {
        "expected_contacts": len(expected_frames),
        "matched_expected_contacts": len(matched_expected),
        "detected_contacts": len(motions),
        "matched_detected_contacts": len(matched_motions),
        "contact_recall": round(len(matched_expected) / float(len(expected_frames)), 3),
        "contact_precision": round(len(matched_motions) / float(max(1, len(motions))), 3),
        "tolerance_frames": tolerance,
    }


def print_summary(result):
    summary = result["summary"]
    print(
        "timeline hypotheses: {points} candidates, {motions} serve motions, "
        "{spans} activity spans".format(
            points=summary["point_hypotheses"],
            motions=summary["serve_motions"],
            spans=summary["activity_spans"],
        )
    )
    print(
        "ball observations: {with_ball}/{total} ({with_pct}%), distinct real "
        "{distinct}/{total} ({distinct_pct}%)".format(
            with_ball=summary["frames_with_ball"],
            total=summary["total_frames"],
            with_pct=summary["frames_with_ball_pct"],
            distinct=summary["distinct_real_observations"],
            distinct_pct=summary["distinct_real_observations_pct"],
        )
    )
    print("confidence caveat: {caveat}".format(caveat=summary["confidence_caveat"]))
    for hypothesis in result["hypotheses"]:
        first = hypothesis["attempts"][0]
        print(
            "{id} f{start}-{end} {conf} score={score:.3f} "
            "server={server} contact=f{contact} serves={serve_count} "
            "landing={landing}".format(
                id=hypothesis["id"],
                start=hypothesis["start_frame"],
                end=hypothesis["end_frame"],
                conf=hypothesis["confidence"],
                score=hypothesis["confidence_score"],
                server=first["server"],
                contact=first["contact_frame"],
                serve_count=hypothesis["serve_count"],
                landing=first["landing"]["reason"],
            )
        )
    evaluation = result.get("evaluation")
    if evaluation:
        print(
            "manifest agreement: truth_overlap={truth}/{truth_total} "
            "mean_iou={iou} ({caveat})".format(
                truth=evaluation["truth_points_with_any_overlap"],
                truth_total=evaluation["truth_points"],
                iou=evaluation["mean_iou_agreement_not_accuracy"],
                caveat=evaluation["caveat"],
            )
        )
    contact_evaluation = result.get("contact_evaluation")
    if contact_evaluation:
        print(
            "contact agreement: recall={recall} precision={precision} "
            "matched={matched}/{expected} detected={detected} tolerance=+/-{tol}f".format(
                recall=contact_evaluation["contact_recall"],
                precision=contact_evaluation["contact_precision"],
                matched=contact_evaluation["matched_expected_contacts"],
                expected=contact_evaluation["expected_contacts"],
                detected=contact_evaluation["detected_contacts"],
                tol=contact_evaluation["tolerance_frames"],
            )
        )


def main():
    args = parse_args()
    rows, by_frame = read_tracking_log(args.jsonl)
    result = build_hypotheses(
        rows,
        by_frame,
        args.court_calib_file,
        args.fps,
        args.activity_gap_seconds,
        args.span_pad_seconds,
        args.scan_window_seconds,
        args.scan_step_seconds,
        single_server=args.single_server,
    )
    result["source_jsonl"] = args.jsonl
    if args.manifest:
        result["evaluation"] = evaluate_against_manifest(result["hypotheses"], args.manifest)
    expected_contacts = parse_contact_frames(args.expected_contact_frames)
    if expected_contacts:
        result["contact_evaluation"] = evaluate_contacts(
            result["serve_motions"],
            expected_contacts,
            args.contact_tolerance_frames,
        )
    print_summary(result)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
