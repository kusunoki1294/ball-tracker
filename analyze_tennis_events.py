import argparse
import json
import math
import os
from bisect import bisect_left, bisect_right

from track_ball_yolo import (
    ball_contact_point,
    build_inverse_court_homography,
    load_court_calibration,
    point_to_bbox_distance,
    project_to_court_world,
)


FPS_DEFAULT = 30.0
COURT_NET_Y_FT = 39.0
COURT_WIDTH_FT = 36.0
SINGLES_LEFT_FT = 4.5
SINGLES_RIGHT_FT = 31.5
SERVICE_LEFT_FT = 18.0
FAR_SERVICE_Y_MIN_FT = 18.0
FAR_SERVICE_Y_MAX_FT = 39.0
NEAR_SERVICE_Y_MIN_FT = 39.0
NEAR_SERVICE_Y_MAX_FT = 60.0
FT_PER_SEC_TO_MPH = 0.681818
FT_PER_SEC_TO_KMH = 1.09728


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process tennis tracking logs into shots, bounces, and score.")
    parser.add_argument("--jsonl", required=True, help="Input tracking JSONL from track_ball_yolo.py.")
    parser.add_argument("--output", required=True, help="Output analysis JSON path.")
    parser.add_argument(
        "--court-calib-file",
        default="yoloVids/calibration/court_calib_tennis7.json",
        help="Court calibration used by the tracking run.",
    )
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT, help="Video frames per second.")
    parser.add_argument(
        "--point-frames",
        default="",
        help="Comma-separated inclusive point frame ranges, e.g. 1-1152,1153-1545.",
    )
    parser.add_argument(
        "--point-winners",
        default="",
        help="Comma-separated point winners, using near/far. Required for score output.",
    )
    parser.add_argument("--server", choices=["near", "far"], default="near", help="Initial server side.")
    parser.add_argument(
        "--receiver",
        choices=["near", "far"],
        default="far",
        help="Initial receiver side.",
    )
    parser.add_argument(
        "--near-handedness",
        choices=["right", "left"],
        default="right",
        help="Near player handedness for forehand/backhand inference.",
    )
    parser.add_argument(
        "--far-handedness",
        choices=["right", "left"],
        default="right",
        help="Far player handedness for forehand/backhand inference.",
    )
    parser.add_argument(
        "--max-shot-search-frames",
        type=int,
        default=150,
        help="Maximum frames to search backward from a bounce for the causing shot.",
    )
    parser.add_argument(
        "--auto-score",
        action="store_true",
        help="Use inferred serve/rally outcomes when possible; manual point winners remain fallback.",
    )
    parser.add_argument(
        "--ignore-bounces-before-frame",
        type=int,
        default=0,
        help="Ignore bounce events before this frame, useful for warm-up/non-point balls at clip start.",
    )
    parser.add_argument(
        "--official-double-fault-points",
        default="",
        help="Comma-separated point indexes that should be treated as official double faults.",
    )
    parser.add_argument(
        "--exclude-bounce-frames",
        default="",
        help="Comma-separated bounce frames/ranges to exclude from live point analysis, e.g. 1243,1303-1401.",
    )
    parser.add_argument(
        "--initial-game-score",
        default="0-0",
        help="Starting games in the current set, in server-receiver order.",
    )
    parser.add_argument(
        "--initial-set-score",
        default="0-0",
        help="Starting sets, in server-receiver order.",
    )
    return parser.parse_args()


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


def parse_frame_ranges(raw):
    return parse_point_ranges(raw)


def frame_in_ranges(frame, ranges):
    return any(item["start_frame"] <= frame <= item["end_frame"] for item in ranges)


def parse_sides(raw):
    if not raw:
        return []
    sides = []
    for token in raw.split(","):
        side = token.strip().lower()
        if side not in {"near", "far"}:
            raise ValueError(f"Invalid side in --point-winners: {token}")
        sides.append(side)
    return sides


def parse_score_pair(raw, label):
    if not raw or "-" not in raw:
        raise ValueError(f"{label} must be formatted like 0-0")
    left, right = raw.split("-", 1)
    return {"server": int(left), "receiver": int(right)}


def parse_indexes(raw):
    if not raw:
        return set()
    return {int(token.strip()) for token in raw.split(",") if token.strip()}


def ball_world_point(row, inv_homography):
    ball = row.get("ball")
    if not ball:
        return None
    point = ball_contact_point(ball) or ball.get("center")
    return project_to_court_world(point, inv_homography)


def player_center(player):
    if not player:
        return None
    x1, y1, x2, y2 = player["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def expanded_strike_zone(player, side):
    if not player:
        return None
    x1, y1, x2, y2 = player["bbox"]
    height = max(1.0, y2 - y1)
    width = max(1.0, x2 - x1)
    if side == "near":
        return [
            x1 - max(36.0, width * 0.55),
            y1 - max(70.0, height * 0.18),
            x2 + max(42.0, width * 0.65),
            y1 + height * 0.62,
        ]
    return [
        x1 - max(22.0, width * 0.85),
        y1 - max(32.0, height * 0.28),
        x2 + max(22.0, width * 0.85),
        y2 + max(24.0, height * 0.28),
    ]


def side_for_world(world_point):
    if world_point is None:
        return None
    return "far" if world_point[1] < COURT_NET_Y_FT else "near"


def side_opponent(side):
    return "near" if side == "far" else "far"


def get_player(row, side):
    return row.get("player_near") if side == "near" else row.get("player_far")


def nearest_racket(row, point):
    best = None
    for det in row.get("scene") or []:
        if det.get("class_name") != "tennis racket":
            continue
        distance = point_to_bbox_distance(point, det["bbox"])
        if best is None or distance < best["distance"]:
            best = {"distance": distance, "detection": det}
    return best


def racket_distance(row, point):
    nearest = nearest_racket(row, point)
    return nearest["distance"] if nearest else None


def world_point_in_bounds(world_point, margin=6.0):
    if world_point is None:
        return False
    xw, yw = world_point
    return -margin <= xw <= 36.0 + margin and -margin <= yw <= 78.0 + margin


def world_point_on_player_side(world_point, side, margin=8.0):
    if world_point is None:
        return False
    _, yw = world_point
    if side == "near":
        return yw >= COURT_NET_Y_FT - margin
    return yw <= COURT_NET_Y_FT + margin


def find_shot_for_bounce(rows, frame_numbers, bounce, previous_bounce_frame, point_start_frame, args, inv_homography):
    bounce_frame = bounce["frame"]
    hitter = side_opponent(bounce["side"])
    search_start = max(
        previous_bounce_frame + 1 if previous_bounce_frame else 1,
        point_start_frame or 1,
        bounce_frame - args.max_shot_search_frames,
    )
    search_end = max(search_start, bounce_frame - 3)
    left = bisect_left(frame_numbers, search_start)
    right = bisect_right(frame_numbers, search_end)

    best = None
    for row in rows[left:right]:
        ball = row.get("ball")
        if not ball:
            continue
        point = ball_contact_point(ball) or ball.get("center")
        if point is None:
            continue
        player = get_player(row, hitter)
        zone = expanded_strike_zone(player, hitter)
        if zone is None:
            continue
        player_distance = point_to_bbox_distance(point, zone)
        nearest = nearest_racket(row, point)
        racket_dist = nearest["distance"] if nearest else None
        world_point = project_to_court_world(point, inv_homography)
        row_side = side_for_world(world_point)
        side_bonus = 0.0 if row_side in {None, hitter} else 18.0
        if world_point is None:
            world_bonus = 15.0
        elif not world_point_in_bounds(world_point, margin=10.0):
            world_bonus = 32.0
        elif not world_point_on_player_side(world_point, hitter, margin=10.0):
            world_bonus = 18.0
        else:
            world_bonus = 0.0
        racket_bonus = min(racket_dist if racket_dist is not None else 120.0, 120.0) * 0.75
        recency = (bounce_frame - row["frame"]) * 0.04
        score = player_distance + side_bonus + world_bonus + racket_bonus + recency
        quality = "medium"
        if player_distance > 45.0 or not world_point_in_bounds(world_point, margin=10.0):
            quality = "low"
        if racket_dist is not None and racket_dist <= 55.0 and player_distance <= 35.0:
            quality = "high"
        candidate = {
            "score": score,
            "frame": row["frame"],
            "point": [round(float(point[0]), 1), round(float(point[1]), 1)],
            "world_point": [round(world_point[0], 2), round(world_point[1], 2)] if world_point else None,
            "player": hitter,
            "player_distance_px": round(player_distance, 1),
            "racket_distance_px": round(racket_dist, 1) if racket_dist is not None else None,
            "quality": quality,
            "racket_bbox": nearest["detection"]["bbox"] if nearest else None,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate

    if best is None:
        fallback_frame = search_end
        row = rows[bisect_left(frame_numbers, fallback_frame)] if rows else None
        best = {
            "frame": fallback_frame,
            "point": None,
            "world_point": None,
            "player": hitter,
            "player_distance_px": None,
            "racket_distance_px": None,
            "quality": "missing",
            "score": None,
        }
        if row and row.get("ball"):
            point = ball_contact_point(row["ball"]) or row["ball"].get("center")
            world_point = project_to_court_world(point, inv_homography)
            best["point"] = [round(float(point[0]), 1), round(float(point[1]), 1)]
            best["world_point"] = [round(world_point[0], 2), round(world_point[1], 2)] if world_point else None

    return best


def infer_stroke_side(shot, row, handedness):
    if shot.get("point") is None:
        return "unknown"
    if shot.get("quality") not in {"high", "medium"}:
        return "unknown"
    debug = shot.get("debug") or shot
    racket_distance_px = debug.get("racket_distance_px")
    if racket_distance_px is None or racket_distance_px > 80.0:
        world_point = shot.get("world_point")
        if not world_point_in_bounds(world_point, margin=8.0) or not world_point_on_player_side(
            world_point, shot["player"], margin=10.0
        ):
            return "unknown"
    player = get_player(row, shot["player"]) if row else None
    center = player_center(player)
    if center is None:
        return "unknown"
    contact_x = shot["point"][0]
    racket_bbox = debug.get("racket_bbox")
    if racket_bbox is not None and racket_distance_px is not None and racket_distance_px <= 80.0:
        contact_x = (racket_bbox[0] + racket_bbox[2]) / 2.0
    player_x = center[0]
    if abs(contact_x - player_x) < 14.0:
        return "body"
    if shot["player"] == "near":
        right_side_contact = contact_x > player_x
    else:
        right_side_contact = contact_x < player_x
    if handedness == "left":
        right_side_contact = not right_side_contact
    return "forehand" if right_side_contact else "backhand"


def estimate_bounce_interval_speed(previous_bounce, bounce, fps):
    if not previous_bounce or not previous_bounce.get("world_point") or not bounce.get("world_point"):
        return {"mph": None, "kmh": None, "quality": "missing", "source": "missing"}
    frame_delta = max(1, bounce["frame"] - previous_bounce["frame"])
    seconds = frame_delta / fps
    p0 = previous_bounce["world_point"]
    p1 = bounce["world_point"]
    distance_ft = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    ft_per_sec = distance_ft / seconds
    return {
        "mph": round(ft_per_sec * FT_PER_SEC_TO_MPH, 1),
        "kmh": round(ft_per_sec * FT_PER_SEC_TO_KMH, 1),
        "flat_distance_ft": round(distance_ft, 1),
        "flight_frames": frame_delta,
        "source": "bounce_interval",
        "quality": "low",
    }


def estimate_speed(shot, bounce, previous_bounce, fps):
    shot_world = shot.get("world_point")
    bounce_world = bounce.get("world_point")
    if not bounce_world:
        return {"mph": None, "kmh": None, "quality": "missing"}
    speed_source = "contact_to_bounce"
    if (
        not shot_world
        or not world_point_in_bounds(shot_world, margin=8.0)
        or not world_point_on_player_side(shot_world, shot["player"], margin=10.0)
    ):
        fallback = estimate_bounce_interval_speed(previous_bounce, bounce, fps)
        fallback["contact_rejected"] = True
        return fallback

    frame_delta = max(1, bounce["frame"] - shot["frame"])
    seconds = frame_delta / fps
    distance_ft = math.hypot(bounce_world[0] - shot_world[0], bounce_world[1] - shot_world[1])
    ft_per_sec = distance_ft / seconds
    mph = ft_per_sec * FT_PER_SEC_TO_MPH
    if mph < 8.0 or mph > 90.0:
        fallback = estimate_bounce_interval_speed(previous_bounce, bounce, fps)
        fallback["contact_rejected"] = True
        fallback["rejected_contact_mph"] = round(mph, 1)
        return fallback
    elif shot.get("quality") in {"high", "medium"} and frame_delta <= 90:
        quality = "medium"
    else:
        quality = "low"
    return {
        "mph": round(mph, 1),
        "kmh": round(ft_per_sec * FT_PER_SEC_TO_KMH, 1),
        "flat_distance_ft": round(distance_ft, 1),
        "flight_frames": frame_delta,
        "source": speed_source,
        "quality": quality,
    }


def tennis_score_label(points_won):
    labels = ["0", "15", "30", "40"]
    return labels[points_won] if points_won < len(labels) else "40"


def format_score(score, server, receiver):
    server_points = score[server]
    receiver_points = score[receiver]
    if server_points >= 3 and receiver_points >= 3:
        if server_points == receiver_points:
            return "deuce"
        if server_points == receiver_points + 1:
            return "ad-in"
        if receiver_points == server_points + 1:
            return "ad-out"
    return f"{tennis_score_label(score[server])}-{tennis_score_label(score[receiver])}"


def format_side_score(score, server, receiver):
    return f"{score[server]}-{score[receiver]}"


def game_winner_for_points(score, winner):
    if winner is None:
        return None
    loser = side_opponent(winner)
    if score[winner] >= 4 and score[winner] - score[loser] >= 2:
        return winner
    return None


def set_winner_for_games(games, game_winner):
    if game_winner is None:
        return None
    loser = side_opponent(game_winner)
    if games[game_winner] >= 6 and games[game_winner] - games[loser] >= 2:
        return game_winner
    return None


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


def service_box_side(world_point):
    if not world_point:
        return None
    xw, _ = world_point
    if xw < SERVICE_LEFT_FT:
        return "left"
    return "right"


def classify_serve_attempt(bounce, server, receiver, attempt_number):
    world_point = bounce.get("world_point")
    if not world_point:
        result = "unknown"
        reason = "missing_world_point"
    elif world_point_in_receiver_service_box(world_point, receiver):
        result = "in"
        reason = "receiver_service_box"
    else:
        result = "fault"
        reason = "outside_receiver_service_box"
    return {
        "attempt": attempt_number,
        "bounce_id": bounce["id"],
        "bounce_frame": bounce["frame"],
        "server": server,
        "receiver": receiver,
        "result": result,
        "reason": reason,
        "service_box_side": service_box_side(world_point),
        "world_point": world_point,
    }


def infer_serve_sequence(point_bounces, server, receiver, official_double_fault=False):
    attempts = []
    for bounce in point_bounces[:2]:
        attempt = classify_serve_attempt(bounce, server, receiver, len(attempts) + 1)
        attempts.append(attempt)
        if attempt["result"] == "in":
            break
        if attempt["result"] == "unknown":
            break

    if not attempts:
        return {
            "status": "unknown",
            "point_end_reason": None,
            "inferred_winner": None,
            "attempts": [],
        }
    if len(attempts) >= 2 and attempts[0]["result"] == "fault" and attempts[1]["result"] == "fault":
        post_second_serve_bounces = max(0, len(point_bounces) - 2)
        if official_double_fault or post_second_serve_bounces <= 1:
            return {
                "status": "double_fault",
                "point_end_reason": "double_fault",
                "inferred_winner": receiver,
                "attempts": attempts,
                "post_second_serve_bounces": post_second_serve_bounces,
                "official_override": bool(official_double_fault),
            }
        return {
            "status": "geometric_double_fault_played_out",
            "point_end_reason": None,
            "inferred_winner": None,
            "attempts": attempts,
            "post_second_serve_bounces": post_second_serve_bounces,
            "official_override": False,
        }
    if attempts[0]["result"] == "fault" and len(attempts) >= 2 and attempts[1]["result"] == "in":
        return {
            "status": "second_serve_in",
            "point_end_reason": None,
            "inferred_winner": None,
            "attempts": attempts,
        }
    if attempts[0]["result"] == "fault":
        return {
            "status": "first_serve_fault",
            "point_end_reason": None,
            "inferred_winner": None,
            "attempts": attempts,
        }
    if attempts[0]["result"] == "in":
        return {
            "status": "serve_in",
            "point_end_reason": None,
            "inferred_winner": None,
            "attempts": attempts,
        }
    return {
        "status": "unknown",
        "point_end_reason": None,
        "inferred_winner": None,
        "attempts": attempts,
    }


def point_for_frame(point_ranges, frame):
    for index, point_range in enumerate(point_ranges, start=1):
        if point_range["start_frame"] <= frame <= point_range["end_frame"]:
            return index
    return None


def point_start_for_index(point_ranges, point_index):
    if point_index is None or point_index < 1 or point_index > len(point_ranges):
        return None
    return point_ranges[point_index - 1]["start_frame"]


def terminal_ball_state(rows, point_range, inv_homography):
    start = point_range["start_frame"]
    end = point_range["end_frame"]
    window_start = max(start, end - 90)
    rows_in_window = [row for row in rows if window_start <= row["frame"] <= end]
    ball_rows = [row for row in rows_in_window if row.get("ball")]
    if len(ball_rows) < 4:
        return {"status": "unknown", "reason": "not_enough_ball_frames"}

    last_rows = ball_rows[-8:]
    last = last_rows[-1]
    first = last_rows[0]
    last_center = last["ball"].get("center")
    first_center = first["ball"].get("center")
    last_world = ball_world_point(last, inv_homography)
    first_world = ball_world_point(first, inv_homography)
    missing_after_last = max(0, end - int(last["frame"]))
    near_frame_edge = False
    moving_outward = False
    frame_out_direction = None
    if last_center and first_center:
        dx = last_center[0] - first_center[0]
        dy = last_center[1] - first_center[1]
        x, y = last_center
        near_frame_edge = x <= 80 or x >= 1840 or y <= 80 or y >= 1020
        if x <= 120 and dx < -20:
            moving_outward = True
            frame_out_direction = "left"
        elif x >= 1800 and dx > 20:
            moving_outward = True
            frame_out_direction = "right"
        elif y <= 90 and dy < -20:
            moving_outward = True
            frame_out_direction = "top"
        elif y >= 990 and dy > 20:
            moving_outward = True
            frame_out_direction = "bottom"

    world_out = last_world is not None and not world_point_in_bounds(last_world, margin=2.0)
    world_direction = None
    if last_world and first_world:
        dxw = last_world[0] - first_world[0]
        dyw = last_world[1] - first_world[1]
        xw, yw = last_world
        if xw < 0.0 and dxw < -0.5:
            world_direction = "left"
        elif xw > COURT_WIDTH_FT and dxw > 0.5:
            world_direction = "right"
        elif yw < 0.0 and dyw < -0.5:
            world_direction = "far_long"
        elif yw > 78.0 and dyw > 0.5:
            world_direction = "near_long"

    if world_out or (near_frame_edge and moving_outward) or missing_after_last >= 2:
        return {
            "status": "out",
            "reason": "terminal_ball_left_play_area",
            "last_ball_frame": int(last["frame"]),
            "missing_after_last": missing_after_last,
            "last_center": last_center,
            "last_world_point": [round(last_world[0], 2), round(last_world[1], 2)] if last_world else None,
            "world_out": world_out,
            "world_direction": world_direction,
            "frame_out_direction": frame_out_direction,
            "confidence": "medium" if world_out or moving_outward else "low",
        }
    return {
        "status": "unknown",
        "reason": "terminal_ball_not_clearly_out",
        "last_ball_frame": int(last["frame"]),
        "last_center": last_center,
        "last_world_point": [round(last_world[0], 2), round(last_world[1], 2)] if last_world else None,
    }


def shot_type_for_link(point_shot_index, player, server, receiver):
    if point_shot_index == 1:
        return "serve" if player == server else "opening_shot"
    if point_shot_index == 2 and player == receiver:
        return "return"
    return "groundstroke"


def build_analysis(rows, args):
    court_calib = load_court_calibration(args.court_calib_file)
    inv_homography = build_inverse_court_homography(court_calib)
    frame_numbers = [row["frame"] for row in rows]
    by_frame = {row["frame"]: row for row in rows}

    point_ranges = parse_point_ranges(args.point_frames)
    winners = parse_sides(args.point_winners)
    official_double_fault_points = parse_indexes(args.official_double_fault_points)
    excluded_bounce_frame_ranges = parse_frame_ranges(args.exclude_bounce_frames)
    if point_ranges and winners and len(point_ranges) != len(winners):
        raise ValueError("--point-frames and --point-winners must have the same number of entries")

    raw_bounces = []
    for bounce_index, row in enumerate(
        [row for row in rows if row.get("event") and row.get("event", {}).get("type") == "bounce"], start=1
    ):
        event = row.get("event")
        if args.ignore_bounces_before_frame and int(event["frame"]) < args.ignore_bounces_before_frame:
            continue
        frame = int(event["frame"])
        point_index = point_for_frame(point_ranges, frame) if point_ranges else None
        raw_bounces.append(
            {
                "id": f"bounce_{len(raw_bounces) + 1:03d}",
                "frame": frame,
                "log_frame": int(row["frame"]),
                "side": event.get("side"),
                "pattern": event.get("pattern"),
                "point": event.get("point"),
                "world_point": event.get("world_point"),
                "bounce_strength": event.get("bounce_strength"),
                "point_index": point_index,
                "live": True,
                "exclude_reason": None,
            }
        )

    shots = []
    links = []
    per_point_bounce_counts = {}
    for bounce in raw_bounces:
        point_index = bounce.get("point_index")
        per_point_bounce_counts[point_index] = per_point_bounce_counts.get(point_index, 0) + 1
        if frame_in_ranges(bounce["frame"], excluded_bounce_frame_ranges):
            bounce["live"] = False
            bounce["exclude_reason"] = "manual_dead_ball"
        elif point_index in official_double_fault_points and per_point_bounce_counts[point_index] > 2:
            bounce["live"] = False
            bounce["exclude_reason"] = "post_double_fault_dead_ball"

    live_bounces = [bounce for bounce in raw_bounces if bounce.get("live")]
    previous_bounce_frame = None
    previous_bounce_record = None
    point_shot_counts = {}
    for shot_index, bounce in enumerate(live_bounces, start=1):
        point_index = bounce.get("point_index")
        point_start_frame = point_start_for_index(point_ranges, point_index)
        if point_start_frame is not None and previous_bounce_frame is not None and previous_bounce_frame < point_start_frame:
            previous_bounce_frame = None
            previous_bounce_record = None
        shot = find_shot_for_bounce(
            rows,
            frame_numbers,
            bounce,
            previous_bounce_frame,
            point_start_frame,
            args,
            inv_homography,
        )
        handedness = args.near_handedness if shot["player"] == "near" else args.far_handedness
        shot_row = by_frame.get(shot["frame"])
        point_shot_counts[point_index] = point_shot_counts.get(point_index, 0) + 1
        shot_type = shot_type_for_link(point_shot_counts[point_index], shot["player"], args.server, args.receiver)
        stroke_side = infer_stroke_side(shot, shot_row, handedness)
        speed = estimate_speed(shot, bounce, previous_bounce_record, args.fps)
        shot_id = f"shot_{shot_index:03d}"
        bounce_id = bounce["id"]
        shot_record = {
            "id": shot_id,
            "frame": shot["frame"],
            "player": shot["player"],
            "point": shot.get("point"),
            "world_point": shot.get("world_point"),
            "stroke_side": stroke_side,
            "type": shot_type,
            "speed": speed,
            "quality": shot.get("quality"),
            "debug": {
                "player_distance_px": shot.get("player_distance_px"),
                "racket_distance_px": shot.get("racket_distance_px"),
                "racket_bbox": shot.get("racket_bbox"),
                "candidate_score": round(shot["score"], 2) if shot.get("score") is not None else None,
            },
        }
        bounce_record = {**bounce, "id": bounce_id, "point_index": point_index}
        shots.append(shot_record)
        links.append(
            {
                "id": f"link_{shot_index:03d}",
                "point_index": point_index,
                "shot_id": shot_id,
                "bounce_id": bounce_id,
                "shot_frame": shot_record["frame"],
                "bounce_frame": bounce["frame"],
                "player": shot_record["player"],
                "stroke_side": stroke_side,
                "shot_type": shot_type,
                "speed_mph": speed["mph"],
                "quality": min_quality(shot_record["quality"], speed["quality"]),
            }
        )
        bounce.clear()
        bounce.update(bounce_record)
        previous_bounce_frame = bounce["frame"]
        previous_bounce_record = bounce

    points = []
    point_score = {"near": 0, "far": 0}
    initial_games = parse_score_pair(args.initial_game_score, "--initial-game-score")
    initial_sets = parse_score_pair(args.initial_set_score, "--initial-set-score")
    game_score = {
        args.server: initial_games["server"],
        args.receiver: initial_games["receiver"],
    }
    set_score = {
        args.server: initial_sets["server"],
        args.receiver: initial_sets["receiver"],
    }
    for index, point_range in enumerate(point_ranges, start=1):
        point_score_before = format_score(point_score, args.server, args.receiver)
        game_score_before = format_side_score(game_score, args.server, args.receiver)
        set_score_before = format_side_score(set_score, args.server, args.receiver)
        point_links = [link for link in links if link["point_index"] == index]
        point_bounces = [bounce for bounce in live_bounces if bounce.get("point_index") == index]
        serve_analysis = infer_serve_sequence(
            point_bounces,
            args.server,
            args.receiver,
            official_double_fault=index in official_double_fault_points,
        )
        terminal_state = terminal_ball_state(rows, point_range, inv_homography)
        point_end_reason = serve_analysis.get("point_end_reason")
        if point_end_reason is None and terminal_state.get("status") == "out":
            point_end_reason = "out"
        manual_winner = winners[index - 1] if winners else None
        if args.auto_score and serve_analysis.get("inferred_winner") is not None:
            winner = serve_analysis["inferred_winner"]
            winner_source = "auto"
        else:
            winner = manual_winner
            winner_source = "manual" if manual_winner else None
        game_winner = None
        set_winner = None
        if winner:
            point_score[winner] += 1
            game_winner = game_winner_for_points(point_score, winner)
            if game_winner:
                game_score[game_winner] += 1
                set_winner = set_winner_for_games(game_score, game_winner)
                if set_winner:
                    set_score[set_winner] += 1
                point_score = {"near": 0, "far": 0}
            score_after = format_score(point_score, args.server, args.receiver)
        else:
            score_after = None
        game_score_after = format_side_score(game_score, args.server, args.receiver)
        set_score_after = format_side_score(set_score, args.server, args.receiver)
        points.append(
            {
                "index": index,
                **point_range,
                "server": args.server,
                "receiver": args.receiver,
                "winner": winner,
                "winner_source": winner_source,
                "point_end_reason": point_end_reason,
                "terminal_ball": terminal_state,
                "serve_status": serve_analysis.get("status"),
                "serve_analysis": serve_analysis,
                "point_score_before": point_score_before,
                "score_after": score_after,
                "game_score_before": game_score_before,
                "game_score_after": game_score_after,
                "set_score_before": set_score_before,
                "set_score_after": set_score_after,
                "game_winner": game_winner,
                "set_winner": set_winner,
                "bounce_count": len(point_links),
                "shot_ids": [link["shot_id"] for link in point_links],
                "bounce_ids": [link["bounce_id"] for link in point_links],
            }
        )

    return {
        "source_jsonl": args.jsonl,
        "court_calib_file": args.court_calib_file,
        "fps": args.fps,
        "players": {
            "near": {"handedness": args.near_handedness},
            "far": {"handedness": args.far_handedness},
        },
        "summary": {
            "frames": len(rows),
            "bounces": len(live_bounces),
            "raw_bounces": len(raw_bounces),
            "excluded_bounces": len(raw_bounces) - len(live_bounces),
            "shots": len(shots),
            "points": len(points),
            "scoring_mode": "auto_with_manual_fallback" if args.auto_score else ("manual_point_winners" if winners else "none"),
            "final_point_score": points[-1]["score_after"] if points and points[-1].get("score_after") else format_score(point_score, args.server, args.receiver),
            "final_game_score": format_side_score(game_score, args.server, args.receiver),
            "final_set_score": format_side_score(set_score, args.server, args.receiver),
        },
        "points": points,
        "bounces": raw_bounces,
        "live_bounce_ids": [bounce["id"] for bounce in live_bounces],
        "excluded_bounces": [bounce for bounce in raw_bounces if not bounce.get("live")],
        "shots": shots,
        "shot_bounce_links": links,
    }


def min_quality(*qualities):
    order = {"missing": 0, "low": 1, "medium": 2, "high": 3}
    labels = [quality for quality in qualities if quality]
    if not labels:
        return "unknown"
    return min(labels, key=lambda label: order.get(label, 1))


def main():
    args = parse_args()
    rows, _by_frame = read_tracking_log(args.jsonl)
    analysis = build_analysis(rows, args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
        handle.write("\n")
    summary = analysis["summary"]
    print(
        "wrote {output}: {points} points, {shots} shots, {bounces} bounces".format(
            output=args.output,
            points=summary["points"],
            shots=summary["shots"],
            bounces=summary["bounces"],
        )
    )


if __name__ == "__main__":
    main()
