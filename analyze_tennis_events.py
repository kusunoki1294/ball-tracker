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
    parser.add_argument("--manifest", default="", help="Optional JSON manifest with analysis inputs and overrides.")
    parser.add_argument("--jsonl", default="", help="Input tracking JSONL from track_ball_yolo.py.")
    parser.add_argument("--output", default="", help="Output analysis JSON path.")
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
        "--server-player",
        default="server",
        help="Stable player ID/name for the player serving the first tracked game.",
    )
    parser.add_argument(
        "--receiver-player",
        default="receiver",
        help="Stable player ID/name for the player receiving the first tracked game.",
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
    args = parser.parse_args()
    if args.manifest:
        args = apply_manifest(args, args.manifest)
    if not args.jsonl:
        parser.error("--jsonl is required unless provided by --manifest")
    if not args.output:
        parser.error("--output is required unless provided by --manifest")
    return args


def apply_manifest(args, manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    field_map = {
        "jsonl": "jsonl",
        "output": "output",
        "court_calib_file": "court_calib_file",
        "fps": "fps",
        "point_frames": "point_frames",
        "point_winners": "point_winners",
        "server": "server",
        "receiver": "receiver",
        "server_player": "server_player",
        "receiver_player": "receiver_player",
        "near_handedness": "near_handedness",
        "far_handedness": "far_handedness",
        "max_shot_search_frames": "max_shot_search_frames",
        "auto_score": "auto_score",
        "ignore_bounces_before_frame": "ignore_bounces_before_frame",
        "official_double_fault_points": "official_double_fault_points",
        "exclude_bounce_frames": "exclude_bounce_frames",
        "initial_game_score": "initial_game_score",
        "initial_set_score": "initial_set_score",
    }
    defaults = {
        "jsonl": "",
        "output": "",
        "court_calib_file": "yoloVids/calibration/court_calib_tennis7.json",
        "fps": FPS_DEFAULT,
        "point_frames": "",
        "point_winners": "",
        "server": "near",
        "receiver": "far",
        "server_player": "server",
        "receiver_player": "receiver",
        "near_handedness": "right",
        "far_handedness": "right",
        "max_shot_search_frames": 150,
        "auto_score": False,
        "ignore_bounces_before_frame": 0,
        "official_double_fault_points": "",
        "exclude_bounce_frames": "",
        "initial_game_score": "0-0",
        "initial_set_score": "0-0",
    }
    for manifest_key, attr in field_map.items():
        if manifest_key not in manifest:
            continue
        current = getattr(args, attr)
        if current != defaults[attr]:
            continue
        setattr(args, attr, manifest_value_to_arg(manifest[manifest_key]))
    return args


def manifest_value_to_arg(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(item, dict) and "start_frame" in item and "end_frame" in item for item in value):
            return ",".join(f"{item['start_frame']}-{item['end_frame']}" for item in value)
        return ",".join(str(item) for item in value)
    return value


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


def infer_stroke(shot, row, handedness):
    if shot.get("point") is None:
        return {"side": "unknown", "confidence": "missing", "reason": "missing_contact"}
    if shot.get("quality") not in {"high", "medium"}:
        return {"side": "unknown", "confidence": "low", "reason": "low_shot_quality"}
    debug = shot.get("debug") or shot
    racket_distance_px = debug.get("racket_distance_px")
    if racket_distance_px is None or racket_distance_px > 80.0:
        world_point = shot.get("world_point")
        if not world_point_in_bounds(world_point, margin=8.0) or not world_point_on_player_side(
            world_point, shot["player"], margin=10.0
        ):
            return {"side": "unknown", "confidence": "low", "reason": "contact_not_on_player_side"}
    player = get_player(row, shot["player"]) if row else None
    center = player_center(player)
    if center is None:
        return {"side": "unknown", "confidence": "missing", "reason": "missing_player"}
    contact_x = shot["point"][0]
    racket_bbox = debug.get("racket_bbox")
    used_racket = False
    if racket_bbox is not None and racket_distance_px is not None and racket_distance_px <= 80.0:
        contact_x = (racket_bbox[0] + racket_bbox[2]) / 2.0
        used_racket = True
    player_x = center[0]
    offset = abs(contact_x - player_x)
    if offset < 14.0:
        confidence = "high" if used_racket and racket_distance_px <= 45.0 else "medium"
        return {"side": "body", "confidence": confidence, "reason": "center_contact"}
    if offset < 28.0 and not used_racket:
        return {"side": "unknown", "confidence": "low", "reason": "small_offset_without_racket"}
    if shot["player"] == "near":
        right_side_contact = contact_x > player_x
    else:
        right_side_contact = contact_x < player_x
    if handedness == "left":
        right_side_contact = not right_side_contact
    confidence = "high" if used_racket and racket_distance_px <= 55.0 else "medium"
    return {
        "side": "forehand" if right_side_contact else "backhand",
        "confidence": confidence,
        "reason": "racket_contact" if used_racket else "ball_player_offset",
    }


def infer_stroke_side(shot, row, handedness):
    return infer_stroke(shot, row, handedness)["side"]


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
    elif shot.get("quality") == "high" and frame_delta <= 75:
        quality = "high"
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


def format_score(score, server, receiver, tiebreak=False):
    if tiebreak:
        return f"{score[server]}-{score[receiver]}"
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


def format_ordered_score(score, order):
    return f"{score[order[0]]}-{score[order[1]]}"


def opposing_player(player, player_order):
    return player_order[1] if player == player_order[0] else player_order[0]


def invert_mapping(mapping):
    return {value: key for key, value in mapping.items()}


def should_change_ends(games):
    return sum(games.values()) % 2 == 1


def game_winner_for_points(score, winner, player_order=None):
    if winner is None:
        return None
    loser = opposing_player(winner, player_order) if player_order else side_opponent(winner)
    if score[winner] >= 4 and score[winner] - score[loser] >= 2:
        return winner
    return None


def set_winner_for_games(games, game_winner, player_order=None):
    if game_winner is None:
        return None
    loser = opposing_player(game_winner, player_order) if player_order else side_opponent(game_winner)
    if games[game_winner] >= 6 and games[game_winner] - games[loser] >= 2:
        return game_winner
    return None


def is_tiebreak_game(games):
    values = list(games.values())
    return len(values) == 2 and values[0] == 6 and values[1] == 6


def tiebreak_winner_for_points(score, winner, player_order=None):
    if winner is None:
        return None
    loser = opposing_player(winner, player_order) if player_order else side_opponent(winner)
    if score[winner] >= 7 and score[winner] - score[loser] >= 2:
        return winner
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


def serve_state_result(
    status,
    point_end_reason,
    inferred_winner,
    attempts,
    state,
    confidence,
    reasons,
    **extra,
):
    result = {
        "status": status,
        "state": state,
        "confidence": confidence,
        "reasons": reasons,
        "point_end_reason": point_end_reason,
        "inferred_winner": inferred_winner,
        "attempts": attempts,
    }
    result.update(extra)
    return result


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
        return serve_state_result(
            "unknown",
            None,
            None,
            [],
            "waiting_for_serve",
            "low",
            ["no_live_bounces_for_point"],
        )
    if len(attempts) >= 2 and attempts[0]["result"] == "fault" and attempts[1]["result"] == "fault":
        post_second_serve_bounces = max(0, len(point_bounces) - 2)
        if official_double_fault or post_second_serve_bounces <= 1:
            confidence = "high" if official_double_fault else "medium"
            reasons = ["official_double_fault_override"] if official_double_fault else ["two_fault_bounces_no_rally"]
            return serve_state_result(
                "double_fault",
                "double_fault",
                receiver,
                attempts,
                "double_fault",
                confidence,
                reasons,
                post_second_serve_bounces=post_second_serve_bounces,
                official_override=bool(official_double_fault),
            )
        return serve_state_result(
            "geometric_double_fault_played_out",
            None,
            None,
            attempts,
            "played_out_after_geometric_fault",
            "low",
            ["two_geometric_faults_but_rally_continued"],
            post_second_serve_bounces=post_second_serve_bounces,
            official_override=False,
        )
    if attempts[0]["result"] == "fault" and len(attempts) >= 2 and attempts[1]["result"] == "in":
        return serve_state_result(
            "second_serve_in",
            None,
            None,
            attempts,
            "second_serve_in",
            "high",
            ["first_fault_then_second_serve_in"],
        )
    if attempts[0]["result"] == "fault":
        return serve_state_result(
            "first_serve_fault",
            None,
            None,
            attempts,
            "first_serve_fault",
            "medium",
            ["first_serve_landed_out"],
        )
    if attempts[0]["result"] == "in":
        return serve_state_result(
            "serve_in",
            None,
            None,
            attempts,
            "first_serve_in",
            "high",
            ["first_serve_landed_in"],
        )
    return serve_state_result(
        "unknown",
        None,
        None,
        attempts,
        "unknown",
        "low",
        ["serve_attempt_unknown"],
    )


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


def infer_terminal_contact(rows, point_range, terminal_state, after_frame, inv_homography):
    terminal_frame = terminal_state.get("last_ball_frame") or point_range["end_frame"]
    search_start = max(point_range["start_frame"], (after_frame or point_range["start_frame"]) + 1, terminal_frame - 120)
    search_end = min(point_range["end_frame"], terminal_frame)
    best = None
    for row in rows:
        frame = int(row["frame"])
        if frame < search_start or frame > search_end:
            continue
        ball = row.get("ball")
        if not ball:
            continue
        point = ball_contact_point(ball) or ball.get("center")
        if point is None:
            continue
        world_point = project_to_court_world(point, inv_homography)
        row_side = side_for_world(world_point)
        for side in ("near", "far"):
            player = get_player(row, side)
            zone = expanded_strike_zone(player, side)
            if zone is None:
                continue
            player_distance = point_to_bbox_distance(point, zone)
            nearest = nearest_racket(row, point)
            racket_distance_px = nearest["distance"] if nearest else None
            racket_score = min(racket_distance_px if racket_distance_px is not None else 120.0, 120.0) * 0.55
            side_bonus = 0.0 if row_side in {None, side} else 22.0
            recency = (search_end - frame) * 0.08
            score = player_distance + racket_score + side_bonus + recency
            quality = "low"
            if racket_distance_px is not None and racket_distance_px <= 60.0 and player_distance <= 45.0:
                quality = "high"
            elif player_distance <= 65.0 or (racket_distance_px is not None and racket_distance_px <= 85.0):
                quality = "medium"
            candidate = {
                "frame": frame,
                "player": side,
                "point": [round(float(point[0]), 1), round(float(point[1]), 1)],
                "world_point": [round(world_point[0], 2), round(world_point[1], 2)] if world_point else None,
                "quality": quality,
                "score": round(score, 2),
                "player_distance_px": round(player_distance, 1),
                "racket_distance_px": round(racket_distance_px, 1) if racket_distance_px is not None else None,
            }
            if best is None or score < best["score"]:
                best = candidate
    if best is None:
        return {
            "player": None,
            "quality": "missing",
            "reason": "no_terminal_contact_candidate",
            "search_range": [search_start, search_end],
        }
    if best["quality"] == "low":
        best["reason"] = "weak_terminal_contact_candidate"
    else:
        best["reason"] = "terminal_contact_candidate"
    best["search_range"] = [search_start, search_end]
    return best


def pressure_from_previous_shot(previous_shot, previous_bounce):
    reasons = []
    if not previous_shot:
        return {"forced": False, "reasons": ["missing_previous_shot"]}
    speed = previous_shot.get("speed") or {}
    if quality_at_least(speed.get("quality"), "medium") and (speed.get("mph") or 0) >= 40.0:
        reasons.append("previous_shot_fast")
    world = (previous_bounce or {}).get("world_point")
    if world:
        xw, yw = world
        if xw <= SINGLES_LEFT_FT + 2.0 or xw >= SINGLES_RIGHT_FT - 2.0:
            reasons.append("previous_bounce_wide")
        if yw <= 6.0 or yw >= 72.0:
            reasons.append("previous_bounce_deep")
    return {"forced": bool(reasons), "reasons": reasons or ["no_pressure_signal"]}


def classify_point_end(serve_analysis, terminal_state, terminal_contact, point_links, shot_lookup, bounce_lookup):
    reasons = []
    review_flags = []
    serve_state = serve_analysis.get("state")
    if serve_state == "played_out_after_geometric_fault":
        review_flags.append("serve_geometry_disagrees_with_play_continuation")
    if serve_analysis.get("status") == "double_fault":
        return {
            "type": "double_fault",
            "confidence": serve_analysis.get("confidence") or "medium",
            "reasons": serve_analysis.get("reasons") or ["double_fault"],
            "review_flags": review_flags,
            "inferred_winner": serve_analysis.get("inferred_winner"),
            "scoring_eligible": quality_at_least(serve_analysis.get("confidence"), "medium"),
        }

    if terminal_state.get("status") == "out":
        if terminal_state.get("world_out"):
            reasons.append("terminal_ball_world_out")
        if terminal_state.get("frame_out_direction"):
            reasons.append("terminal_ball_moving_out_of_frame")
        if terminal_state.get("missing_after_last", 0) >= 2:
            reasons.append("final_ball_missing_before_point_end")
        if terminal_state.get("confidence") == "low":
            review_flags.append("low_confidence_terminal_out")
        if not terminal_state.get("world_out") and not terminal_state.get("frame_out_direction"):
            review_flags.append("final_ball_out_of_frame")

        contact_quality = terminal_contact.get("quality")
        terminal_quality = terminal_state.get("confidence") or "low"
        if not terminal_contact.get("player") or not quality_at_least(contact_quality, "medium"):
            return {
                "type": "unknown_end",
                "confidence": "low",
                "reasons": reasons + [terminal_contact.get("reason") or "missing_terminal_contact"],
                "review_flags": review_flags + ["cannot_identify_final_hitter"],
                "terminal_contact": terminal_contact,
                "inferred_winner": None,
                "scoring_eligible": False,
            }

        previous_links = [
            link for link in point_links if int(link.get("shot_frame") or 0) < int(terminal_contact.get("frame") or 0)
        ]
        previous_link = previous_links[-1] if previous_links else None
        previous_shot = shot_lookup.get(previous_link.get("shot_id")) if previous_link else None
        previous_bounce = bounce_lookup.get(previous_link.get("bounce_id")) if previous_link else None
        pressure = pressure_from_previous_shot(previous_shot, previous_bounce)
        end_type = "forced_error_out" if pressure["forced"] else "unforced_error_out"
        confidence = min_quality(terminal_quality, contact_quality)
        if confidence == "high":
            confidence = "medium"
        return {
            "type": end_type,
            "confidence": confidence,
            "reasons": reasons + pressure["reasons"],
            "review_flags": review_flags,
            "terminal_contact": terminal_contact,
            "inferred_winner": side_opponent(terminal_contact["player"]),
            "scoring_eligible": quality_at_least(confidence, "medium"),
        }

    last_world = terminal_state.get("last_world_point")
    if last_world and abs(last_world[1] - COURT_NET_Y_FT) <= 5.0:
        return {
            "type": "net_error",
            "confidence": "low",
            "reasons": ["terminal_ball_near_net"],
            "review_flags": review_flags + ["low_confidence_net_error"],
            "terminal_contact": terminal_contact,
            "inferred_winner": side_opponent(terminal_contact["player"]) if terminal_contact.get("player") else None,
            "scoring_eligible": False,
        }

    return {
        "type": "unknown_end",
        "confidence": "low",
        "reasons": [terminal_state.get("reason") or "unknown_point_end"],
        "review_flags": review_flags + ["unknown_point_end"],
        "terminal_contact": terminal_contact,
        "inferred_winner": None,
        "scoring_eligible": False,
    }


def shot_type_for_link(point_shot_index, player, server, receiver):
    if point_shot_index == 1:
        return "serve" if player == server else "opening_shot"
    if point_shot_index == 2 and player == receiver:
        return "return"
    return "groundstroke"


def mark_dead_ball_candidates(raw_bounces, links, shots, point_ranges):
    links_by_bounce = {link["bounce_id"]: link for link in links}
    shots_by_id = {shot["id"]: shot for shot in shots}
    bounces_by_point = {}
    for bounce in raw_bounces:
        bounces_by_point.setdefault(bounce.get("point_index"), []).append(bounce)

    for bounce in raw_bounces:
        reasons = []
        review_reasons = []
        if not bounce.get("live"):
            reasons.append("already_excluded")
        link = links_by_bounce.get(bounce.get("id"))
        shot = shots_by_id.get(link.get("shot_id")) if link else None
        if link is None and bounce.get("live"):
            reasons.append("no_live_shot_link")
        if link and link.get("quality") in {"low", "missing"}:
            review_reasons.append("low_link_quality")
        if shot:
            speed = shot.get("speed") or {}
            if shot.get("stroke_confidence") == "low":
                review_reasons.append("low_stroke_confidence")
            if speed.get("quality") in {"low", "missing"}:
                review_reasons.append("low_speed_quality")

        point_bounces = bounces_by_point.get(bounce.get("point_index")) or []
        live_point_bounces = [item for item in point_bounces if item.get("live")]
        live_index = None
        for index, item in enumerate(live_point_bounces, start=1):
            if item.get("id") == bounce.get("id"):
                live_index = index
                break
        if live_index == 1 and link:
            shot_type = link.get("shot_type")
            shot_player = link.get("player")
            if shot_type != "serve" and shot_player != "near" and bounce.get("side") == "near":
                reasons.append("nonserve_first_bounce_near_side")
        if bounce.get("point_index") is None:
            reasons.append("outside_point_range")
        if point_ranges and bounce.get("point_index") is not None:
            point_range = point_ranges[bounce["point_index"] - 1]
            if bounce["frame"] - point_range["start_frame"] < 45 and live_index and live_index > 1:
                reasons.append("early_multi_bounce_sequence")

        bounce["dead_ball_candidate"] = bool(reasons)
        bounce["dead_ball_reasons"] = sorted(set(reasons))
        bounce["review_reasons"] = sorted(set(review_reasons))


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
        stroke = infer_stroke(shot, shot_row, handedness)
        stroke_side = stroke["side"]
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
            "stroke_confidence": stroke["confidence"],
            "stroke_reason": stroke["reason"],
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
                "stroke_confidence": stroke["confidence"],
                "shot_type": shot_type,
                "speed_mph": speed["mph"],
                "quality": min_quality(shot_record["quality"], speed["quality"]),
            }
        )
        bounce.clear()
        bounce.update(bounce_record)
        previous_bounce_frame = bounce["frame"]
        previous_bounce_record = bounce

    mark_dead_ball_candidates(raw_bounces, links, shots, point_ranges)
    shot_lookup = {shot["id"]: shot for shot in shots}
    bounce_lookup = {bounce["id"]: bounce for bounce in raw_bounces}

    points = []
    player_order = [args.server_player, args.receiver_player]
    point_score = {player: 0 for player in player_order}
    initial_games = parse_score_pair(args.initial_game_score, "--initial-game-score")
    initial_sets = parse_score_pair(args.initial_set_score, "--initial-set-score")
    game_score = {
        args.server_player: initial_games["server"],
        args.receiver_player: initial_games["receiver"],
    }
    set_score = {
        args.server_player: initial_sets["server"],
        args.receiver_player: initial_sets["receiver"],
    }
    side_to_player = {
        args.server: args.server_player,
        args.receiver: args.receiver_player,
    }
    games = []
    current_game_start_point = 1
    for index, point_range in enumerate(point_ranges, start=1):
        completed_games = sum(game_score.values())
        game_index = completed_games + 1
        current_server_player = player_order[completed_games % 2]
        current_receiver_player = opposing_player(current_server_player, player_order)
        player_to_side = invert_mapping(side_to_player)
        current_server_side = player_to_side[current_server_player]
        current_receiver_side = player_to_side[current_receiver_player]
        tiebreak_before = is_tiebreak_game(game_score)
        point_score_before = format_score(
            point_score,
            current_server_player,
            current_receiver_player,
            tiebreak=tiebreak_before,
        )
        game_score_before = format_ordered_score(game_score, player_order)
        set_score_before = format_ordered_score(set_score, player_order)
        side_to_player_before = dict(side_to_player)
        point_links = [link for link in links if link["point_index"] == index]
        point_bounces = [bounce for bounce in live_bounces if bounce.get("point_index") == index]
        serve_analysis = infer_serve_sequence(
            point_bounces,
            current_server_side,
            current_receiver_side,
            official_double_fault=index in official_double_fault_points,
        )
        terminal_state = terminal_ball_state(rows, point_range, inv_homography)
        last_bounce_frame = max([bounce["frame"] for bounce in point_bounces], default=point_range["start_frame"])
        terminal_contact = infer_terminal_contact(
            rows,
            point_range,
            terminal_state,
            last_bounce_frame,
            inv_homography,
        )
        point_end_analysis = classify_point_end(
            serve_analysis,
            terminal_state,
            terminal_contact,
            point_links,
            shot_lookup,
            bounce_lookup,
        )
        point_end_reason = serve_analysis.get("point_end_reason")
        if point_end_reason is None:
            if point_end_analysis.get("type") in {"forced_error_out", "unforced_error_out"}:
                point_end_reason = "out"
            elif point_end_analysis.get("type") == "net_error":
                point_end_reason = "net"
            elif terminal_state.get("status") == "out":
                point_end_reason = "out"
        manual_winner = winners[index - 1] if winners else None
        auto_winner = None
        if point_end_analysis.get("scoring_eligible"):
            auto_winner = point_end_analysis.get("inferred_winner")
        if args.auto_score and auto_winner is not None:
            winner = auto_winner
            winner_source = "auto"
        else:
            winner = manual_winner
            if manual_winner and args.auto_score and point_end_analysis.get("inferred_winner") is not None:
                winner_source = "manual_low_confidence_auto_fallback"
            else:
                winner_source = "manual" if manual_winner else None
        winner_player = side_to_player.get(winner) if winner else None
        game_winner = None
        set_winner = None
        tiebreak_winner = None
        changeover_after = False
        if winner_player:
            point_score[winner_player] += 1
            if tiebreak_before:
                tiebreak_winner = tiebreak_winner_for_points(point_score, winner_player, player_order)
                if tiebreak_winner:
                    game_score[tiebreak_winner] += 1
                    game_winner = tiebreak_winner
                    set_winner = tiebreak_winner
                    set_score[set_winner] += 1
                    point_score = {player: 0 for player in player_order}
            else:
                game_winner = game_winner_for_points(point_score, winner_player, player_order)
                if game_winner:
                    game_score[game_winner] += 1
                    set_winner = set_winner_for_games(game_score, game_winner, player_order)
                    if set_winner:
                        set_score[set_winner] += 1
                    point_score = {player: 0 for player in player_order}
            tiebreak_after = is_tiebreak_game(game_score)
            if set_winner:
                point_score = {player: 0 for player in player_order}
                tiebreak_after = False
            if game_winner and should_change_ends(game_score):
                side_to_player = {
                    "near": side_to_player["far"],
                    "far": side_to_player["near"],
                }
                changeover_after = True
            score_after = format_score(
                point_score,
                current_server_player,
                current_receiver_player,
                tiebreak=tiebreak_after,
            )
        else:
            score_after = None
            tiebreak_after = tiebreak_before
        game_score_after = format_ordered_score(game_score, player_order)
        set_score_after = format_ordered_score(set_score, player_order)
        next_completed_games = sum(game_score.values())
        next_server_player = player_order[next_completed_games % 2]
        next_receiver_player = opposing_player(next_server_player, player_order)
        next_player_to_side = invert_mapping(side_to_player)
        next_server_side = next_player_to_side.get(next_server_player)
        next_receiver_side = next_player_to_side.get(next_receiver_player)
        if game_winner:
            games.append(
                {
                    "index": game_index,
                    "start_point_index": current_game_start_point,
                    "end_point_index": index,
                    "server": current_server_side,
                    "receiver": current_receiver_side,
                    "server_player": current_server_player,
                    "receiver_player": current_receiver_player,
                    "winner_player": game_winner,
                    "score_before": game_score_before,
                    "score_after": game_score_after,
                    "set_score_after": set_score_after,
                    "changeover_after": changeover_after,
                    "side_to_player_after": dict(side_to_player),
                    "next_server": next_server_side,
                    "next_receiver": next_receiver_side,
                    "next_server_player": next_server_player,
                    "next_receiver_player": next_receiver_player,
                    "tiebreak": bool(tiebreak_before),
                }
            )
            current_game_start_point = index + 1
        points.append(
            {
                "index": index,
                **point_range,
                "server": current_server_side,
                "receiver": current_receiver_side,
                "server_player": current_server_player,
                "receiver_player": current_receiver_player,
                "side_to_player_before": side_to_player_before,
                "winner": winner,
                "winner_player": winner_player,
                "winner_source": winner_source,
                "point_end_reason": point_end_reason,
                "point_end_type": point_end_analysis.get("type"),
                "point_end_confidence": point_end_analysis.get("confidence"),
                "point_end_reasons": point_end_analysis.get("reasons"),
                "point_review_flags": point_end_analysis.get("review_flags"),
                "point_end_analysis": point_end_analysis,
                "terminal_ball": terminal_state,
                "serve_status": serve_analysis.get("status"),
                "serve_state": serve_analysis.get("state"),
                "serve_confidence": serve_analysis.get("confidence"),
                "serve_reasons": serve_analysis.get("reasons"),
                "serve_analysis": serve_analysis,
                "point_score_before": point_score_before,
                "score_after": score_after,
                "tiebreak_before": tiebreak_before,
                "tiebreak_after": tiebreak_after,
                "game_score_before": game_score_before,
                "game_score_after": game_score_after,
                "set_score_before": set_score_before,
                "set_score_after": set_score_after,
                "game_index": game_index,
                "game_winner": game_winner,
                "set_winner": set_winner,
                "tiebreak_winner": tiebreak_winner,
                "changeover_after": changeover_after,
                "next_server": next_server_side if game_winner else None,
                "next_receiver": next_receiver_side if game_winner else None,
                "next_server_player": next_server_player if game_winner else None,
                "next_receiver_player": next_receiver_player if game_winner else None,
                "side_to_player_after": dict(side_to_player),
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
            "order": player_order,
            "initial_side_to_player": {
                args.server: args.server_player,
                args.receiver: args.receiver_player,
            },
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
            "final_point_score": points[-1]["score_after"] if points and points[-1].get("score_after") else format_score(point_score, player_order[0], player_order[1]),
            "final_game_score": format_ordered_score(game_score, player_order),
            "final_set_score": format_ordered_score(set_score, player_order),
            "completed_games": len(games),
        },
        "points": points,
        "games": games,
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


def quality_at_least(quality, minimum):
    order = {"missing": 0, "low": 1, "medium": 2, "high": 3}
    return order.get(quality, 0) >= order.get(minimum, 0)


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
