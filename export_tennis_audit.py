import argparse
import csv
import json
import os

import cv2
import numpy as np


COURT_WIDTH_FT = 36.0
COURT_LENGTH_FT = 78.0
SINGLES_LEFT_FT = 4.5
SINGLES_RIGHT_FT = 31.5
NET_Y_FT = 39.0
SERVICE_Y_FAR_FT = 18.0
SERVICE_Y_NEAR_FT = 60.0
SERVICE_X_CENTER_FT = 18.0


def parse_args():
    parser = argparse.ArgumentParser(description="Export review artifacts from tennis analysis JSON.")
    parser.add_argument("--analysis", required=True, help="Analysis JSON from analyze_tennis_events.py.")
    parser.add_argument("--csv", required=True, help="Output CSV path for bounce/shot review rows.")
    parser.add_argument("--summary-json", required=True, help="Output compact summary JSON path.")
    parser.add_argument("--court-map", required=True, help="Output PNG court map path.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def by_id(items):
    return {item.get("id"): item for item in items if item.get("id")}


def point_by_index(points):
    return {point.get("index"): point for point in points if point.get("index") is not None}


def export_csv(analysis, path):
    ensure_parent(path)
    shots = by_id(analysis.get("shots", []))
    links_by_bounce = {
        link.get("bounce_id"): link for link in analysis.get("shot_bounce_links", []) if link.get("bounce_id")
    }
    points = point_by_index(analysis.get("points", []))
    fieldnames = [
        "bounce_id",
        "live",
        "exclude_reason",
        "dead_ball_candidate",
        "dead_ball_reasons",
        "review_reasons",
        "point_index",
        "bounce_frame",
        "bounce_side",
        "bounce_pattern",
        "bounce_world_x",
        "bounce_world_y",
        "serve_status",
        "serve_state",
        "serve_confidence",
        "serve_reasons",
        "serve_attempt",
        "serve_result",
        "point_end_reason",
        "point_end_type",
        "point_end_confidence",
        "point_end_reasons",
        "point_review_flags",
        "point_winner",
        "point_winner_player",
        "shot_id",
        "shot_frame",
        "shot_player",
        "shot_type",
        "stroke_side",
        "stroke_confidence",
        "stroke_reason",
        "speed_mph",
        "speed_quality",
        "speed_source",
        "link_quality",
    ]
    serve_attempts = {}
    for point in analysis.get("points", []):
        for attempt in (point.get("serve_analysis") or {}).get("attempts") or []:
            serve_attempts[attempt.get("bounce_id")] = attempt
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bounce in analysis.get("bounces", []):
            link = links_by_bounce.get(bounce.get("id")) or {}
            shot = shots.get(link.get("shot_id")) or {}
            point = points.get(bounce.get("point_index")) or {}
            attempt = serve_attempts.get(bounce.get("id")) or {}
            world = bounce.get("world_point") or [None, None]
            speed = shot.get("speed") or {}
            writer.writerow(
                {
                    "bounce_id": bounce.get("id"),
                    "live": bounce.get("live"),
                    "exclude_reason": bounce.get("exclude_reason"),
                    "dead_ball_candidate": bounce.get("dead_ball_candidate"),
                    "dead_ball_reasons": "|".join(bounce.get("dead_ball_reasons") or []),
                    "review_reasons": "|".join(bounce.get("review_reasons") or []),
                    "point_index": bounce.get("point_index"),
                    "bounce_frame": bounce.get("frame"),
                    "bounce_side": bounce.get("side"),
                    "bounce_pattern": bounce.get("pattern"),
                    "bounce_world_x": world[0],
                    "bounce_world_y": world[1],
                    "serve_status": point.get("serve_status"),
                    "serve_state": point.get("serve_state"),
                    "serve_confidence": point.get("serve_confidence"),
                    "serve_reasons": "|".join(point.get("serve_reasons") or []),
                    "serve_attempt": attempt.get("attempt"),
                    "serve_result": attempt.get("result"),
                    "point_end_reason": point.get("point_end_reason"),
                    "point_end_type": point.get("point_end_type"),
                    "point_end_confidence": point.get("point_end_confidence"),
                    "point_end_reasons": "|".join(point.get("point_end_reasons") or []),
                    "point_review_flags": "|".join(point.get("point_review_flags") or []),
                    "point_winner": point.get("winner"),
                    "point_winner_player": point.get("winner_player"),
                    "shot_id": shot.get("id"),
                    "shot_frame": shot.get("frame"),
                    "shot_player": shot.get("player"),
                    "shot_type": shot.get("type"),
                    "stroke_side": shot.get("stroke_side"),
                    "stroke_confidence": shot.get("stroke_confidence"),
                    "stroke_reason": shot.get("stroke_reason"),
                    "speed_mph": speed.get("mph"),
                    "speed_quality": speed.get("quality"),
                    "speed_source": speed.get("source"),
                    "link_quality": link.get("quality"),
                }
            )


def export_summary(analysis, path):
    ensure_parent(path)
    summary = {
        "summary": analysis.get("summary", {}),
        "players": analysis.get("players", {}),
        "points": [
            {
                "index": point.get("index"),
                "frames": [point.get("start_frame"), point.get("end_frame")],
                "server": point.get("server"),
                "server_player": point.get("server_player"),
                "receiver": point.get("receiver"),
                "receiver_player": point.get("receiver_player"),
                "winner": point.get("winner"),
                "winner_player": point.get("winner_player"),
                "winner_source": point.get("winner_source"),
                "serve_status": point.get("serve_status"),
                "serve_state": point.get("serve_state"),
                "serve_confidence": point.get("serve_confidence"),
                "serve_reasons": point.get("serve_reasons"),
                "point_end_reason": point.get("point_end_reason"),
                "point_end_type": point.get("point_end_type"),
                "point_end_confidence": point.get("point_end_confidence"),
                "point_end_reasons": point.get("point_end_reasons"),
                "point_review_flags": point.get("point_review_flags"),
                "point_end_analysis": point.get("point_end_analysis"),
                "point_score_before": point.get("point_score_before"),
                "score_after": point.get("score_after"),
                "game_score_after": point.get("game_score_after"),
                "set_score_after": point.get("set_score_after"),
                "bounce_count": point.get("bounce_count"),
            }
            for point in analysis.get("points", [])
        ],
        "games": analysis.get("games", []),
        "excluded_bounces": analysis.get("excluded_bounces", []),
        "dead_ball_candidates": [
            bounce for bounce in analysis.get("bounces", []) if bounce.get("dead_ball_candidate")
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def world_to_image(point, margin=70, scale=10):
    xw, yw = point
    return (int(round(margin + xw * scale)), int(round(margin + yw * scale)))


def draw_line(image, p1, p2, color, thickness=2):
    cv2.line(image, world_to_image(p1), world_to_image(p2), color, thickness, cv2.LINE_AA)


def draw_court(image):
    white = (235, 235, 235)
    muted = (160, 160, 160)
    draw_line(image, (0, 0), (COURT_WIDTH_FT, 0), white)
    draw_line(image, (COURT_WIDTH_FT, 0), (COURT_WIDTH_FT, COURT_LENGTH_FT), white)
    draw_line(image, (COURT_WIDTH_FT, COURT_LENGTH_FT), (0, COURT_LENGTH_FT), white)
    draw_line(image, (0, COURT_LENGTH_FT), (0, 0), white)
    draw_line(image, (SINGLES_LEFT_FT, 0), (SINGLES_LEFT_FT, COURT_LENGTH_FT), muted)
    draw_line(image, (SINGLES_RIGHT_FT, 0), (SINGLES_RIGHT_FT, COURT_LENGTH_FT), muted)
    draw_line(image, (0, NET_Y_FT), (COURT_WIDTH_FT, NET_Y_FT), (220, 220, 220), 3)
    draw_line(image, (SINGLES_LEFT_FT, SERVICE_Y_FAR_FT), (SINGLES_RIGHT_FT, SERVICE_Y_FAR_FT), muted)
    draw_line(image, (SINGLES_LEFT_FT, SERVICE_Y_NEAR_FT), (SINGLES_RIGHT_FT, SERVICE_Y_NEAR_FT), muted)
    draw_line(image, (SERVICE_X_CENTER_FT, SERVICE_Y_FAR_FT), (SERVICE_X_CENTER_FT, SERVICE_Y_NEAR_FT), muted)


def bounce_color(bounce, serve_attempts):
    if not bounce.get("live"):
        return (120, 120, 120)
    if bounce.get("dead_ball_candidate"):
        return (0, 220, 255)
    attempt = serve_attempts.get(bounce.get("id"))
    if attempt and attempt.get("result") == "fault":
        return (0, 120, 255)
    if attempt and attempt.get("result") == "in":
        return (0, 210, 0)
    return (255, 80, 80)


def export_court_map(analysis, path):
    ensure_parent(path)
    margin = 70
    scale = 10
    legend_width = 180
    width = int(COURT_WIDTH_FT * scale + margin * 2 + legend_width)
    height = int(COURT_LENGTH_FT * scale + margin * 2)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 42, 35)
    draw_court(image)
    serve_attempts = {}
    for point in analysis.get("points", []):
        for attempt in (point.get("serve_analysis") or {}).get("attempts") or []:
            serve_attempts[attempt.get("bounce_id")] = attempt
    for bounce in analysis.get("bounces", []):
        world = bounce.get("world_point")
        if not world:
            continue
        x, y = world_to_image(world)
        color = bounce_color(bounce, serve_attempts)
        cv2.circle(image, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            bounce.get("id", "").replace("bounce_", ""),
            (x + 8, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    legend = [
        ("live rally", (255, 80, 80)),
        ("serve in", (0, 210, 0)),
        ("serve fault", (0, 120, 255)),
        ("dead candidate", (0, 220, 255)),
        ("excluded", (120, 120, 120)),
    ]
    legend_x = int(COURT_WIDTH_FT * scale + margin + 28)
    for idx, (label, color) in enumerate(legend):
        y = margin + 8 + idx * 28
        cv2.circle(image, (legend_x, y - 5), 6, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (legend_x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.imwrite(path, image)


def main():
    args = parse_args()
    analysis = load_json(args.analysis)
    export_csv(analysis, args.csv)
    export_summary(analysis, args.summary_json)
    export_court_map(analysis, args.court_map)
    print(f"wrote {args.csv}")
    print(f"wrote {args.summary_json}")
    print(f"wrote {args.court_map}")


if __name__ == "__main__":
    main()
