import argparse
import json
import os
from collections import deque

import cv2


SCORE_COLOR = (255, 255, 255)
PANEL_COLOR = (20, 24, 28)
SHOT_COLOR = (255, 0, 255)
FAULT_COLOR = (0, 120, 255)
TEXT_SHADOW = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Render tennis event analysis overlays onto an annotated video.")
    parser.add_argument("--video", required=True, help="Input video, typically ai9.3.avi.")
    parser.add_argument("--analysis", required=True, help="Analysis JSON from analyze_tennis_events.py.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--shot-label-frames", type=int, default=75, help="Frames to keep shot labels visible.")
    return parser.parse_args()


def load_analysis(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def open_writer(path, width, height, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for codec in ("MJPG", "mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def draw_text(frame, text, origin, scale=0.65, color=SCORE_COLOR, thickness=2):
    x, y = origin
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def point_for_frame(points, frame_index):
    active = None
    for point in points:
        if point["start_frame"] <= frame_index <= point["end_frame"]:
            return point
        if frame_index > point["end_frame"]:
            active = point
    return active


def score_before_point(points, point):
    if point is None:
        return "0-0"
    return point.get("point_score_before") or "0-0"


def point_score_for_display(point, show_result):
    if point is None:
        return "0-0"
    if show_result and point.get("score_after"):
        return point["score_after"]
    return point.get("point_score_before") or score_before_point([], point)


def match_scores_for_display(point, show_result):
    if point is None:
        return "Games 0-0  Sets 0-0"
    game_score = point.get("game_score_after") if show_result else point.get("game_score_before")
    set_score = point.get("set_score_after") if show_result else point.get("set_score_before")
    tiebreak = point.get("tiebreak_after") if show_result else point.get("tiebreak_before")
    prefix = "TB  " if tiebreak else ""
    return f"{prefix}Games {game_score or '0-0'}  Sets {set_score or '0-0'}"


def shot_label(shot):
    if shot.get("type") == "first_serve":
        return "1st serve"
    if shot.get("type") == "second_serve":
        return "2nd serve"
    stroke_confidence = shot.get("stroke_confidence")
    if stroke_confidence in {"medium", "high"}:
        stroke = {"forehand": "FH", "backhand": "BH", "body": "BODY"}.get(shot.get("stroke_side"), "SHOT")
    else:
        stroke = "SHOT"
    shot_type = {"opening_shot": "opening"}.get(shot.get("type"), shot.get("type") or "shot")
    speed_info = shot.get("speed", {})
    mph = speed_info.get("mph")
    show_speed = speed_info.get("quality") == "high"
    speed = f" {mph:.0f} mph" if show_speed and isinstance(mph, (int, float)) else ""
    return f"{stroke} {shot_type}{speed}"


def main():
    args = parse_args()
    analysis = load_analysis(args.analysis)
    shots_by_frame = {}
    fault_bounce_ids = set()
    shot_ids_by_fault_bounce = set()
    links_by_bounce_id = {
        link.get("bounce_id"): link for link in analysis.get("shot_bounce_links", []) if link.get("bounce_id")
    }
    for point in analysis.get("points", []):
        for attempt in (point.get("serve_analysis") or {}).get("attempts") or []:
            if attempt.get("result") == "fault":
                bounce_id = attempt.get("bounce_id")
                fault_bounce_ids.add(bounce_id)
                link = links_by_bounce_id.get(bounce_id)
                if link and link.get("shot_id"):
                    shot_ids_by_fault_bounce.add(link["shot_id"])
    for shot in analysis.get("shots", []):
        if shot.get("id") in shot_ids_by_fault_bounce:
            continue
        shots_by_frame.setdefault(int(shot["frame"]), []).append(shot)
    missed_bounce_candidates_by_frame = {}
    for candidate in analysis.get("missed_bounce_candidates", []):
        point = candidate.get("point")
        if not point:
            continue
        missed_bounce_candidates_by_frame.setdefault(int(candidate["frame"]), []).append(candidate)
    serve_labels_by_frame = {}
    for point in analysis.get("points", []):
        serve_analysis = point.get("serve_analysis") or {}
        attempts = serve_analysis.get("attempts") or []
        for attempt in attempts:
            # A serve can be located without its landing ever being seen (the
            # bounce falls in a gap in the ball track), in which case there is no
            # frame to hang a label on. Skip rather than crash the render.
            if attempt.get("bounce_frame") is None:
                continue
            frame = int(attempt["bounce_frame"])
            label = None
            if serve_analysis.get("status") == "double_fault" and attempt.get("attempt") == 2:
                label = "DOUBLE FAULT"
            elif serve_analysis.get("status") == "geometric_double_fault_played_out" and attempt.get("attempt") == 2:
                label = "FAULT? PLAYED"
            elif attempt.get("result") == "fault":
                label = "FAULT"
            elif attempt.get("result") == "in" and len(attempts) > 1:
                label = "2ND SERVE IN"
            if label:
                serve_labels_by_frame.setdefault(frame, []).append(
                    {
                        "label": label,
                        "point": attempt.get("world_point"),
                    }
                )
        terminal_ball = point.get("terminal_ball") or {}
        if point.get("point_end_reason") == "out" and terminal_ball.get("last_ball_frame"):
            serve_labels_by_frame.setdefault(int(terminal_ball["last_ball_frame"]), []).append(
                {
                    "label": "OUT",
                    "point": terminal_ball.get("last_center"),
                }
            )

    points = analysis.get("points", [])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = open_writer(args.output, width, height, fps)
    if writer is None:
        raise RuntimeError(f"Could not open output writer: {args.output}")

    active_labels = deque()
    active_serve_labels = deque()
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        for shot in shots_by_frame.get(frame_index, []):
            point = shot.get("point")
            if point:
                active_labels.append(
                    {
                        "expires": frame_index + args.shot_label_frames,
                        "point": (int(round(point[0])), int(round(point[1]))),
                        "label": shot_label(shot),
                    }
                )
        for candidate in missed_bounce_candidates_by_frame.get(frame_index, []):
            point = candidate.get("point")
            if point:
                active_labels.append(
                    {
                        "expires": frame_index + args.shot_label_frames,
                        "point": (int(round(point[0])), int(round(point[1]))),
                        "label": f"B? {candidate.get('confidence', 'low')}",
                        "color": (255, 0, 255),
                    }
                )
        for item in serve_labels_by_frame.get(frame_index, []):
            active_serve_labels.append({"expires": frame_index + 70, "label": item["label"]})

        while active_labels and active_labels[0]["expires"] < frame_index:
            active_labels.popleft()
        while active_serve_labels and active_serve_labels[0]["expires"] < frame_index:
            active_serve_labels.popleft()

        active_point = point_for_frame(points, frame_index)
        point_label = f"Point {active_point['index']}" if active_point else "Point"
        show_result = active_point and frame_index >= active_point["end_frame"] - 20 and active_point.get("score_after")
        current_score = point_score_for_display(active_point, show_result)
        match_score = match_scores_for_display(active_point, show_result)

        cv2.rectangle(frame, (18, 58), (430, 156), PANEL_COLOR, -1)
        cv2.rectangle(frame, (18, 58), (430, 156), (90, 100, 110), 1)
        draw_text(frame, f"{point_label}  {current_score}", (34, 90), scale=0.82)
        draw_text(frame, match_score, (34, 118), scale=0.56, color=(220, 230, 240))
        if show_result:
            reason = active_point.get("point_end_reason")
            end_type = active_point.get("point_end_type")
            confident_end_type = active_point.get("point_end_confidence") in {"medium", "high"}
            result_text = f"Point won: {active_point['winner']}"
            if reason == "double_fault":
                result_text = "Double fault"
            elif confident_end_type and end_type == "forced_error_out":
                result_text = f"Won: {active_point['winner']} (forced out)"
            elif confident_end_type and end_type == "unforced_error_out":
                result_text = f"Won: {active_point['winner']} (unforced out)"
            elif confident_end_type and end_type == "net_error":
                result_text = f"Won: {active_point['winner']} (net)"
            elif reason == "out":
                result_text = f"Won: {active_point['winner']} (out)"
            elif active_point.get("serve_status") == "geometric_double_fault_played_out":
                result_text = f"Played out: {active_point['winner']}"
            draw_text(frame, result_text, (34, 146), scale=0.54, color=(220, 230, 240))
        else:
            draw_text(frame, "Shot analysis", (34, 146), scale=0.54, color=(220, 230, 240))

        for index, item in enumerate(active_serve_labels):
            draw_text(frame, item["label"], (34, 188 + (index * 32)), scale=0.95, color=FAULT_COLOR, thickness=3)

        for item in active_labels:
            x, y = item["point"]
            y = max(32, min(height - 24, y - 24))
            label_width = max(180, min(460, len(item["label"]) * 14))
            x = max(16, min(width - label_width, x + 12))
            draw_text(frame, item["label"], (x, y), scale=0.56, color=item.get("color", SHOT_COLOR), thickness=2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"wrote {args.output}: {frame_index} frames")


if __name__ == "__main__":
    main()
