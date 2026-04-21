import argparse
import json
import math
import os
import statistics
from collections import deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PLAYER_CLASS_ID = 0
BALL_COLOR = (0, 255, 255)
PLAYER_NEAR_COLOR = (255, 128, 0)
PLAYER_FAR_COLOR = (0, 200, 255)
GENERIC_COLOR = (0, 255, 0)
BOUNCE_COLOR = (0, 0, 255)
HIT_COLOR = (255, 0, 255)


class SimpleTracker:
    def __init__(self, max_distance):
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections):
        updated_tracks = {}
        used_track_ids = set()

        for det in detections:
            best_track_id = None
            best_distance = None
            cx, cy = det["center"]

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                if track["class_id"] != det["class_id"]:
                    continue

                tx, ty = track["center"]
                distance = math.hypot(cx - tx, cy - ty)
                if distance > self.max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1

            det["track_id"] = best_track_id
            updated_tracks[best_track_id] = {
                "class_id": det["class_id"],
                "center": det["center"],
            }
            used_track_ids.add(best_track_id)

        self.tracks = updated_tracks
        return detections


class StationaryTrackFilter:
    def __init__(self, movement_px, static_frames):
        self.movement_px = movement_px
        self.static_frames = static_frames
        self.states = {}

    def filter(self, detections):
        active_ids = set()
        filtered = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                filtered.append(det)
                continue

            active_ids.add(track_id)
            center = det["center"]
            state = self.states.get(track_id)
            if state is None:
                self.states[track_id] = {"center": center, "static_count": 0}
                filtered.append(det)
                continue

            distance = math.hypot(center[0] - state["center"][0], center[1] - state["center"][1])
            static_count = state["static_count"] + 1 if distance <= self.movement_px else 0
            self.states[track_id] = {"center": center, "static_count": static_count}
            if static_count < self.static_frames:
                filtered.append(det)

        self.states = {track_id: self.states[track_id] for track_id in active_ids if track_id in self.states}
        return filtered


class MovingBallFilter:
    def __init__(self, history_frames, min_travel_px):
        self.history_frames = history_frames
        self.min_travel_px = min_travel_px
        self.histories = {}

    def filter(self, detections):
        active_ids = set()
        filtered = []

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            active_ids.add(track_id)
            history = self.histories.setdefault(track_id, deque(maxlen=self.history_frames))
            history.append(tuple(det["center"]))

            if len(history) < 2:
                continue

            start_x, start_y = history[0]
            end_x, end_y = history[-1]
            travel = math.hypot(end_x - start_x, end_y - start_y)
            if travel >= self.min_travel_px:
                filtered.append(det)

        self.histories = {track_id: self.histories[track_id] for track_id in active_ids if track_id in self.histories}
        return filtered


def point_to_bbox_distance(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


def average_point(points):
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    x = sum(point[0] for point in valid) / len(valid)
    y = sum(point[1] for point in valid) / len(valid)
    return (x, y)


def ball_contact_point(ball):
    if not ball:
        return None
    bbox = ball.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, _y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))
    center = ball.get("center")
    if isinstance(center, list) and len(center) == 2:
        return (float(center[0]), float(center[1]))
    return None


def order_court_corners(points):
    points = sorted(points, key=lambda p: p[1])
    far = points[:2]
    near = points[2:]
    far_left, far_right = sorted(far, key=lambda p: p[0])
    near_left, near_right = sorted(near, key=lambda p: p[0])
    return [near_left, near_right, far_right, far_left]


def get_mini_court_layout(frame_shape, size, margin):
    frame_h, frame_w = frame_shape[:2]
    court_len = 78.0
    court_wid = 36.0
    singles_wid = 27.0
    aspect = court_len / court_wid
    overlay_w = max(120, int(size))
    overlay_h = int(overlay_w * aspect)

    if overlay_h + margin * 2 > frame_h:
        overlay_h = max(160, frame_h - margin * 2)
        overlay_w = int(overlay_h / aspect)
    if overlay_w + margin * 2 > frame_w:
        overlay_w = max(120, frame_w - margin * 2)
        overlay_h = int(overlay_w * aspect)

    x1 = max(0, frame_w - overlay_w - margin)
    y1 = margin
    x2 = min(frame_w - 1, x1 + overlay_w)
    y2 = min(frame_h - 1, y1 + overlay_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "overlay_w": x2 - x1,
        "overlay_h": y2 - y1,
        "court_len": court_len,
        "court_wid": court_wid,
        "singles_wid": singles_wid,
    }


def mini_court_point(layout, xw, yw):
    px = int(layout["x1"] + (xw / layout["court_wid"]) * layout["overlay_w"])
    py = int(layout["y1"] + (yw / layout["court_len"]) * layout["overlay_h"])
    return (px, py)


def draw_mini_court(frame, enabled, size, margin):
    if not enabled:
        return None

    layout = get_mini_court_layout(frame.shape, size=size, margin=margin)
    if layout is None:
        return None

    x1 = layout["x1"]
    y1 = layout["y1"]
    x2 = layout["x2"]
    y2 = layout["y2"]
    overlay_w = layout["overlay_w"]
    overlay_h = layout["overlay_h"]
    panel = frame[y1:y2, x1:x2]
    if panel.size == 0:
        return None

    tint = panel.copy()
    tint[:] = (24, 42, 18)
    cv2.addWeighted(tint, 0.72, panel, 0.28, 0, panel)

    color = (235, 245, 235)
    thick = 1 if overlay_w < 180 else 2

    def pt(xw, yl):
        return mini_court_point(layout, xw, yl)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 220, 200), 1)

    court_len = layout["court_len"]
    court_wid = layout["court_wid"]
    singles_wid = layout["singles_wid"]
    singles_left = (court_wid - singles_wid) / 2.0
    singles_right = singles_left + singles_wid
    service_y_top = 21.0
    service_y_bottom = court_len - 21.0
    net_y = court_len / 2.0
    center_x = court_wid / 2.0

    cv2.rectangle(frame, pt(0.0, 0.0), pt(court_wid, court_len), color, thick)
    cv2.rectangle(frame, pt(singles_left, 0), pt(singles_right, court_len), color, thick)
    cv2.line(frame, pt(singles_left, service_y_top), pt(singles_right, service_y_top), color, thick)
    cv2.line(frame, pt(singles_left, service_y_bottom), pt(singles_right, service_y_bottom), color, thick)
    cv2.line(frame, pt(singles_left, net_y), pt(singles_right, net_y), color, thick)
    cv2.line(frame, pt(center_x, service_y_top), pt(center_x, service_y_bottom), color, thick)

    mark_len = 2.0
    cv2.line(frame, pt(center_x, 0), pt(center_x, mark_len), color, thick)
    cv2.line(frame, pt(center_x, court_len - mark_len), pt(center_x, court_len), color, thick)
    return layout


def draw_mini_court_points(frame, layout, ball_world_point, bounce_world_points):
    if layout is None:
        return

    def in_bounds(world_point):
        if world_point is None:
            return False
        xw, yw = world_point
        return 0.0 <= xw <= layout["court_wid"] and 0.0 <= yw <= layout["court_len"]

    for world_point in bounce_world_points[-8:]:
        if not in_bounds(world_point):
            continue
        point = mini_court_point(layout, world_point[0], world_point[1])
        cv2.circle(frame, point, 4, BOUNCE_COLOR, -1)
        cv2.circle(frame, point, 7, (255, 255, 255), 1)

    if in_bounds(ball_world_point):
        point = mini_court_point(layout, ball_world_point[0], ball_world_point[1])
        cv2.circle(frame, point, 6, BALL_COLOR, -1)
        cv2.circle(frame, point, 9, (40, 40, 40), 1)


def load_court_calibration(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    points = data.get("points")
    net_points = data.get("net_points")
    if not isinstance(points, list) or len(points) != 4:
        return None

    try:
        court_points = order_court_corners([(float(x), float(y)) for x, y in points])
        net = None
        if isinstance(net_points, list) and len(net_points) == 2:
            net = [(float(x), float(y)) for x, y in net_points]
    except Exception:
        return None
    return {"points": court_points, "net_points": net}


def build_inverse_court_homography(court_calib):
    if not court_calib or court_calib.get("points") is None:
        return None
    # Calibration points are stored as near-left, near-right, far-right, far-left.
    # Map them to world coordinates with the far baseline at y=0 and the near
    # baseline at y=78 so near/far classification and the mini-court share the
    # same orientation.
    world = np.array([[0.0, 78.0], [36.0, 78.0], [36.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    image = np.array(court_calib["points"], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(world, image)
    try:
        return np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None


def calibration_fits_frame(court_calib, frame_shape, margin_px=24):
    if not court_calib:
        return False
    frame_h, frame_w = frame_shape[:2]
    all_points = list(court_calib.get("points") or [])
    all_points.extend(court_calib.get("net_points") or [])
    if not all_points:
        return False
    for x, y in all_points:
        if x < -margin_px or x > frame_w + margin_px or y < -margin_px or y > frame_h + margin_px:
            return False
    return True


def project_to_court_world(center, inv_homography):
    if center is None or inv_homography is None:
        return None
    point = np.array([[center]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, inv_homography)[0][0]
    return float(world[0]), float(world[1])


def world_point_in_court(world_point, margin=0.0):
    if world_point is None:
        return False
    xw, yw = world_point
    return -margin <= xw <= 36.0 + margin and -margin <= yw <= 78.0 + margin


class EventDetector:
    def __init__(
        self,
        bounce_min_vertical_change,
        bounce_min_gap_frames,
        bounce_x_margin_ratio,
        bounce_y_margin_ratio,
        player_hit_margin_px,
        racket_hit_margin_px,
        min_event_travel,
        player_hit_upper_body_ratio,
        hit_min_gap_frames,
        hit_min_angle_change_deg,
        hit_min_speed_change_ratio,
        bounce_min_y_ratio,
        inv_court_homography,
        fallback_net_y,
    ):
        self.bounce_min_vertical_change = bounce_min_vertical_change
        self.bounce_min_gap_frames = bounce_min_gap_frames
        self.bounce_x_margin_ratio = bounce_x_margin_ratio
        self.bounce_y_margin_ratio = bounce_y_margin_ratio
        self.player_hit_margin_px = player_hit_margin_px
        self.racket_hit_margin_px = racket_hit_margin_px
        self.min_event_travel = min_event_travel
        self.player_hit_upper_body_ratio = player_hit_upper_body_ratio
        self.hit_min_gap_frames = hit_min_gap_frames
        self.hit_min_angle_change_deg = hit_min_angle_change_deg
        self.hit_min_speed_change_ratio = hit_min_speed_change_ratio
        self.bounce_min_y_ratio = bounce_min_y_ratio
        self.inv_court_homography = inv_court_homography
        self.fallback_net_y = fallback_net_y
        self.history = deque(maxlen=11)
        self.last_bounce_frame = -10**9
        self.events = []
        self.bounces = []
        self.event_frames = set()

    def update(self, frame_index, ball, frame_shape, players, rackets):
        self.history.append(
            {
                "frame": frame_index,
                "ball": tuple(ball["center"]) if ball is not None else None,
                "ball_contact": ball_contact_point(ball),
                "ball_track_id": ball.get("track_id") if ball is not None else None,
                "players": [player for player in players if player is not None],
                "rackets": [racket for racket in rackets if racket is not None],
            }
        )

        if len(self.history) < 9:
            return None

        candidate_index = len(self.history) // 2
        event = self._evaluate_candidate(candidate_index, frame_shape)
        return event

    def _evaluate_candidate(self, index, frame_shape):
        history_items = list(self.history)
        center_item = history_items[index]
        candidate_frame = center_item["frame"]
        if candidate_frame in self.event_frames:
            return None

        prev_items = history_items[max(0, index - 4) : index]
        next_items = history_items[index + 1 : index + 5]
        if len(prev_items) < 3 or len(next_items) < 3:
            return None

        prev_points = [item["ball"] for item in prev_items]
        next_points = [item["ball"] for item in next_items]
        prev_contact_points = [item["ball_contact"] for item in prev_items]
        next_contact_points = [item["ball_contact"] for item in next_items]
        prev_valid = [point for point in prev_points if point is not None]
        next_valid = [point for point in next_points if point is not None]
        if len(prev_valid) < 2 or len(next_valid) < 2:
            return None
        if self._distinct_point_count(prev_valid) < 3 or self._distinct_point_count(next_valid) < 3:
            return None

        p0 = average_point(prev_points)
        p1 = center_item["ball"]
        p2 = average_point(next_points)
        if p0 is None or p1 is None or p2 is None:
            return None

        prev_close = prev_valid[-1]
        next_close = next_valid[0]
        if prev_close is None or next_close is None:
            return None

        contact_point = center_item["ball_contact"]
        prev_contact_close = next((point for point in reversed(prev_contact_points) if point is not None), None)
        next_contact_close = next((point for point in next_contact_points if point is not None), None)

        in_vec = (p1[0] - p0[0], p1[1] - p0[1])
        out_vec = (p2[0] - p1[0], p2[1] - p1[1])
        in_speed = math.hypot(in_vec[0], in_vec[1])
        out_speed = math.hypot(out_vec[0], out_vec[1])
        if in_speed < self.min_event_travel or out_speed < self.min_event_travel:
            return None

        bounce_event = self._detect_bounce(
            candidate_frame=candidate_frame,
            point=p1,
            frame_shape=frame_shape,
            in_vec=in_vec,
            out_vec=out_vec,
            prev_points=prev_valid,
            next_points=next_valid,
            players=center_item["players"],
            rackets=center_item["rackets"],
            prev_close=prev_close,
            next_close=next_close,
            contact_point=contact_point,
            prev_contact_close=prev_contact_close,
            next_contact_close=next_contact_close,
        )
        if bounce_event is not None:
            return bounce_event
        return None

    def _detect_bounce(
        self,
        candidate_frame,
        point,
        frame_shape,
        in_vec,
        out_vec,
        prev_points,
        next_points,
        players,
        rackets,
        prev_close,
        next_close,
        contact_point,
        prev_contact_close,
        next_contact_close,
    ):
        if candidate_frame - self.last_bounce_frame < self.bounce_min_gap_frames:
            return None

        if self._trajectory_has_outlier(prev_points, point, next_points):
            return None

        contact_point = contact_point or point
        world_point = project_to_court_world(contact_point, self.inv_court_homography)
        side = self._classify_court_side(point, world_point)
        if world_point is not None:
            court_margin = 0.75
            if not world_point_in_court(world_point, margin=court_margin):
                return None
        frame_h, frame_w = frame_shape[:2]
        x_margin = int(frame_w * self.bounce_x_margin_ratio)
        y_margin = int(frame_h * self.bounce_y_margin_ratio)
        x, y = contact_point
        if x < x_margin or x > frame_w - x_margin:
            return None
        min_y_ratio = self.bounce_min_y_ratio if side == "near" else self.bounce_min_y_ratio * 0.45
        if y < max(y_margin, frame_h * min_y_ratio) or y > frame_h - y_margin:
            return None

        if self._near_contact_zone(contact_point, players, rackets, side):
            return None

        prev_avg_y = sum(ball_point[1] for ball_point in prev_points) / len(prev_points)
        next_avg_y = sum(ball_point[1] for ball_point in next_points) / len(next_points)
        pre_drop = y - prev_avg_y
        post_rise = y - next_avg_y
        close_drop = y - prev_close[1]
        close_rise = y - next_close[1]
        pre_rise = prev_avg_y - y
        close_pre_rise = prev_close[1] - y

        dy_in = in_vec[1]
        dy_out = out_vec[1]
        vertical_change = dy_in - dy_out
        speed_in = math.hypot(in_vec[0], in_vec[1])
        speed_out = math.hypot(out_vec[0], out_vec[1])
        travel_threshold = self.min_event_travel if side == "near" else self.min_event_travel * 0.4
        vertical_threshold = self.bounce_min_vertical_change if side == "near" else self.bounce_min_vertical_change * 0.4
        far_margin = max(2.0, vertical_threshold * 0.35)
        if speed_in < travel_threshold or speed_out < travel_threshold:
            return None
        pattern = "near_rebound"
        if side == "near":
            if dy_in <= -far_margin or dy_out >= far_margin:
                return None
            bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
            if bounce_strength < vertical_threshold:
                return None
            pre_post_threshold = vertical_threshold * 0.7
            if pre_drop < pre_post_threshold or post_rise < pre_post_threshold:
                return None
            close_threshold = vertical_threshold * 0.25
            if close_drop < close_threshold or close_rise < close_threshold:
                return None
            if y < prev_close[1] or y < next_close[1]:
                return None
        else:
            far_inflection = dy_in < -far_margin and dy_out < far_margin and (dy_out - dy_in) >= vertical_threshold
            far_rebound = dy_in > -far_margin and dy_out < far_margin
            if not far_inflection and not far_rebound:
                return None
            if far_inflection:
                pattern = "far_inflection"
                bounce_strength = max(dy_out - dy_in, pre_rise, post_rise, close_pre_rise + close_rise)
                if pre_rise < vertical_threshold * 1.4 or post_rise < vertical_threshold * 0.7:
                    return None
                if close_pre_rise < vertical_threshold * 0.3 or close_rise < vertical_threshold * 0.15:
                    return None
            else:
                pattern = "far_rebound"
                bounce_strength = max(vertical_change, pre_drop + post_rise, close_drop + close_rise)
                if pre_drop < vertical_threshold * 0.45 or post_rise < vertical_threshold * 0.45:
                    return None
                if close_drop < vertical_threshold * 0.10 or close_rise < vertical_threshold * 0.10:
                    return None

        x_direction_consistent = (in_vec[0] == 0) or (out_vec[0] == 0) or (in_vec[0] * out_vec[0] >= 0)
        if not x_direction_consistent and abs(in_vec[0] - out_vec[0]) > max(abs(vertical_change), speed_in * 0.7):
            return None

        event = {
            "frame": candidate_frame,
            "point": [int(round(x)), int(round(y))],
            "type": "bounce",
            "side": side,
            "pattern": pattern,
            "vertical_change": round(vertical_change, 1),
            "bounce_strength": round(bounce_strength, 1),
            "pre_drop": round(pre_drop, 1),
            "post_rise": round(post_rise, 1),
            "world_point": [round(world_point[0], 2), round(world_point[1], 2)] if world_point is not None else None,
        }
        self._record_event(event)
        self.last_bounce_frame = candidate_frame
        return event

    def _classify_court_side(self, point, world_point):
        if world_point is not None:
            return "far" if world_point[1] < 39.0 else "near"
        if self.fallback_net_y is not None:
            return "far" if point[1] < self.fallback_net_y else "near"
        return "far" if point[1] < 540 else "near"

    def _near_contact_zone(self, point, players, rackets, side):
        player_x_margin = max(16.0, self.player_hit_margin_px * (2.5 if side == "near" else 1.4))
        player_y_margin = max(6.0, self.player_hit_margin_px * (0.8 if side == "near" else 0.6))
        racket_margin = self.racket_hit_margin_px * (1.5 if side == "near" else 1.0)

        for player in players:
            x1, y1, x2, y2 = player["bbox"]
            height = y2 - y1
            upper_y1 = y1 + (height * 0.08)
            upper_y2 = y1 + (height * self.player_hit_upper_body_ratio)
            zone = [
                x1 - player_x_margin,
                upper_y1 - player_y_margin,
                x2 + player_x_margin,
                upper_y2 + player_y_margin,
            ]
            if point_to_bbox_distance(point, zone) <= 0:
                return True
        for racket in rackets:
            if point_to_bbox_distance(point, racket["bbox"]) <= racket_margin:
                return True
        return False

    def _trajectory_has_outlier(self, prev_points, point, next_points):
        points = list(prev_points) + [point] + list(next_points)
        steps = []
        for p0, p1 in zip(points, points[1:]):
            if p0 is None or p1 is None:
                continue
            steps.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if len(steps) < 4:
            return False
        median_step = statistics.median(steps)
        if median_step <= 0:
            return False
        max_step = max(steps)
        return max_step > 72.0 and max_step > median_step * 1.7

    def _distinct_point_count(self, points):
        distinct = set()
        for point in points:
            if point is None:
                continue
            distinct.add((int(round(point[0])), int(round(point[1]))))
        return len(distinct)

    def _record_event(self, event):
        self.event_frames.add(event["frame"])
        self.events.append(event)
        self.bounces.append(event)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-only tennis tracking for the ball, players, and other scene objects."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", help="Optional annotated output video path.")
    parser.add_argument(
        "--court-calib-file",
        default="court_calib.json",
        help="Court calibration file path used for side-aware bounce rules.",
    )
    parser.add_argument(
        "--ball-model",
        default="vids/models/tennisball.pt",
        help="YOLO model path for the tennis ball detector.",
    )
    parser.add_argument(
        "--scene-model",
        default="yolov8n.pt",
        help="YOLO model path for players and other scene objects.",
    )
    parser.add_argument("--ball-conf", type=float, default=0.15, help="Ball confidence threshold.")
    parser.add_argument("--scene-conf", type=float, default=0.25, help="Scene confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size.")
    parser.add_argument("--device", default="", help="YOLO device, such as cpu, mps, 0.")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config name or path.",
    )
    parser.add_argument(
        "--ball-max-distance",
        type=float,
        default=120.0,
        help="Maximum frame-to-frame distance for ball ID matching.",
    )
    parser.add_argument(
        "--object-max-distance",
        type=float,
        default=180.0,
        help="Maximum frame-to-frame distance for player/object ID matching.",
    )
    parser.add_argument(
        "--ball-stationary-px",
        type=float,
        default=6.0,
        help="Treat a ball track as stationary if it moves less than this many pixels.",
    )
    parser.add_argument(
        "--ball-stationary-frames",
        type=int,
        default=10,
        help="Hide ball tracks that stay stationary for this many frames.",
    )
    parser.add_argument(
        "--ball-motion-history",
        type=int,
        default=5,
        help="Frames of history used to decide whether a ball is moving.",
    )
    parser.add_argument(
        "--ball-min-travel",
        type=float,
        default=12.0,
        help="Minimum travel in pixels before a ball track is considered moving.",
    )
    parser.add_argument(
        "--far-ball-roi-height",
        type=float,
        default=0.58,
        help="Fraction of frame height used for a high-resolution far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-roi-width",
        type=float,
        default=0.82,
        help="Fraction of frame width used for a high-resolution far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-conf",
        type=float,
        default=0.10,
        help="Confidence threshold for the far-court ball pass.",
    )
    parser.add_argument(
        "--far-ball-imgsz",
        type=int,
        default=1600,
        help="Inference size for the far-court ball pass.",
    )
    parser.add_argument(
        "--bounce-min-vertical-change",
        type=float,
        default=12.0,
        help="Minimum vertical direction change in pixels to count as a bounce.",
    )
    parser.add_argument(
        "--bounce-min-gap-frames",
        type=int,
        default=8,
        help="Minimum frames between bounce markers.",
    )
    parser.add_argument(
        "--bounce-x-margin-ratio",
        type=float,
        default=0.06,
        help="Ignore bounce candidates too close to the left or right frame edge.",
    )
    parser.add_argument(
        "--bounce-y-margin-ratio",
        type=float,
        default=0.08,
        help="Ignore bounce candidates too close to the top or bottom frame edge.",
    )
    parser.add_argument(
        "--player-hit-margin-px",
        type=int,
        default=8,
        help="Fallback player-box margin for classifying an event as a hit.",
    )
    parser.add_argument(
        "--racket-hit-margin-px",
        type=int,
        default=20,
        help="Classify an event near a racket box as a hit.",
    )
    parser.add_argument(
        "--event-min-travel",
        type=float,
        default=12.0,
        help="Minimum pre/post event travel in pixels required before creating a hit or bounce.",
    )
    parser.add_argument(
        "--player-hit-upper-body-ratio",
        type=float,
        default=0.50,
        help="Only the upper portion of a player box is eligible for hit classification.",
    )
    parser.add_argument(
        "--hit-min-gap-frames",
        type=int,
        default=10,
        help="Minimum frames between hit markers.",
    )
    parser.add_argument(
        "--hit-min-angle-change-deg",
        type=float,
        default=28.0,
        help="Minimum path angle change for confirming a hit near a player or racket.",
    )
    parser.add_argument(
        "--hit-min-speed-change-ratio",
        type=float,
        default=0.22,
        help="Minimum speed change ratio for confirming a hit near a player or racket.",
    )
    parser.add_argument(
        "--bounce-min-y-ratio",
        type=float,
        default=0.22,
        help="Ignore bounce candidates that are too high in the frame.",
    )
    parser.add_argument(
        "--no-court-overlay",
        action="store_true",
        help="Disable the mini singles-court overlay.",
    )
    parser.add_argument(
        "--court-overlay-size",
        type=int,
        default=180,
        help="Mini court overlay width in pixels.",
    )
    parser.add_argument(
        "--court-overlay-margin",
        type=int,
        default=12,
        help="Mini court overlay margin in pixels.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=20,
        help="Ball trail length in frames.",
    )
    parser.add_argument(
        "--hide-other-objects",
        action="store_true",
        help="Only draw the tennis ball and the two players.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the preview window.")
    parser.add_argument(
        "--log-jsonl",
        help="Optional JSONL output with per-frame ball, player, and scene detections.",
    )
    return parser.parse_args()


def load_model(path):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install it with: pip install ultralytics")
    return YOLO(path)


def open_writer(path, width, height, fps):
    candidates = ["mp4v", "avc1", "MJPG"]
    for codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def run_track(model, frame, conf, imgsz, device, tracker, classes=None):
    device_arg = device or None
    results = model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        device=device_arg,
        verbose=False,
        classes=classes,
    )
    return results[0] if results else None


def run_ball_detection(model, frame, conf, imgsz, device):
    result = run_track(model, frame, conf=conf, imgsz=imgsz, device=device, tracker=None)
    return extract_detections(result)


def offset_detections(detections, offset_x, offset_y):
    shifted = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        shifted.append(
            {
                **det,
                "bbox": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "center": [det["center"][0] + offset_x, det["center"][1] + offset_y],
            }
        )
    return shifted


def dedupe_ball_detections(detections, center_distance=12.0):
    kept = []
    ordered = sorted(detections, key=lambda det: det.get("conf") or 0.0, reverse=True)
    for det in ordered:
        cx, cy = det["center"]
        duplicate = False
        for kept_det in kept:
            kx, ky = kept_det["center"]
            if math.hypot(cx - kx, cy - ky) <= center_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def extract_detections(result):
    detections = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names or {}
    boxes = result.boxes
    classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [None] * len(boxes)
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [None] * len(boxes)
    coords = boxes.xyxy.cpu().tolist()

    for i, xyxy in enumerate(coords):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_id = classes[i]
        detections.append(
            {
                "track_id": None,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "conf": float(confs[i]) if confs[i] is not None else None,
                "bbox": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            }
        )
    return detections


def detect_ball_candidates(frame, model, args):
    detections = run_ball_detection(
        model,
        frame,
        conf=args.ball_conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    if args.far_ball_roi_height > 0 and args.far_ball_roi_width > 0:
        frame_h, frame_w = frame.shape[:2]
        roi_h = max(1, min(frame_h, int(frame_h * args.far_ball_roi_height)))
        roi_w = max(1, min(frame_w, int(frame_w * args.far_ball_roi_width)))
        roi_x1 = max(0, min(frame_w - roi_w, int((frame_w - roi_w) * 0.5)))
        roi_y1 = 0
        roi = frame[roi_y1 : roi_y1 + roi_h, roi_x1 : roi_x1 + roi_w]
        roi_detections = run_ball_detection(
            model,
            roi,
            conf=args.far_ball_conf,
            imgsz=args.far_ball_imgsz,
            device=args.device,
        )
        detections.extend(offset_detections(roi_detections, roi_x1, roi_y1))

    return dedupe_ball_detections(detections)


def select_ball(ball_detections):
    if not ball_detections:
        return None
    return max(ball_detections, key=lambda det: (det["conf"] or 0.0, det["bbox"][3] - det["bbox"][1]))


def split_players(scene_detections):
    players = [det for det in scene_detections if det["class_id"] == PLAYER_CLASS_ID]
    if not players:
        return None, None, []

    players.sort(key=lambda det: det["center"][1])
    far_player = players[0]
    near_player = players[-1] if len(players) > 1 else None
    selected = [far_player]
    if near_player is not None and near_player is not far_player:
        selected.append(near_player)
    others = [det for det in scene_detections if det not in selected]
    return near_player, far_player, others


def draw_box(frame, detection, color, label):
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ball(frame, detection, trail):
    x1, y1, x2, y2 = detection["bbox"]
    center = tuple(detection["center"])
    radius = max(3, int(min(x2 - x1, y2 - y1) / 2))
    cv2.circle(frame, center, radius, BALL_COLOR, 2)
    cv2.putText(
        frame,
        f"Ball {format_track_suffix(detection)}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BALL_COLOR,
        2,
        cv2.LINE_AA,
    )

    trail.appendleft(center)
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        thickness = max(1, 5 - int(i / 4))
        cv2.line(frame, trail[i - 1], trail[i], BALL_COLOR, thickness)


def draw_events(frame, bounces):
    for i, bounce in enumerate(bounces, start=1):
        x, y = bounce["point"]
        cv2.drawMarker(
            frame,
            (x, y),
            BOUNCE_COLOR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Bounce {i}",
            (x + 8, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOUNCE_COLOR,
            2,
            cv2.LINE_AA,
        )


def format_track_suffix(detection):
    track_id = detection.get("track_id")
    return f"#{track_id}" if track_id is not None else ""


def write_frame_log(handle, frame_index, ball, near_player, far_player, scene_detections, event):
    row = {
        "frame": frame_index,
        "ball": ball,
        "event": event,
        "player_near": near_player,
        "player_far": far_player,
        "scene": scene_detections,
    }
    handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.ball_model):
        raise FileNotFoundError(f"Ball model not found: {args.ball_model}")

    ball_model = load_model(args.ball_model)
    scene_model = load_model(args.scene_model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    court_calib = load_court_calibration(args.court_calib_file)
    if court_calib is not None and not calibration_fits_frame(court_calib, (height, width, 3)):
        print(
            f"Warning: ignoring court calibration {args.court_calib_file} because it does not fit "
            f"the video frame {width}x{height}."
        )
        court_calib = None

    inv_court_homography = build_inverse_court_homography(court_calib)
    fallback_net_y = None
    if court_calib is not None and court_calib.get("net_points") is not None:
        net_points = court_calib["net_points"]
        fallback_net_y = (net_points[0][1] + net_points[1][1]) / 2.0

    writer = None
    if args.output:
        writer = open_writer(args.output, width, height, fps)
        if writer is None:
            raise RuntimeError(f"Could not open output writer: {args.output}")

    log_handle = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None
    ball_trail = deque(maxlen=max(1, args.trail))
    ball_tracker = SimpleTracker(max_distance=args.ball_max_distance)
    ball_filter = StationaryTrackFilter(
        movement_px=args.ball_stationary_px,
        static_frames=max(1, args.ball_stationary_frames),
    )
    moving_ball_filter = MovingBallFilter(
        history_frames=max(2, args.ball_motion_history),
        min_travel_px=args.ball_min_travel,
    )
    event_detector = EventDetector(
        bounce_min_vertical_change=args.bounce_min_vertical_change,
        bounce_min_gap_frames=max(1, args.bounce_min_gap_frames),
        bounce_x_margin_ratio=args.bounce_x_margin_ratio,
        bounce_y_margin_ratio=args.bounce_y_margin_ratio,
        player_hit_margin_px=max(0, args.player_hit_margin_px),
        racket_hit_margin_px=max(0, args.racket_hit_margin_px),
        min_event_travel=args.event_min_travel,
        player_hit_upper_body_ratio=args.player_hit_upper_body_ratio,
        hit_min_gap_frames=max(1, args.hit_min_gap_frames),
        hit_min_angle_change_deg=max(0.0, args.hit_min_angle_change_deg),
        hit_min_speed_change_ratio=max(0.0, args.hit_min_speed_change_ratio),
        bounce_min_y_ratio=max(0.0, min(1.0, args.bounce_min_y_ratio)),
        inv_court_homography=inv_court_homography,
        fallback_net_y=fallback_net_y,
    )
    scene_tracker = SimpleTracker(max_distance=args.object_max_distance)
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        scene_result = run_track(
            scene_model,
            frame,
            conf=args.scene_conf,
            imgsz=args.imgsz,
            device=args.device,
            tracker=args.tracker,
        )
        scene_detections = scene_tracker.update(extract_detections(scene_result))
        ball_detections = ball_tracker.update(detect_ball_candidates(frame, ball_model, args))
        ball_detections = ball_filter.filter(ball_detections)
        ball_detections = moving_ball_filter.filter(ball_detections)
        ball = select_ball(ball_detections)
        near_player, far_player, other_objects = split_players(scene_detections)
        racket_detections = [det for det in scene_detections if det["class_name"] == "tennis racket"]
        event = event_detector.update(
            frame_index,
            ball,
            frame.shape,
            [far_player, near_player],
            racket_detections,
        )

        if far_player is not None:
            draw_box(frame, far_player, PLAYER_FAR_COLOR, f"Player Far {format_track_suffix(far_player)}".strip())
        if near_player is not None:
            draw_box(frame, near_player, PLAYER_NEAR_COLOR, f"Player Near {format_track_suffix(near_player)}".strip())
        if not args.hide_other_objects:
            for det in other_objects:
                label = f"{det['class_name']} {format_track_suffix(det)}".strip()
                draw_box(frame, det, GENERIC_COLOR, label)
        if ball is not None:
            draw_ball(frame, ball, ball_trail)
        else:
            ball_trail.appendleft(None)
        draw_events(frame, event_detector.bounces)
        mini_court_layout = draw_mini_court(
            frame,
            enabled=not args.no_court_overlay,
            size=args.court_overlay_size,
            margin=args.court_overlay_margin,
        )
        ball_world_point = project_to_court_world(ball["center"], inv_court_homography) if ball is not None else None
        bounce_world_points = []
        for bounce in event_detector.bounces:
            world_point = bounce.get("world_point")
            if isinstance(world_point, list) and len(world_point) == 2:
                bounce_world_points.append((float(world_point[0]), float(world_point[1])))
        draw_mini_court_points(frame, mini_court_layout, ball_world_point, bounce_world_points)

        status = [
            f"frame={frame_index}",
            f"ball={'yes' if ball is not None else 'no'}",
            f"bounces={len(event_detector.bounces)}",
            f"players={len([p for p in (far_player, near_player) if p is not None])}",
            f"objects={len(scene_detections)}",
        ]
        cv2.putText(
            frame,
            "  ".join(status),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if log_handle is not None:
            write_frame_log(log_handle, frame_index, ball, near_player, far_player, scene_detections, event)
        if writer is not None:
            writer.write(frame)
        if not args.headless:
            cv2.imshow("YOLO Tennis Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if log_handle is not None:
        log_handle.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
