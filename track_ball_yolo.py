import argparse
import json
import math
import os
from collections import deque

import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PLAYER_CLASS_ID = 0
BALL_COLOR = (0, 255, 255)
PLAYER_NEAR_COLOR = (255, 128, 0)
PLAYER_FAR_COLOR = (0, 200, 255)
GENERIC_COLOR = (0, 255, 0)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-only tennis tracking for the ball, players, and other scene objects."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", help="Optional annotated output video path.")
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


def format_track_suffix(detection):
    track_id = detection.get("track_id")
    return f"#{track_id}" if track_id is not None else ""


def write_frame_log(handle, frame_index, ball, near_player, far_player, scene_detections):
    row = {
        "frame": frame_index,
        "ball": ball,
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

        status = [
            f"frame={frame_index}",
            f"ball={'yes' if ball is not None else 'no'}",
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
            write_frame_log(log_handle, frame_index, ball, near_player, far_player, scene_detections)
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
