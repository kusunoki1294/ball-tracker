import argparse
import json
import math
import os
from collections import deque
import subprocess
import sys
import threading

import cv2
import numpy as np

from court_geometry import project_to_court_world

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


PRESETS = {
    "annotated2": {
        "hmin": 20,
        "hmax": 85,
        "smin": 50,
        "smax": 255,
        "vmin": 50,
        "vmax": 255,
        "roi_top": 0.25,
        "use_motion": True,
        "min_motion_ratio": 0.03,
        "max_radius": 12,
        "max_area": 400,
        "min_circularity": 0.35,
        "min_solidity": 0.90,
        "min_fill_ratio": 0.60,
        "min_extent": 0.50,
        "prefer_small": True,
        "use_kalman": True,
    },
    "tennis5": {
        "hmin": 20,
        "hmax": 80,
        "smin": 50,
        "smax": 255,
        "vmin": 50,
        "vmax": 255,
        "roi_top": 0.25,
        "use_motion": True,
        "min_motion_ratio": 0.04,
        "max_radius": 12,
        "max_area": 320,
        "min_circularity": 0.32,
        "min_solidity": 0.88,
        "min_fill_ratio": 0.58,
        "min_extent": 0.48,
        "prefer_small": True,
        "use_kalman": True,
        "kalman_max_distance": 250,
        "max_jump": 500,
    },
}


class FFmpegCapture:
    def __init__(self, process, width, height, fps):
        self._process = process
        self._width = width
        self._height = height
        self._fps = fps
        self._opened = process is not None

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        frame_size = self._width * self._height * 3
        data = self._process.stdout.read(frame_size)
        if data is None or len(data) < frame_size:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self._height, self._width, 3)).copy()
        return True, frame

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS and self._fps:
            return float(self._fps)
        return 0.0

    def release(self):
        if not self._opened:
            return
        try:
            self._process.terminate()
        except Exception:
            pass
        self._opened = False


class FFmpegWriter:
    def __init__(self, process):
        self._process = process
        self._opened = process is not None

    def isOpened(self):
        return self._opened

    def write(self, frame):
        if not self._opened or self._process.stdin is None:
            return
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self._opened = False

    def release(self):
        if not self._opened:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.wait()
        finally:
            self._opened = False


def open_video(path, force_ffmpeg=False):
    if not force_ffmpeg:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
    ffmpeg_cap = open_video_with_ffmpeg(path)
    if ffmpeg_cap is not None and ffmpeg_cap.isOpened():
        return ffmpeg_cap
    return cv2.VideoCapture(path)


def open_writer_with_ffmpeg(path, width, height, fps):
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.3f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None

    if process.stdin is None:
        return None
    return FFmpegWriter(process)


def open_video_with_ffmpeg(path):
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if probe.returncode != 0:
        return None

    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    try:
        width = int(lines[0])
        height = int(lines[1])
        fps = parse_fps(lines[2])
    except ValueError:
        return None

    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-noautorotate",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None

    if process.stdout is None:
        return None
    return FFmpegCapture(process, width, height, fps)


def parse_fps(rate):
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_val = float(den)
        return float(num) / den_val if den_val else 0.0
    return float(rate)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file name, e.g. tennis.mp4")
    ap.add_argument("--output", help="save annotated video to this path (e.g. output.mp4)")
    ap.add_argument("--buffer", type=int, default=64, help="trail length")
    ap.add_argument("--roi-top", type=float, default=0.25, help="ignore top portion of frame (0-1)")
    ap.add_argument("--hmin", type=int, default=20, help="HSV lower H (0-179)")
    ap.add_argument("--smin", type=int, default=50, help="HSV lower S (0-255)")
    ap.add_argument("--vmin", type=int, default=50, help="HSV lower V (0-255)")
    ap.add_argument("--hmax", type=int, default=85, help="HSV upper H (0-179)")
    ap.add_argument("--smax", type=int, default=255, help="HSV upper S (0-255)")
    ap.add_argument("--vmax", type=int, default=255, help="HSV upper V (0-255)")
    ap.add_argument("--preset", type=str, help="apply a named preset (e.g. annotated2)")
    ap.add_argument("--auto-hsv", action="store_true", help="auto-calibrate HSV range from motion samples")
    ap.add_argument("--auto-hsv-frames", type=int, default=180, help="frames to sample for auto HSV")
    ap.add_argument("--auto-hsv-samples", type=int, default=5000, help="max HSV samples to collect")
    ap.add_argument("--auto-hsv-expand-h", type=int, default=8, help="expand auto HSV hue by this amount")
    ap.add_argument("--auto-hsv-expand-s", type=int, default=40, help="expand auto HSV saturation by this amount")
    ap.add_argument("--auto-hsv-expand-v", type=int, default=40, help="expand auto HSV value by this amount")
    ap.add_argument("--auto-hsv-smin", type=int, default=40, help="min S for auto HSV samples")
    ap.add_argument("--auto-hsv-vmin", type=int, default=40, help="min V for auto HSV samples")
    ap.add_argument("--motion-thresh", type=int, default=15, help="frame-diff threshold")
    ap.add_argument("--hsv-erode", type=int, default=1, help="HSV mask erosion iterations")
    ap.add_argument("--hsv-dilate", type=int, default=1, help="HSV mask dilation iterations")
    ap.add_argument("--motion-erode", type=int, default=1, help="motion mask erosion iterations")
    ap.add_argument("--motion-dilate", type=int, default=2, help="motion mask dilation iterations")
    ap.add_argument("--use-motion", action="store_true", help="AND motion mask with HSV mask")
    ap.add_argument("--min-motion-ratio", type=float, default=0.05, help="min motion overlap ratio (0-1)")
    ap.add_argument("--use-kalman", action="store_true", help="use Kalman prediction to gate tracking")
    ap.add_argument("--kalman-max-distance", type=float, default=120, help="max distance to predicted point")
    ap.add_argument("--kalman-miss-reset", type=int, default=12, help="reset Kalman after N misses")
    ap.add_argument("--min-radius", type=float, default=1, help="min enclosing-circle radius")
    ap.add_argument("--max-radius", type=float, default=25, help="max enclosing-circle radius")
    ap.add_argument("--min-area", type=float, default=5, help="min contour area")
    ap.add_argument("--max-area", type=float, default=2000, help="max contour area")
    ap.add_argument("--min-circularity", type=float, default=0.2, help="min circularity 0-1")
    ap.add_argument("--min-fill-ratio", type=float, default=0.55, help="min area / circle area ratio (0-1)")
    ap.add_argument("--min-extent", type=float, default=0.4, help="min area / bbox area ratio (0-1)")
    ap.add_argument("--max-aspect-ratio", type=float, default=1.6, help="max bbox aspect ratio (w/h or h/w)")
    ap.add_argument("--min-solidity", type=float, default=0.85, help="min contour solidity (0-1)")
    ap.add_argument("--min-ellipse-ratio", type=float, default=0.6, help="min ellipse axis ratio (0-1)")
    ap.add_argument("--max-jump", type=int, default=250, help="max pixel jump between frames")
    ap.add_argument("--lost-reset", type=int, default=10, help="reset after N lost frames")
    ap.add_argument("--ignore-left", type=float, default=0.0, help="ignore left portion of frame (0-1)")
    ap.add_argument("--ignore-right", type=float, default=0.0, help="ignore right portion of frame (0-1)")
    ap.add_argument("--prefer-small", action="store_true", help="prefer smaller candidates when ungated")
    ap.add_argument("--tune", action="store_true", help="enable trackbar tuning UI")
    ap.add_argument("--tune-keyboard", action="store_true", help="tune HSV via keyboard in the main window")
    ap.add_argument("--tune-stdin", action="store_true", help="tune HSV by typing values in the terminal")
    ap.add_argument("--headless", action="store_true", help="disable all OpenCV windows")
    ap.add_argument("--display-max-width", type=int, default=1280, help="max display width (0 = no limit)")
    ap.add_argument("--display-max-height", type=int, default=720, help="max display height (0 = no limit)")
    ap.add_argument("--no-court-overlay", action="store_true", help="disable mini court overlay")
    ap.add_argument("--court-overlay-size", type=int, default=260, help="mini court overlay width in pixels")
    ap.add_argument("--court-overlay-margin", type=int, default=12, help="mini court overlay margin in pixels")
    ap.add_argument("--auto-court-lines", action="store_true", help="auto-detect court lines once from the first frame")
    ap.add_argument("--court-calib-file", default="court_calib.json", help="court calibration file path")
    ap.add_argument("--no-court-lines", action="store_true", help="disable perspective court lines")
    ap.add_argument("--manual-court", action="store_true", help="manually click 6 court points on the first frame")
    ap.add_argument("--doubles", action="store_true", help="use doubles boundaries for in/out check")
    ap.add_argument("--yolo-model", type=str, help="path to YOLO .pt model for ball detection")
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    ap.add_argument("--yolo-device", type=str, default="", help="YOLO device (e.g. cpu, 0)")
    ap.add_argument("--yolo-every", type=int, default=1, help="run YOLO every N frames")
    ap.add_argument("--yolo-only", action="store_true", help="use YOLO only (skip HSV/motion)")
    ap.add_argument("--yolo-log", type=str, help="write YOLO detections to CSV (frame,x,y,conf)")
    ap.add_argument("--yolo-log-max-frames", type=int, default=0, help="max frames to log (0 = all)")
    ap.add_argument("--force-ffmpeg", action="store_true", help="always use ffmpeg for decoding input")
    ap.add_argument("--court-gate-margin", type=float, default=2.0, help="allow this many feet outside court bounds when gating detections")
    return ap.parse_args()


def setup_tuning(args):
    cv2.namedWindow("tuning", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Hmin", "tuning", args.hmin, 179, lambda v: None)
    cv2.createTrackbar("Smin", "tuning", args.smin, 255, lambda v: None)
    cv2.createTrackbar("Vmin", "tuning", args.vmin, 255, lambda v: None)
    cv2.createTrackbar("Hmax", "tuning", args.hmax, 179, lambda v: None)
    cv2.createTrackbar("Smax", "tuning", args.smax, 255, lambda v: None)
    cv2.createTrackbar("Vmax", "tuning", args.vmax, 255, lambda v: None)
    cv2.createTrackbar("Motion", "tuning", args.motion_thresh, 255, lambda v: None)
    cv2.createTrackbar("ROI%", "tuning", int(args.roi_top * 100), 100, lambda v: None)
    cv2.createTrackbar("MinR", "tuning", int(args.min_radius), 50, lambda v: None)
    cv2.createTrackbar("MaxR", "tuning", int(args.max_radius), 50, lambda v: None)
    cv2.createTrackbar("MinA", "tuning", int(args.min_area), 2000, lambda v: None)
    cv2.createTrackbar("MaxA", "tuning", int(args.max_area), 2000, lambda v: None)
    cv2.createTrackbar("MaxJump", "tuning", args.max_jump, 1000, lambda v: None)
    cv2.createTrackbar("MinCirc", "tuning", int(args.min_circularity * 100), 100, lambda v: None)


def read_tuning():
    hmin = cv2.getTrackbarPos("Hmin", "tuning")
    smin = cv2.getTrackbarPos("Smin", "tuning")
    vmin = cv2.getTrackbarPos("Vmin", "tuning")
    hmax = cv2.getTrackbarPos("Hmax", "tuning")
    smax = cv2.getTrackbarPos("Smax", "tuning")
    vmax = cv2.getTrackbarPos("Vmax", "tuning")
    motion_thresh = cv2.getTrackbarPos("Motion", "tuning")
    roi_top = cv2.getTrackbarPos("ROI%", "tuning") / 100.0
    min_radius = max(0, cv2.getTrackbarPos("MinR", "tuning"))
    max_radius = max(0, cv2.getTrackbarPos("MaxR", "tuning"))
    min_area = max(0, cv2.getTrackbarPos("MinA", "tuning"))
    max_area = max(0, cv2.getTrackbarPos("MaxA", "tuning"))
    max_jump = max(0, cv2.getTrackbarPos("MaxJump", "tuning"))
    min_circ = cv2.getTrackbarPos("MinCirc", "tuning") / 100.0

    lower = np.array([min(hmin, hmax), min(smin, smax), min(vmin, vmax)], dtype=np.uint8)
    upper = np.array([max(hmin, hmax), max(smin, smax), max(vmin, vmax)], dtype=np.uint8)
    return lower, upper, motion_thresh, roi_top, min_radius, max_radius, min_area, max_area, max_jump, min_circ


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def apply_keyboard_tuning(key, hmin, hmax, smin, smax, vmin, vmax):
    step = 1
    if key == ord("h"):
        hmin = clamp(hmin - step, 0, 179)
    elif key == ord("H"):
        hmin = clamp(hmin + step, 0, 179)
    elif key == ord("j"):
        hmax = clamp(hmax - step, 0, 179)
    elif key == ord("J"):
        hmax = clamp(hmax + step, 0, 179)
    elif key == ord("s"):
        smin = clamp(smin - step, 0, 255)
    elif key == ord("S"):
        smin = clamp(smin + step, 0, 255)
    elif key == ord("d"):
        smax = clamp(smax - step, 0, 255)
    elif key == ord("D"):
        smax = clamp(smax + step, 0, 255)
    elif key == ord("v"):
        vmin = clamp(vmin - step, 0, 255)
    elif key == ord("V"):
        vmin = clamp(vmin + step, 0, 255)
    elif key == ord("b"):
        vmax = clamp(vmax - step, 0, 255)
    elif key == ord("B"):
        vmax = clamp(vmax + step, 0, 255)
    return hmin, hmax, smin, smax, vmin, vmax


def start_stdin_reader():
    queue = deque()
    lock = threading.Lock()

    def _reader():
        for line in sys.stdin:
            with lock:
                queue.append(line.rstrip("\n"))

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return queue, lock


def poll_stdin_line(queue, lock):
    with lock:
        if not queue:
            return None
        return queue.popleft()


def parse_tune_line(line, hmin, hmax, smin, smax, vmin, vmax):
    if not line:
        return hmin, hmax, smin, smax, vmin, vmax
    if line.lower() == "p":
        print(f"Hmin={hmin} Hmax={hmax} Smin={smin} Smax={smax} Vmin={vmin} Vmax={vmax}")
        return hmin, hmax, smin, smax, vmin, vmax
    parts = line.replace(",", " ").split()
    if len(parts) == 6 and all(p.isdigit() for p in parts):
        hmin, hmax, smin, smax, vmin, vmax = map(int, parts)
    else:
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if not value.isdigit():
                continue
            val = int(value)
            key = key.lower()
            if key == "hmin":
                hmin = val
            elif key == "hmax":
                hmax = val
            elif key == "smin":
                smin = val
            elif key == "smax":
                smax = val
            elif key == "vmin":
                vmin = val
            elif key == "vmax":
                vmax = val
    hmin = clamp(hmin, 0, 179)
    hmax = clamp(hmax, 0, 179)
    smin = clamp(smin, 0, 255)
    smax = clamp(smax, 0, 255)
    vmin = clamp(vmin, 0, 255)
    vmax = clamp(vmax, 0, 255)
    return hmin, hmax, smin, smax, vmin, vmax


def compute_roi(frame, roi_top):
    h, w = frame.shape[:2]
    roi_top = min(max(roi_top, 0.0), 0.95)
    y0 = int(h * roi_top)
    return frame[y0:h, 0:w], y0


def hsv_mask(roi, lower, upper, erode_iters, dilate_iters):
    blurred = cv2.GaussianBlur(roi, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if erode_iters > 0:
        mask = cv2.erode(mask, None, iterations=erode_iters)
    if dilate_iters > 0:
        mask = cv2.dilate(mask, None, iterations=dilate_iters)
    return mask, blurred


def motion_mask(blurred, prev_gray, motion_thresh, erode_iters, dilate_iters):
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = np.zeros_like(gray)
    if prev_gray is not None:
        delta = cv2.absdiff(prev_gray, gray)
        motion_thresh = max(1, motion_thresh)
        _, mask = cv2.threshold(delta, motion_thresh, 255, cv2.THRESH_BINARY)
        if erode_iters > 0:
            mask = cv2.erode(mask, None, iterations=erode_iters)
        if dilate_iters > 0:
            mask = cv2.dilate(mask, None, iterations=dilate_iters)

    return mask, gray


def contour_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    area = cv2.contourArea(contour)
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def world_point_in_court(world_pt, singles=True, margin=0.0):
    if world_pt is None:
        return False
    xw, yw = world_pt
    court_wid = 27.0 if singles else 36.0
    x_min = (36.0 - court_wid) / 2.0 if singles else 0.0
    x_max = x_min + court_wid
    margin = max(0.0, margin)
    return x_min - margin <= xw <= x_max + margin and -margin <= yw <= 78.0 + margin


def select_ball_contour(
    mask,
    motion,
    motion_valid,
    min_motion_ratio,
    pred_point,
    max_pred_dist,
    y0,
    frame_width,
    ignore_left_ratio,
    ignore_right_ratio,
    last_center,
    min_radius,
    max_radius,
    min_area,
    max_area,
    min_circ,
    min_fill_ratio,
    min_extent,
    max_aspect_ratio,
    min_solidity,
    min_ellipse_ratio,
    max_jump,
    prefer_small,
    inv_court_homography=None,
    court_gate_margin=0.0,
    singles=True,
):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_radius = 0.0
    best_dist = None
    best_score = None
    debug = {
        "contours": len(cnts),
        "passed_radius": 0,
        "passed_area": 0,
        "passed_shape": 0,
        "passed_motion": 0,
        "passed_gate": 0,
        "passed_court": 0,
    }
    ignore_left_px = int(frame_width * ignore_left_ratio)
    ignore_right_px = int(frame_width * (1.0 - ignore_right_ratio))

    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius < min_radius or radius > max_radius:
            continue
        debug["passed_radius"] += 1
        if int(x) < ignore_left_px:
            continue
        if int(x) > ignore_right_px:
            continue

        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        debug["passed_area"] += 1

        if radius > 0:
            circle_area = np.pi * radius * radius
            fill_ratio = area / float(circle_area)
            if fill_ratio < min_fill_ratio:
                continue

        x0, y0_rect, w, h = cv2.boundingRect(c)
        if w > 0 and h > 0:
            extent = area / float(w * h)
            if extent < min_extent:
                continue
            aspect_ratio = max(w / float(h), h / float(w))
            if aspect_ratio > max_aspect_ratio:
                continue

        circ = contour_circularity(c)
        if circ < min_circ:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / float(hull_area)
            if solidity < min_solidity:
                continue

        if len(c) >= 5:
            (_, _), (ma, ma_minor), _ = cv2.fitEllipse(c)
            if ma_minor > 0:
                ellipse_ratio = min(ma, ma_minor) / max(ma, ma_minor)
                if ellipse_ratio < min_ellipse_ratio:
                    continue
        debug["passed_shape"] += 1

        if motion_valid and min_motion_ratio > 0:
            motion_mask = np.zeros_like(mask)
            cv2.drawContours(motion_mask, [c], -1, 255, -1)
            motion_hits = cv2.countNonZero(cv2.bitwise_and(motion, motion_mask))
            motion_area = cv2.countNonZero(motion_mask)
            if motion_area == 0:
                continue
            motion_ratio = motion_hits / float(motion_area)
            if motion_ratio < min_motion_ratio:
                continue
        debug["passed_motion"] += 1

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        cand_center = (cx, cy + y0)

        if inv_court_homography is not None:
            world_pt = project_to_court_world(cand_center, inv_court_homography)
            if not world_point_in_court(world_pt, singles=singles, margin=court_gate_margin):
                continue
        debug["passed_court"] += 1

        gate_last_center = last_center
        if gate_last_center is not None and inv_court_homography is not None:
            last_world = project_to_court_world(gate_last_center, inv_court_homography)
            if not world_point_in_court(last_world, singles=singles, margin=court_gate_margin):
                gate_last_center = None

        if pred_point is not None and max_pred_dist is not None:
            dxp = cand_center[0] - pred_point[0]
            dyp = cand_center[1] - pred_point[1]
            dist2 = dxp * dxp + dyp * dyp
            if dist2 > max_pred_dist * max_pred_dist:
                continue
        elif gate_last_center is not None:
            dx = cand_center[0] - gate_last_center[0]
            dy = cand_center[1] - gate_last_center[1]
            if dx * dx + dy * dy > max_jump * max_jump:
                if inv_court_homography is not None and gate_last_center[1] - cand_center[1] > max_jump:
                    pass
                else:
                    continue
        debug["passed_gate"] += 1

        if pred_point is not None:
            dxp = cand_center[0] - pred_point[0]
            dyp = cand_center[1] - pred_point[1]
            dist2 = dxp * dxp + dyp * dyp
            if best is None or dist2 < best_dist:
                best = (x, y, radius, cand_center)
                best_dist = dist2
        else:
            if prefer_small:
                score = -radius
            else:
                score = radius
            if best is None or score > best_score:
                best_score = score
                best_radius = radius
                best = (x, y, radius, cand_center)

    return best, debug


def scale_for_display(image, max_width, max_height):
    if max_width <= 0 and max_height <= 0:
        return image
    h, w = image.shape[:2]
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / float(w))
    if max_height > 0:
        scale = min(scale, max_height / float(h))
    if scale >= 1.0:
        return image
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def show_window(name, image, max_width, max_height):
    display = scale_for_display(image, max_width, max_height)
    cv2.imshow(name, display)

def draw_court_overlay(frame, enabled, size, margin, include_singles=True):
    if not enabled:
        return
    h, w = frame.shape[:2]
    size = int(max(120, size))
    margin = int(max(0, margin))

    # Standard doubles court: 78 ft (length) x 36 ft (width)
    court_len = 78.0
    court_wid = 36.0
    aspect = court_len / court_wid

    overlay_w = size
    overlay_h = int(round(overlay_w * aspect))
    if overlay_h + 2 * margin > h:
        overlay_h = max(120, h - 2 * margin)
        overlay_w = int(round(overlay_h / aspect))
    x1 = max(0, w - margin - overlay_w)
    y1 = max(0, margin)
    x2 = min(w - margin, x1 + overlay_w)
    y2 = min(h - margin, y1 + overlay_h)

    # Recompute in case of clamping
    overlay_w = max(1, x2 - x1)
    overlay_h = max(1, y2 - y1)

    # Helper to map court coords to image
    # Court coords: (0..court_wid, 0..court_len)
    def pt(xw, yl):
        x = int(x1 + (xw / court_wid) * overlay_w)
        y = int(y1 + (yl / court_len) * overlay_h)
        return x, y

    # Draw background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    color = (255, 255, 255)
    thick = 2

    # Outer doubles court
    cv2.rectangle(frame, pt(0, 0), pt(court_wid, court_len), color, thick)

    # Singles sidelines (27 ft wide, centered)
    if include_singles:
        singles_left = (court_wid - 27.0) / 2.0
        singles_right = singles_left + 27.0
        cv2.line(frame, pt(singles_left, 0), pt(singles_left, court_len), color, thick)
        cv2.line(frame, pt(singles_right, 0), pt(singles_right, court_len), color, thick)

    # Net line (center)
    net_y = court_len / 2.0
    cv2.line(frame, pt(0, net_y), pt(court_wid, net_y), color, thick)

    # Service lines (21 ft from net)
    service_dist = 21.0
    if include_singles:
        cv2.line(frame, pt(singles_left, net_y - service_dist), pt(singles_right, net_y - service_dist), color, thick)
        cv2.line(frame, pt(singles_left, net_y + service_dist), pt(singles_right, net_y + service_dist), color, thick)

    # Center service line
    center_x = court_wid / 2.0
    if include_singles:
        cv2.line(frame, pt(center_x, net_y - service_dist), pt(center_x, net_y + service_dist), color, thick)

    # Center marks (1 ft)
    mark_len = 1.0
    cv2.line(frame, pt(center_x, 0), pt(center_x, mark_len), color, thick)
    cv2.line(frame, pt(center_x, court_len - mark_len), pt(center_x, court_len), color, thick)



def load_court_calibration(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    points = data.get("points")
    net_points = data.get("net_points")
    if not isinstance(points, list) or len(points) != 4:
        return None
    try:
        court_points = [(float(x), float(y)) for x, y in points]
        net = None
        if isinstance(net_points, list) and len(net_points) == 2:
            net = [(float(x), float(y)) for x, y in net_points]
        return {"points": court_points, "net_points": net}
    except Exception:
        return None


def save_court_calibration(path, points, net_points=None):
    data = {"points": [[float(x), float(y)] for x, y in points]}
    if net_points is not None:
        data["net_points"] = [[float(x), float(y)] for x, y in net_points]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def court_world_lines(include_singles=True):
    court_len = 78.0
    court_wid = 36.0
    singles_wid = 27.0
    singles_left = (court_wid - singles_wid) / 2.0
    singles_right = singles_left + singles_wid
    net_y = court_len / 2.0
    service_dist = 21.0
    center_x = court_wid / 2.0
    mark_len = 1.0

    lines = []
    # Outer doubles court
    lines.append(([(0, 0), (court_wid, 0), (court_wid, court_len), (0, court_len)], True))
    # Singles sidelines
    if include_singles:
        lines.append(([(singles_left, 0), (singles_left, court_len)], False))
        lines.append(([(singles_right, 0), (singles_right, court_len)], False))
    # Net line
    lines.append(([(0, net_y), (court_wid, net_y)], False))
    # Service lines
    lines.append(([(singles_left, net_y - service_dist), (singles_right, net_y - service_dist)], False))
    lines.append(([(singles_left, net_y + service_dist), (singles_right, net_y + service_dist)], False))
    # Center service line
    lines.append(([(center_x, net_y - service_dist), (center_x, net_y + service_dist)], False))
    # Center marks
    lines.append(([(center_x, 0), (center_x, mark_len)], False))
    lines.append(([(center_x, court_len - mark_len), (center_x, court_len)], False))
    return lines


def draw_court_lines(frame, homography, color=(255, 255, 255), thickness=2, include_singles=True):
    if homography is None:
        return
    for pts, closed in court_world_lines(include_singles=include_singles):
        src = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, homography).astype(int)
        cv2.polylines(frame, [dst], closed, color, thickness, lineType=cv2.LINE_AA)


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line_from_points(x1, y1, x2, y2):
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm == 0:
        return None
    return a / norm, b / norm, c / norm


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return x, y


def _order_court_corners(points):
    # Assumes camera is behind the near baseline, so near side is bottom of image.
    points = sorted(points, key=lambda p: p[1])
    far = points[:2]
    near = points[2:]
    far_left, far_right = sorted(far, key=lambda p: p[0])
    near_left, near_right = sorted(near, key=lambda p: p[0])
    return [near_left, near_right, far_right, far_left]


def detect_court_corners(frame):
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > 1280:
        scale = 1280.0 / max_dim
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White line mask (low saturation, high value)
    lower = np.array([0, 0, 160], dtype=np.uint8)
    upper = np.array([179, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    edges = cv2.Canny(mask, 50, 150)
    min_len = int(0.2 * min(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=30)
    if lines is None:
        return None

    angles = []
    lengths = []
    line_data = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0:
            ang += 180.0
        angles.append(ang)
        lengths.append(length)
        line_data.append((x1, y1, x2, y2, ang, length))

    if len(angles) < 4:
        return None

    # Histogram to find two dominant directions
    hist = np.zeros(180, dtype=np.float32)
    for ang, length in zip(angles, lengths):
        hist[int(ang) % 180] += length

    peak1 = int(np.argmax(hist))
    hist2 = hist.copy()
    for i in range(180):
        if _angle_diff(i, peak1) < 20:
            hist2[i] = 0
    peak2 = int(np.argmax(hist2))
    if hist2[peak2] == 0:
        return None

    group_a = []
    group_b = []
    for x1, y1, x2, y2, ang, length in line_data:
        if _angle_diff(ang, peak1) <= _angle_diff(ang, peak2):
            group_a.append((x1, y1, x2, y2))
        else:
            group_b.append((x1, y1, x2, y2))

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    def outer_lines(group):
        lines_nf = []
        for x1, y1, x2, y2 in group:
            line = _line_from_points(x1, y1, x2, y2)
            if line is None:
                continue
            lines_nf.append(line)
        if len(lines_nf) < 2:
            return None
        cs = [l[2] for l in lines_nf]
        min_i = int(np.argmin(cs))
        max_i = int(np.argmax(cs))
        return lines_nf[min_i], lines_nf[max_i]

    a1, a2 = outer_lines(group_a) or (None, None)
    b1, b2 = outer_lines(group_b) or (None, None)
    if a1 is None or b1 is None:
        return None

    pts = []
    for la in (a1, a2):
        for lb in (b1, b2):
            p = _intersect(la, lb)
            if p is None:
                continue
            x, y = p
            pts.append((x, y))

    if len(pts) < 4:
        return None

    # Filter to points roughly inside image
    pts = [(x, y) for x, y in pts if -0.2 * w <= x <= 1.2 * w and -0.2 * h <= y <= 1.2 * h]
    if len(pts) < 4:
        return None

    # If extra points, keep the 4 that form the largest area
    if len(pts) > 4:
        pts = pts[:4]

    if scale != 1.0:
        pts = [(x / scale, y / scale) for x, y in pts]

    return _order_court_corners(pts)


def classify_ball_in_out(center, inv_homography, singles=True):
    world = project_to_court_world(center, inv_homography)
    if world is None:
        return None, None
    xw, yw = world

    court_wid = 36.0
    court_len = 78.0
    if singles:
        singles_left = (court_wid - 27.0) / 2.0
        singles_right = singles_left + 27.0
        in_bounds = singles_left <= xw <= singles_right and 0.0 <= yw <= court_len
    else:
        in_bounds = 0.0 <= xw <= court_wid and 0.0 <= yw <= court_len
    return in_bounds, (xw, yw)


def collect_manual_court_points(frame, max_width=1280, max_height=720):
    points = []
    net_points = []
    window_name = "Manual Court Calibration"
    h, w = frame.shape[:2]
    scale = 1.0
    if max_width > 0:
        scale = min(scale, max_width / float(w))
    if max_height > 0:
        scale = min(scale, max_height / float(h))

    def _draw():
        if scale < 1.0:
            display = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display = frame.copy()
        overlay = display.copy()
        for i, (x, y) in enumerate(points):
            dx, dy = int(x * scale), int(y * scale)
            cv2.circle(overlay, (dx, dy), 6, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f"{i+1}",
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        for i, (x, y) in enumerate(net_points):
            dx, dy = int(x * scale), int(y * scale)
            cv2.circle(overlay, (dx, dy), 6, (0, 255, 255), -1)
            cv2.putText(
                overlay,
                f"N{i+1}",
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        instructions = [
            "Click 4 corners in order:",
            "1) Near-left  2) Near-right  3) Far-right  4) Far-left",
            "Then click net bottom points: 5) Net-left  6) Net-right",
            "Press 'r' to reset, 'q' to cancel, or wait until 6 points are set.",
        ]
        y = 30
        for line in instructions:
            cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        cv2.imshow(window_name, overlay)

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if scale > 0:
            x = int(round(x / scale))
            y = int(round(y / scale))
        if len(points) < 4:
            points.append((x, y))
        elif len(net_points) < 2:
            net_points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        _draw()
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
            net_points.clear()
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return None
        if len(points) == 4 and len(net_points) == 2:
            cv2.destroyWindow(window_name)
            return points, net_points


def detect_ball_yolo(frame, model, conf, imgsz, device):
    if model is None:
        return None, None, None
    device_arg = device if device else None
    results = model.predict(frame, conf=conf, imgsz=imgsz, device=device_arg, verbose=False)
    if not results:
        return None, None, None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None
    confs = boxes.conf
    if confs is None or len(confs) == 0:
        return None, None, None
    best_i = int(np.argmax(confs.cpu().numpy()))
    x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    radius = int(max(2, min(x2 - x1, y2 - y1) / 2.0))
    return (cx, cy), radius, float(confs[best_i])


def draw_debug_overlay(frame, lines, origin=(10, 10)):
    if not lines:
        return
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_h = 18
    width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        width = max(width, w)
    pad = 6
    height = line_h * len(lines) + pad * 2 - 4
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - pad),
        (x + width + pad, y + height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        yy = y + i * line_h + 12
        cv2.putText(frame, line, (x, yy), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def auto_calibrate_hsv(video_path, args):
    cap = open_video(video_path, force_ffmpeg=args.force_ffmpeg)
    if not cap.isOpened():
        print("Auto HSV: could not open video.")
        return None

    samples = []
    prev_gray = None
    total_samples = 0
    frame_i = 0

    while frame_i < args.auto_hsv_frames:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break
        frame_i += 1

        roi, _ = compute_roi(frame, args.roi_top)
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is None:
            prev_gray = gray
            continue
        delta = cv2.absdiff(prev_gray, gray)
        prev_gray = gray

        motion_thresh = max(1, args.motion_thresh)
        _, motion = cv2.threshold(delta, motion_thresh, 255, cv2.THRESH_BINARY)
        if args.motion_erode > 0:
            motion = cv2.erode(motion, None, iterations=args.motion_erode)
        if args.motion_dilate > 0:
            motion = cv2.dilate(motion, None, iterations=args.motion_dilate)

        cnts, _ = cv2.findContours(motion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius < args.min_radius or radius > args.max_radius:
                continue
            area = cv2.contourArea(c)
            if area < args.min_area or area > args.max_area:
                continue

            mask = np.zeros(motion.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue

            idx = np.arange(ys.size)
            if idx.size > 200:
                idx = np.random.choice(idx, size=200, replace=False)
            ys = ys[idx]
            xs = xs[idx]

            vals = hsv[ys, xs]
            if vals.size == 0:
                continue
            s_ok = vals[:, 1] >= args.auto_hsv_smin
            v_ok = vals[:, 2] >= args.auto_hsv_vmin
            vals = vals[s_ok & v_ok]
            if vals.size == 0:
                continue

            samples.append(vals)
            total_samples += vals.shape[0]
            if total_samples >= args.auto_hsv_samples:
                break
        if total_samples >= args.auto_hsv_samples:
            break

    cap.release()

    if not samples:
        print("Auto HSV: no samples collected; keeping defaults.")
        return None

    samples = np.vstack(samples)
    h = samples[:, 0]
    s = samples[:, 1]
    v = samples[:, 2]

    hmin = int(np.percentile(h, 5))
    hmax = int(np.percentile(h, 95))
    smin = int(np.percentile(s, 5))
    smax = int(np.percentile(s, 95))
    vmin = int(np.percentile(v, 5))
    vmax = int(np.percentile(v, 95))

    hmin = clamp(hmin - args.auto_hsv_expand_h, 0, 179)
    hmax = clamp(hmax + args.auto_hsv_expand_h, 0, 179)
    smin = clamp(smin - args.auto_hsv_expand_s, 0, 255)
    smax = clamp(smax + args.auto_hsv_expand_s, 0, 255)
    vmin = clamp(vmin - args.auto_hsv_expand_v, 0, 255)
    vmax = clamp(vmax + args.auto_hsv_expand_v, 0, 255)

    print(
        f"Auto HSV: H[{hmin},{hmax}] S[{smin},{smax}] V[{vmin},{vmax}] "
        f"from {samples.shape[0]} samples."
    )
    return hmin, hmax, smin, smax, vmin, vmax


def main():
    args = parse_args()

    video_name = os.path.basename(args.video).lower()
    if not args.preset:
        if video_name.startswith("annotated2"):
            args.preset = "annotated2"
        elif video_name.startswith("tennis5") or video_name.startswith("annotated5"):
            args.preset = "tennis5"

    if args.court_calib_file == "court_calib.json":
        if video_name.startswith("tennis5") or video_name.startswith("annotated5"):
            args.court_calib_file = "court_calib_tennis5.json"

    if args.preset:
        preset = PRESETS.get(args.preset.lower())
        if preset is None:
            print(f"Unknown preset: {args.preset}. Available: {', '.join(PRESETS.keys())}")
            return
        for key, value in preset.items():
            setattr(args, key, value)
        print(f"Preset applied: {args.preset}")

    if args.auto_hsv:
        auto = auto_calibrate_hsv(args.video, args)
        if auto is not None:
            hmin, hmax, smin, smax, vmin, vmax = auto
            args.hmin = hmin
            args.hmax = hmax
            args.smin = smin
            args.smax = smax
            args.vmin = vmin
            args.vmax = vmax

    cap = open_video(args.video, force_ffmpeg=args.force_ffmpeg)
    if not cap.isOpened():
        print("Could not open video:", args.video)
        return

    keyboard_tune = args.tune_keyboard
    stdin_tune = args.tune_stdin
    use_trackbars = args.tune and not args.headless and not keyboard_tune and not stdin_tune
    stdin_queue = None
    stdin_lock = None
    if stdin_tune:
        stdin_queue, stdin_lock = start_stdin_reader()
        print("stdin tuning enabled: type Hmin=.. Hmax=.. Smin=.. Smax=.. Vmin=.. Vmax=.. and press Enter")

    if not args.headless:
        if use_trackbars:
            setup_tuning(args)
        cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
        cv2.namedWindow("hsv_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("motion_mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("final_mask", cv2.WINDOW_NORMAL)

    pts = deque(maxlen=args.buffer)
    last_center = None
    lost_frames = 0
    prev_gray = None
    frame_i = 0
    kalman = None
    kalman_misses = 0

    if args.use_kalman:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)

    hmin, smin, vmin = args.hmin, args.smin, args.vmin
    hmax, smax, vmax = args.hmax, args.smax, args.vmax
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    motion_thresh = args.motion_thresh
    hsv_erode = args.hsv_erode
    hsv_dilate = args.hsv_dilate
    motion_erode = args.motion_erode
    motion_dilate = args.motion_dilate
    roi_top = args.roi_top
    min_radius = args.min_radius
    max_radius = args.max_radius
    min_area = args.min_area
    max_area = args.max_area
    min_circ = args.min_circularity
    min_fill_ratio = args.min_fill_ratio
    min_extent = args.min_extent
    max_aspect_ratio = args.max_aspect_ratio
    min_solidity = args.min_solidity
    min_ellipse_ratio = args.min_ellipse_ratio
    max_jump = args.max_jump
    min_motion_ratio = args.min_motion_ratio
    ignore_left_ratio = min(max(args.ignore_left, 0.0), 0.95)
    ignore_right_ratio = min(max(args.ignore_right, 0.0), 0.95)

    yolo_model = None
    if args.yolo_model:
        if YOLO is None:
            print("YOLO requested but ultralytics is not installed. Install ultralytics to use YOLO.")
            return
        yolo_model = YOLO(args.yolo_model)

    court_calib = load_court_calibration(args.court_calib_file)
    if court_calib is None and args.auto_court_lines:
        tmp_cap = open_video(args.video, force_ffmpeg=args.force_ffmpeg)
        if tmp_cap.isOpened():
            ok, tmp_frame = tmp_cap.read()
            tmp_cap.release()
            if ok and tmp_frame is not None:
                court_points = detect_court_corners(tmp_frame)
                if court_points is not None:
                    save_court_calibration(args.court_calib_file, court_points)
                    print(f"Auto court calibration saved to {args.court_calib_file}")
                    court_calib = {"points": court_points, "net_points": None}

    if args.manual_court:
        tmp_cap = open_video(args.video, force_ffmpeg=args.force_ffmpeg)
        if tmp_cap.isOpened():
            ok, tmp_frame = tmp_cap.read()
            tmp_cap.release()
            if ok and tmp_frame is not None:
                picked = collect_manual_court_points(tmp_frame, args.display_max_width, args.display_max_height)
                if picked is None:
                    print("Manual court calibration cancelled.")
                    return
                court_points, net_points = picked
                save_court_calibration(args.court_calib_file, court_points, net_points)
                print(f"Manual court calibration saved to {args.court_calib_file}")
                court_calib = {"points": court_points, "net_points": net_points}
        else:
            print("Manual court calibration: could not open video.")
            return

    if court_calib is None or court_calib.get("net_points") is None:
        print("Court calibration missing or incomplete. Run with --manual-court to click 6 points.")
        return

    court_homography = None
    inv_court_homography = None
    court_points = None
    net_points = None
    if court_calib is not None:
        court_points = court_calib.get("points")
        net_points = court_calib.get("net_points")
    if court_points is not None:
        world = np.array([[0.0, 0.0], [36.0, 0.0], [36.0, 78.0], [0.0, 78.0]], dtype=np.float32)
        image = np.array(court_points, dtype=np.float32)
        court_homography = cv2.getPerspectiveTransform(world, image)
        try:
            inv_court_homography = np.linalg.inv(court_homography)
        except np.linalg.LinAlgError:
            inv_court_homography = None

    writer = None
    writer_failed = False
    last_yolo = None
    yolo_log_f = None
    if args.yolo_log:
        yolo_log_f = open(args.yolo_log, "w", encoding="utf-8")
        yolo_log_f.write("frame,x,y,conf\n")
        yolo_log_f.flush()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        frame_i += 1
        if frame_i % 60 == 0:
            print("frame", frame_i)

        if use_trackbars:
            try:
                lower, upper, motion_thresh, roi_top, min_radius, max_radius, min_area, max_area, max_jump, min_circ = read_tuning()
                hmin, smin, vmin = int(lower[0]), int(lower[1]), int(lower[2])
                hmax, smax, vmax = int(upper[0]), int(upper[1]), int(upper[2])
            except cv2.error:
                keyboard_tune = True
                use_trackbars = False

        if args.tune and not args.headless and keyboard_tune:
            lower = np.array([hmin, smin, vmin], dtype=np.uint8)
            upper = np.array([hmax, smax, vmax], dtype=np.uint8)
        if stdin_tune:
            line = poll_stdin_line(stdin_queue, stdin_lock)
            if line is not None:
                hmin, hmax, smin, smax, vmin, vmax = parse_tune_line(
                    line, hmin, hmax, smin, smax, vmin, vmax
                )
            lower = np.array([hmin, smin, vmin], dtype=np.uint8)
            upper = np.array([hmax, smax, vmax], dtype=np.uint8)

        yolo_center = None
        yolo_radius = None
        yolo_conf = None
        if yolo_model is not None and args.yolo_every > 0:
            if frame_i % args.yolo_every == 0:
                yolo_center, yolo_radius, yolo_conf = detect_ball_yolo(
                    frame, yolo_model, args.yolo_conf, args.yolo_imgsz, args.yolo_device
                )
                last_yolo = (yolo_center, yolo_radius, yolo_conf)
            elif last_yolo is not None:
                yolo_center, yolo_radius, yolo_conf = last_yolo
        if yolo_log_f is not None and (args.yolo_log_max_frames <= 0 or frame_i <= args.yolo_log_max_frames):
            if yolo_center is None or yolo_conf is None:
                yolo_log_f.write(f"{frame_i},,,\n")
            else:
                yolo_log_f.write(f"{frame_i},{yolo_center[0]},{yolo_center[1]},{yolo_conf:.4f}\n")

        if args.yolo_only and yolo_model is not None:
            roi, y0 = compute_roi(frame, roi_top)
            hsv = np.zeros(roi.shape[:2], dtype=np.uint8)
            motion = np.zeros(roi.shape[:2], dtype=np.uint8)
            final_mask = hsv
            motion_valid = False
            debug_counts = {"contours": 0, "passed_radius": 0, "passed_area": 0, "passed_shape": 0, "passed_motion": 0, "passed_gate": 0, "passed_court": 0}

            pred_point = None
            if kalman is not None:
                prediction = kalman.predict()
                pred_point = (int(prediction[0, 0]), int(prediction[1, 0]))

            ignore_left_px = int(frame.shape[1] * ignore_left_ratio)
            ignore_right_px = int(frame.shape[1] * (1.0 - ignore_right_ratio))
            center = yolo_center
            if center is not None and (center[0] < ignore_left_px or center[0] > ignore_right_px):
                center = None

            if center is not None:
                if yolo_radius:
                    cv2.circle(frame, center, int(yolo_radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                last_center = center
                lost_frames = 0
                if kalman is not None:
                    measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                    kalman.correct(measurement)
                    kalman_misses = 0
            else:
                lost_frames += 1
                if lost_frames > args.lost_reset:
                    last_center = None
                if kalman is not None:
                    kalman_misses += 1
                    if kalman_misses > args.kalman_miss_reset:
                        kalman = None
        else:
            roi, y0 = compute_roi(frame, roi_top)
            hsv, blurred = hsv_mask(roi, lower, upper, hsv_erode, hsv_dilate)
            prev_gray_before = prev_gray
            motion, prev_gray = motion_mask(blurred, prev_gray, motion_thresh, motion_erode, motion_dilate)
            motion_valid = prev_gray_before is not None

            if args.use_motion and motion_valid:
                final_mask = cv2.bitwise_and(hsv, motion)
            else:
                final_mask = hsv

            pred_point = None
            if kalman is not None:
                prediction = kalman.predict()
                pred_point = (int(prediction[0, 0]), int(prediction[1, 0]))

            ignore_left_px = int(frame.shape[1] * ignore_left_ratio)
            ignore_right_px = int(frame.shape[1] * (1.0 - ignore_right_ratio))

            best, debug_counts = select_ball_contour(
                final_mask,
                motion,
                motion_valid,
                min_motion_ratio,
                pred_point,
                args.kalman_max_distance if kalman is not None else None,
                y0,
                frame.shape[1],
                ignore_left_ratio,
                ignore_right_ratio,
                last_center,
                min_radius,
                max_radius,
                min_area,
                max_area,
                min_circ,
                min_fill_ratio,
                min_extent,
                max_aspect_ratio,
                min_solidity,
                min_ellipse_ratio,
                max_jump,
                args.prefer_small,
                inv_court_homography=inv_court_homography,
                court_gate_margin=args.court_gate_margin,
                singles=not args.doubles,
            )

            center = None
            if best is not None:
                x, y, radius, center = best
                if center[0] < ignore_left_px:
                    center = None
                elif center[0] > ignore_right_px:
                    center = None
                else:
                    cv2.circle(frame, (int(x), int(y + y0)), int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, center, 4, (0, 0, 255), -1)
                    last_center = center
                    lost_frames = 0
                if kalman is not None and center is not None:
                    measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                    kalman.correct(measurement)
                    kalman_misses = 0
            else:
                lost_frames += 1
                if lost_frames > args.lost_reset:
                    last_center = None
                if kalman is not None:
                    kalman_misses += 1
                    if kalman_misses > args.kalman_miss_reset:
                        kalman = None

            if center is None and yolo_center is not None:
                if yolo_center[0] >= ignore_left_px and yolo_center[0] <= ignore_right_px:
                    center = yolo_center
                    if yolo_radius:
                        cv2.circle(frame, center, int(yolo_radius), (0, 255, 0), 2)
                    cv2.circle(frame, center, 4, (255, 255, 0), -1)
                    last_center = center
                    lost_frames = 0
                    if kalman is not None:
                        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
                        kalman.correct(measurement)
                        kalman_misses = 0

            if pred_point is not None and pred_point[0] >= ignore_left_px and pred_point[0] <= ignore_right_px:
                cv2.circle(frame, pred_point, 4, (0, 255, 255), -1)
                if center is None:
                    center = pred_point

        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args.buffer / float(i + 1)) * 2)
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

        in_out, world_pt = classify_ball_in_out(center, inv_court_homography, singles=not args.doubles)
        if in_out is not None:
            status = "IN" if in_out else "OUT"
            color = (0, 255, 0) if in_out else (0, 0, 255)
            cv2.putText(
                frame,
                f"BALL: {status}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA,
            )

        cv2.line(frame, (0, y0), (frame.shape[1], y0), (0, 255, 255), 1)
        if ignore_left_px > 0:
            cv2.line(frame, (ignore_left_px, 0), (ignore_left_px, frame.shape[0]), (0, 0, 255), 2)
        if ignore_right_px < frame.shape[1]:
            cv2.line(frame, (ignore_right_px, 0), (ignore_right_px, frame.shape[0]), (0, 0, 255), 2)

        hsv_nz = cv2.countNonZero(hsv)
        motion_nz = cv2.countNonZero(motion) if motion_valid else 0
        final_nz = cv2.countNonZero(final_mask)
        overlay_lines = [
            f"Frame: {frame_i}",
            f"ROI top: {roi_top:.2f} y0={y0}",
            f"Ignore L/R: {ignore_left_ratio:.2f}/{ignore_right_ratio:.2f}",
            f"HSV: H[{hmin},{hmax}] S[{smin},{smax}] V[{vmin},{vmax}]",
            f"HSV erode/dilate: {hsv_erode}/{hsv_dilate}",
            f"Motion thresh: {motion_thresh} erode/dilate: {motion_erode}/{motion_dilate}",
            f"Use motion: {args.use_motion} min_motion_ratio: {min_motion_ratio:.2f}",
            f"Mask nz: hsv={hsv_nz} motion={motion_nz} final={final_nz}",
            f"Contours: {debug_counts['contours']} rad={debug_counts['passed_radius']} area={debug_counts['passed_area']} shape={debug_counts['passed_shape']} motion={debug_counts['passed_motion']} court={debug_counts['passed_court']} gate={debug_counts['passed_gate']}",
            f"Center: {center} Last: {last_center} Lost: {lost_frames}",
            f"Kalman: {kalman is not None} misses={kalman_misses} pred={pred_point}",
            f"Radius/area limits: r[{min_radius},{max_radius}] area[{min_area},{max_area}]",
            f"Shape: circ>={min_circ:.2f} fill>={min_fill_ratio:.2f} ext>={min_extent:.2f} ar<={max_aspect_ratio:.2f} sol>={min_solidity:.2f} ell>={min_ellipse_ratio:.2f}",
            f"Jump: {max_jump} kalman_max_dist: {args.kalman_max_distance if kalman is not None else 'off'} court_gate_margin: {args.court_gate_margin:.1f}",
        ]
        draw_debug_overlay(frame, overlay_lines, origin=(10, 10))
        draw_court_overlay(
            frame,
            not args.no_court_overlay,
            args.court_overlay_size,
            args.court_overlay_margin,
            include_singles=args.doubles,
        )
        if not args.no_court_lines:
            draw_court_lines(frame, court_homography, include_singles=args.doubles)
        if net_points is not None:
            cv2.line(
                frame,
                (int(net_points[0][0]), int(net_points[0][1])),
                (int(net_points[1][0]), int(net_points[1][1])),
                (0, 255, 255),
                2,
            )

        if not args.headless:
            show_window("roi", roi, args.display_max_width, args.display_max_height)
            show_window("hsv_mask", hsv, args.display_max_width, args.display_max_height)
            show_window("motion_mask", motion, args.display_max_width, args.display_max_height)
            show_window("final_mask", final_mask, args.display_max_width, args.display_max_height)
            show_window("Ball Tracker", frame, args.display_max_width, args.display_max_height)

        if args.output and writer is None and not writer_failed:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1:
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
            if not writer.isOpened():
                writer = open_writer_with_ffmpeg(args.output, w, h, fps)
                if writer is None or not writer.isOpened():
                    print("Could not open video writer:", args.output)
                    writer = None
                    writer_failed = True

        if writer is not None:
            writer.write(frame)

        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if args.tune and keyboard_tune:
                hmin, hmax, smin, smax, vmin, vmax = apply_keyboard_tuning(
                    key, hmin, hmax, smin, smax, vmin, vmax
                )
                if key == ord("p"):
                    print(f"Hmin={hmin} Hmax={hmax} Smin={smin} Smax={smax} Vmin={vmin} Vmax={vmax}")
            if key == ord("q"):
                break

    if args.headless:
        print("Video done.")
    else:
        print("Video done. Press any key in the Ball Tracker window to close.")
        cv2.waitKey(0)
    if writer is not None:
        writer.release()
    cap.release()
    if yolo_log_f is not None:
        yolo_log_f.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
