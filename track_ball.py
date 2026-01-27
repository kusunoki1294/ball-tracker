import argparse
from collections import deque

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file name, e.g. tennis.mp4")
    ap.add_argument("--buffer", type=int, default=64, help="trail length")
    ap.add_argument("--roi-top", type=float, default=0.30, help="ignore top portion of frame (0-1)")
    ap.add_argument("--hmin", type=int, default=20, help="HSV lower H (0-179)")
    ap.add_argument("--smin", type=int, default=50, help="HSV lower S (0-255)")
    ap.add_argument("--vmin", type=int, default=50, help="HSV lower V (0-255)")
    ap.add_argument("--hmax", type=int, default=85, help="HSV upper H (0-179)")
    ap.add_argument("--smax", type=int, default=255, help="HSV upper S (0-255)")
    ap.add_argument("--vmax", type=int, default=255, help="HSV upper V (0-255)")
    ap.add_argument("--motion-thresh", type=int, default=15, help="frame-diff threshold")
    ap.add_argument("--hsv-erode", type=int, default=1, help="HSV mask erosion iterations")
    ap.add_argument("--hsv-dilate", type=int, default=1, help="HSV mask dilation iterations")
    ap.add_argument("--motion-erode", type=int, default=1, help="motion mask erosion iterations")
    ap.add_argument("--motion-dilate", type=int, default=2, help="motion mask dilation iterations")
    ap.add_argument("--use-motion", action="store_true", help="AND motion mask with HSV mask")
    ap.add_argument("--min-motion-ratio", type=float, default=0.05, help="min motion overlap ratio (0-1)")
    ap.add_argument("--min-radius", type=float, default=1, help="min enclosing-circle radius")
    ap.add_argument("--max-radius", type=float, default=25, help="max enclosing-circle radius")
    ap.add_argument("--min-area", type=float, default=5, help="min contour area")
    ap.add_argument("--max-area", type=float, default=2000, help="max contour area")
    ap.add_argument("--min-circularity", type=float, default=0.2, help="min circularity 0-1")
    ap.add_argument("--max-jump", type=int, default=250, help="max pixel jump between frames")
    ap.add_argument("--lost-reset", type=int, default=10, help="reset after N lost frames")
    ap.add_argument("--tune", action="store_true", help="enable trackbar tuning UI")
    ap.add_argument("--display-max-width", type=int, default=1280, help="max display width (0 = no limit)")
    ap.add_argument("--display-max-height", type=int, default=720, help="max display height (0 = no limit)")
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


def select_ball_contour(
    mask,
    motion,
    motion_valid,
    min_motion_ratio,
    y0,
    last_center,
    min_radius,
    max_radius,
    min_area,
    max_area,
    min_circ,
    max_jump,
):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_radius = 0.0

    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius < min_radius or radius > max_radius:
            continue

        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        circ = contour_circularity(c)
        if circ < min_circ:
            continue

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

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        cand_center = (cx, cy + y0)

        if last_center is not None:
            dx = cand_center[0] - last_center[0]
            dy = cand_center[1] - last_center[1]
            if dx * dx + dy * dy > max_jump * max_jump:
                continue

        if radius > best_radius:
            best_radius = radius
            best = (x, y, radius, cand_center)

    return best


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


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video:", args.video)
        return

    if args.tune:
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

    lower = np.array([args.hmin, args.smin, args.vmin], dtype=np.uint8)
    upper = np.array([args.hmax, args.smax, args.vmax], dtype=np.uint8)
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
    max_jump = args.max_jump
    min_motion_ratio = args.min_motion_ratio

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        frame_i += 1
        if frame_i % 60 == 0:
            print("frame", frame_i)

        if args.tune:
            lower, upper, motion_thresh, roi_top, min_radius, max_radius, min_area, max_area, max_jump, min_circ = read_tuning()

        roi, y0 = compute_roi(frame, roi_top)
        hsv, blurred = hsv_mask(roi, lower, upper, hsv_erode, hsv_dilate)
        prev_gray_before = prev_gray
        motion, prev_gray = motion_mask(blurred, prev_gray, motion_thresh, motion_erode, motion_dilate)
        motion_valid = prev_gray_before is not None

        if args.use_motion and motion_valid:
            final_mask = cv2.bitwise_and(hsv, motion)
        else:
            final_mask = hsv

        best = select_ball_contour(
            final_mask,
            motion,
            motion_valid,
            min_motion_ratio,
            y0,
            last_center,
            min_radius,
            max_radius,
            min_area,
            max_area,
            min_circ,
            max_jump,
        )

        center = None
        if best is not None:
            x, y, radius, center = best
            cv2.circle(frame, (int(x), int(y + y0)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            last_center = center
            lost_frames = 0
        else:
            lost_frames += 1
            if lost_frames > args.lost_reset:
                last_center = None

        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args.buffer / float(i + 1)) * 2)
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

        cv2.line(frame, (0, y0), (frame.shape[1], y0), (0, 255, 255), 1)

        show_window("roi", roi, args.display_max_width, args.display_max_height)
        show_window("hsv_mask", hsv, args.display_max_width, args.display_max_height)
        show_window("motion_mask", motion, args.display_max_width, args.display_max_height)
        show_window("final_mask", final_mask, args.display_max_width, args.display_max_height)
        show_window("Ball Tracker", frame, args.display_max_width, args.display_max_height)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("Video done. Press any key in the Ball Tracker window to close.")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
