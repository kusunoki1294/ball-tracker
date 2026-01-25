import argparse
from collections import deque

import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file name, e.g. tennis.mp4")
    ap.add_argument("--method", default="hsv", help="keep as hsv for now")
    ap.add_argument("--buffer", type=int, default=64, help="trail length")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video:", args.video)
        return

    # Starting HSV range for tennis-ball-ish yellow (you may tune later)
    lower = np.array([15, 60, 60], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)

    pts = deque(maxlen=args.buffer)
    last_center = None          # last known ball position (full-frame coords)
    MAX_JUMP_PX = 250           # how far the ball is allowed to move per frame (tune later)
    lost_frames = 0             # how many frames we failed to detect the ball
    prev_gray = None
    cv2.namedWindow("hsv_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("motion", cv2.WINDOW_NORMAL)
    frame_i = 0



    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            break

        frame_i += 1
        if frame_i % 60 == 0:
            print("frame", frame_i)

        h, w = frame.shape[:2]
        y0 = int(h * 0.45)          # ignore top 30%
        roi = frame[y0:h, 0:w]      # only process the lower part
        
        # Blur -> HSV -> mask
        blurred = cv2.GaussianBlur(roi, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # --- HSV mask (yellow color) ---
        hsv_mask = cv2.inRange(hsv, lower, upper)
        hsv_mask = cv2.erode(hsv_mask, None, iterations=2)
        hsv_mask = cv2.dilate(hsv_mask, None, iterations=2)

        # --- Motion mask (frame differencing) ---
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)


        motion_mask = np.zeros_like(gray)  # always exists

        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)

            # LOWER this threshold so motion actually shows up
            _, motion_mask = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)

            # for small moving objects (ball), dilation helps
            motion_mask = cv2.erode(motion_mask, None, iterations=1)
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        prev_gray = gray

        # --- Combine: yellow AND moving ---
        mask = hsv_mask

        # Debug windows (this is the key!)
        cv2.imshow("hsv_mask", hsv_mask)
        cv2.imshow("motion", motion_mask)
        cv2.imshow("mask", mask)


        # Find contours
        # Find contours
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        best = None
        best_radius = 0

    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # ball size filter (tune for your video)
        if radius < 1 or radius > 10:
            continue

        area = cv2.contourArea(c)
        if area < 5 or area > 300:
            continue

        # compute center of this contour (ROI coords)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # convert to FULL-FRAME coords (because ROI starts at y0)
        cand_center = (cx, cy + y0)

        # ---- THE IMPORTANT PART: "DON'T TELEPORT" FILTER ----
        if last_center is not None:
            dx = cand_center[0] - last_center[0]
            dy = cand_center[1] - last_center[1]
            if dx*dx + dy*dy > MAX_JUMP_PX * MAX_JUMP_PX:
                continue
        # -----------------------------------------------------

        # pick the biggest radius that passes filters
        if radius > best_radius:
            best_radius = radius
            best = (x, y, radius, cand_center)

        center = None

        if best is not None:
            x, y, radius, center = best

            cv2.circle(frame, (int(x), int(y + y0)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            last_center = center
            lost_frames = 0

        else:
            lost_frames += 1
            if lost_frames > 10:
                last_center = None   # reset if we lose the ball for a while

        pts.appendleft(center)


        # Draw trail
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args.buffer / float(i + 1)) * 2)
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)

        if frame is None or frame.size == 0:
            break
        cv2.imshow("Ball Tracker", frame)
        
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    print("Video done. Press any key in the Ball Tracker window to close.")
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

