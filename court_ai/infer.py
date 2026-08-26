"""Predict court corners with CourtNet and write a track_ball_yolo calibration.

Given a video (or a clean-court image), median-averages to an empty court,
runs CourtNet to regress the 4 doubles corners, and writes a calibration JSON
(points in near_left, near_right, far_right, far_left order) plus an overlay
image for visual verification. net_points are derived from the homography.
"""
import argparse, os, json, sys
import numpy as np
import cv2
import torch

# Must match track_ball_yolo.calibration_fits_frame default margin: a calibration
# whose points fall further than this outside the frame is rejected downstream and
# tracking silently runs uncalibrated. Keep in sync with track_ball_yolo.py.
DOWNSTREAM_MARGIN_PX = 24

from court_ai.model import CourtNet, INPUT_SIZE
from court_ai.clean_court import clean_court_image
from court_ai.preprocess import to_linemap
from court_ai.fit_lines import refine_corners
from court_ai.court_geometry import CORNERS_WORLD, NET_Y, W_DOUBLES


def load_model(ckpt):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = CourtNet().to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()
    return m, dev


def predict_corners(model, dev, court_img):
    H, W = court_img.shape[:2]
    lm = to_linemap(court_img, size=INPUT_SIZE)  # domain-invariant 1-channel
    x = torch.from_numpy(lm).unsqueeze(0).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]  # (4,2) normalized
    return pred * np.array([W, H], dtype=np.float32)


def points_outside_frame(points, frame_shape, margin=DOWNSTREAM_MARGIN_PX):
    """Return the (x, y) points that fall further than `margin` outside the frame.

    Mirrors track_ball_yolo.calibration_fits_frame so infer.py can tell, before
    writing, whether the calibration it produced will be rejected downstream.
    """
    h, w = frame_shape[:2]
    return [(float(x), float(y)) for x, y in points
            if x < -margin or x > w + margin or y < -margin or y > h + margin]


def net_points_from_corners(corners_img):
    """Project world net endpoints through the homography implied by the corners."""
    Hmg = cv2.getPerspectiveTransform(CORNERS_WORLD, corners_img.astype(np.float32))
    net = np.array([[[0.0, NET_Y]], [[W_DOUBLES, NET_Y]]], dtype=np.float32)
    p = cv2.perspectiveTransform(net, Hmg).reshape(-1, 2)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="match video to calibrate")
    ap.add_argument("--court-image", help="use a precomputed clean-court image instead")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "_checkpoints", "courtnet.pt"))
    ap.add_argument("--out", required=True, help="output calibration JSON path")
    ap.add_argument("--overlay", help="output overlay PNG path (default: alongside --out)")
    ap.add_argument("--allow-offframe", action="store_true",
                    help="warn instead of failing when the calibration would be rejected downstream")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="skip fitting the predicted corners to the court line model "
                         "(CourtNet alone is several px off, which is metres of world "
                         "error at the far baseline)")
    args = ap.parse_args()

    court = cv2.imread(args.court_image) if args.court_image else clean_court_image(args.video)
    model, dev = load_model(args.ckpt)
    corners = predict_corners(model, dev, court)  # near_left, near_right, far_right, far_left
    if args.refine:
        corners, before, after = refine_corners(corners, court, verbose=True)
        print(f"line-model fit: score {before:.3f} -> {after:.3f}")
    net = net_points_from_corners(corners)

    # A model prediction can place corners/net well outside the frame (e.g. an
    # extrapolated near baseline). track_ball_yolo rejects such a calibration and
    # runs uncalibrated without a word, so surface it here rather than downstream.
    off = points_outside_frame(np.vstack([corners, net]), court.shape)

    calib = {"points": [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
             "net_points": [[round(float(x), 1), round(float(y), 1)] for x, y in net]}
    with open(args.out, "w") as fh:
        json.dump(calib, fh, indent=2)

    overlay = args.overlay or os.path.splitext(args.out)[0] + "_overlay.png"
    vis = court.copy()
    poly = corners.astype(np.int32)
    cv2.polylines(vis, [poly[[0, 1, 2, 3]]], True, (0, 255, 0), 2)
    names = ["near_left", "near_right", "far_right", "far_left"]
    for (x, y), nm in zip(corners, names):
        cv2.circle(vis, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(vis, nm, (int(x) + 12, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    for (x, y) in net:
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), -1)
    cv2.imwrite(overlay, vis)
    print("wrote", args.out, "and", overlay)
    print("corners:", calib["points"])

    if off:
        h, w = court.shape[:2]
        # Some cameras crop the near doubles corners, so off-frame points can be
        # correct; tell the caller the exact margin that accepts this calibration
        # rather than only that it will be rejected.
        needed = int(np.ceil(max(
            max(-x, x - w, -y, y - h) for x, y in off
        )))
        msg = (f"WARNING: {len(off)} calibration point(s) fall >±{DOWNSTREAM_MARGIN_PX}px "
               f"outside the {w}x{h} frame: {off}. track_ball_yolo will reject this "
               f"calibration and track UNCALIBRATED unless you pass "
               f"--court-calib-margin-px {needed} (or more). Check the overlay first: if "
               f"the court outline is wrong, fix the calibration instead of raising the "
               f"margin. Re-run with --allow-offframe to write the file anyway.")
        print(msg, file=sys.stderr)
        if not args.allow_offframe:
            sys.exit(2)


if __name__ == "__main__":
    main()
