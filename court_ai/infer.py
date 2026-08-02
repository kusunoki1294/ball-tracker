"""Predict court corners with CourtNet and write a track_ball_yolo calibration.

Given a video (or a clean-court image), median-averages to an empty court,
runs CourtNet to regress the 4 doubles corners, and writes a calibration JSON
(points in near_left, near_right, far_right, far_left order) plus an overlay
image for visual verification. net_points are derived from the homography.
"""
import argparse, os, json
import numpy as np
import cv2
import torch

from court_ai.model import CourtNet, INPUT_SIZE
from court_ai.clean_court import clean_court_image
from court_ai.preprocess import to_linemap
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
    args = ap.parse_args()

    court = cv2.imread(args.court_image) if args.court_image else clean_court_image(args.video)
    model, dev = load_model(args.ckpt)
    corners = predict_corners(model, dev, court)  # near_left, near_right, far_right, far_left
    net = net_points_from_corners(corners)

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


if __name__ == "__main__":
    main()
