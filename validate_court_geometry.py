"""Validate shared court geometry against the previous OpenCV projection path.

Run this with `.venv/bin/python`; the comparison path imports OpenCV.
"""

import glob
import json
import sys

import cv2
import numpy as np

from court_geometry import (
    build_inverse_court_homography,
    order_court_corners,
    project_to_court_world,
)


WORLD_CORNERS = np.array(
    [[0.0, 78.0], [36.0, 78.0], [36.0, 0.0], [0.0, 0.0]], dtype=np.float32
)
TEST_POINTS = (
    (0.0, 0.0),
    (320.0, 180.0),
    (640.0, 360.0),
    (960.0, 540.0),
    (1280.0, 720.0),
    (1600.0, 900.0),
    (1920.0, 1080.0),
    (1155.0, 174.0),
)
MAX_ALLOWED_DIFF_FT = 0.01


def cv2_inverse_homography(points):
    image = np.array(points, dtype=np.float32)
    homography = cv2.getPerspectiveTransform(WORLD_CORNERS, image)
    return np.linalg.inv(homography)


def cv2_project(point, inv_homography):
    source = np.array([[point]], dtype=np.float32)
    world = cv2.perspectiveTransform(source, inv_homography)[0][0]
    return float(world[0]), float(world[1])


def load_points(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    points = data.get("points")
    if not isinstance(points, list) or len(points) != 4:
        return None
    return order_court_corners([(float(x), float(y)) for x, y in points])


def main():
    paths = sorted(glob.glob("**/court_calib*.json", recursive=True))
    errors = []
    max_diff = 0.0
    checked = 0
    for path in paths:
        points = load_points(path)
        if not points:
            continue
        shared_inv = build_inverse_court_homography({"points": points})
        cv2_inv = cv2_inverse_homography(points)
        if shared_inv is None:
            errors.append(f"{path}: shared homography solver returned None")
            continue
        for point in TEST_POINTS:
            shared = project_to_court_world(point, shared_inv)
            expected = cv2_project(point, cv2_inv)
            diff = max(abs(shared[0] - expected[0]), abs(shared[1] - expected[1]))
            max_diff = max(max_diff, diff)
            checked += 1
            if diff > MAX_ALLOWED_DIFF_FT:
                errors.append(
                    f"{path} point {point}: diff {diff:.6f}ft exceeds "
                    f"{MAX_ALLOWED_DIFF_FT:.6f}ft"
                )
    if not checked:
        errors.append("no court calibration projections were checked")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"court geometry validation passed ({checked} projections, max diff {max_diff:.6f}ft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
