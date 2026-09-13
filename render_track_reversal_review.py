"""Render the detector-independent reversal inventory as a review-only video."""

import argparse
import csv
import json
import subprocess

import cv2
import numpy as np

import render_tennis_analysis as analysis_render


def load_reviews(path):
    with open(path, newline="") as handle:
        return {int(row["frame"]): row for row in csv.DictReader(handle)}


def load_support(path):
    if not path:
        return {}
    with open(path) as handle:
        data = json.load(handle)
    return {int(row["frame"]): bool(row["moving_tracks"])
            for row in data.get("events", [])}


def load_analysis_markers(path):
    if not path:
        return {}
    with open(path) as handle:
        analysis = json.load(handle)
    serve_ids = {
        attempt["bounce_id"]
        for point in analysis.get("points", [])
        for attempt in (point.get("serve_analysis") or {}).get("attempts") or []
        if attempt.get("bounce_id")
    }
    markers = {}
    number = 0
    for bounce in analysis.get("bounces", []):
        point = bounce.get("point")
        if not point:
            continue
        number += 1
        visible = analysis_render.analysis_bounce_visible(bounce, serve_ids)
        provisional = analysis_render.bounce_is_provisional(bounce, serve_ids)
        if visible:
            label = f"B{number}" + ("?" if provisional else "")
            color = (0, 0, 255) if not provisional else (0, 165, 255)
        else:
            label = analysis_render.candidate_class(bounce, serve_ids)
            color = analysis_render.CANDIDATE_COLORS.get(label, (140, 140, 140))
        markers.setdefault(int(bounce["frame"]), []).append({
            "point": (int(round(point[0])), int(round(point[1]))),
            "label": label,
            "color": color,
            "visible": visible,
        })
    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--support-json")
    parser.add_argument("--analysis",
                        help="optional analysis JSON; overlays existing bounce candidates")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reviews = load_reviews(args.reviews)
    support = load_support(args.support_json)
    analysis_markers = load_analysis_markers(args.analysis)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", args.video],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    fps = numerator / denominator
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.output],
        stdin=subprocess.PIPE,
    )

    decoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", args.video,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    frame_size = width * height * 3
    frame = 0
    active_analysis_markers = []
    while data := decoder.stdout.read(frame_size):
        if len(data) != frame_size:
            break
        # The raw stream is already BGR; reshape without another codec pass.
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3)).copy()
        row = reviews.get(frame)
        for marker in analysis_markers.get(frame, []):
            active_analysis_markers.append({"expires": frame + 45, **marker})
        active_analysis_markers = [
            marker for marker in active_analysis_markers if marker["expires"] >= frame
        ]
        for marker in active_analysis_markers:
            x, y = marker["point"]
            cv2.drawMarker(image, (x, y), marker["color"],
                           markerType=cv2.MARKER_TILTED_CROSS if marker["visible"]
                           else cv2.MARKER_SQUARE,
                           markerSize=14, thickness=2)
            cv2.putText(image, marker["label"], (x + 8, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, marker["color"], 2,
                        cv2.LINE_AA)
        if row:
            x, y = round(float(row["image_x"])), round(float(row["image_y"]))
            color = (0, 215, 255) if support.get(frame, False) else (160, 160, 160)
            cv2.circle(image, (x, y), 16, color, 2)
            suffix = " YOLO" if support.get(frame, False) else " no-YOLO"
            cv2.putText(image, f"R f{frame}{suffix}", (x + 18, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA)
        cv2.rectangle(image, (18, 18), (480, 58), (20, 20, 20), -1)
        cv2.putText(image, "TRACK REVERSALS - REVIEW ONLY", (28, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 215, 255), 2,
                    cv2.LINE_AA)
        cv2.rectangle(image, (18, 66), (390, 106), (20, 20, 20), -1)
        cv2.putText(image, "red/orange: detector bounce   yellow/gray: reversal review", (28, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1,
                    cv2.LINE_AA)
        encoder.stdin.write(image.tobytes())
        frame += 1

    decoder.stdout.close()
    decoder.wait()
    encoder.stdin.close()
    encoder.wait()
    print(f"wrote {args.output}: {frame} frames, "
          f"{len(reviews)} review events")


if __name__ == "__main__":
    main()
