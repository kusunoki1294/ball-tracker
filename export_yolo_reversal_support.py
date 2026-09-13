"""Measure short-window YOLO support for track-reversal review events.

The output is evidence for review, not a bounce decision. A persistent moving
YOLO detection can still be a toss, racket contact, or a wrong association.
"""

import argparse
import csv
import json
import math
import subprocess

import numpy as np
from ultralytics import YOLO


def probe_video(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def targets_from_csv(path):
    with open(path, newline="") as handle:
        return sorted({int(row["frame"]) for row in csv.DictReader(handle)
                       if row["near_existing_candidate"] == "False"})


def detections(video, targets, model_path, confidence, image_size, radius):
    width, height = probe_video(video)
    wanted = {frame for target in targets
              for frame in range(max(0, target - radius), target + radius + 1)}
    decoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", video,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    model = YOLO(model_path)
    frame_size = width * height * 3
    by_frame = {}
    frame = 0
    while data := decoder.stdout.read(frame_size):
        if len(data) != frame_size:
            break
        if frame in wanted:
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            result = model.predict(image, conf=confidence, imgsz=image_size,
                                   device="cpu", verbose=False)[0]
            current = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                current.append({
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "confidence": float(box.conf[0]),
                })
            by_frame[frame] = current
        frame += 1
    decoder.stdout.close()
    decoder.wait()
    return width, height, frame, by_frame


def moving_tracks(by_frame, target, radius):
    tracks = []
    for frame in range(max(0, target - radius), target + radius + 1):
        for detection in by_frame.get(frame, []):
            options = [track for track in tracks
                       if track[-1]["frame"] == frame - 1
                       and math.hypot(detection["x"] - track[-1]["x"],
                                      detection["y"] - track[-1]["y"]) < 90]
            if options:
                track = min(options, key=lambda item: math.hypot(
                    detection["x"] - item[-1]["x"],
                    detection["y"] - item[-1]["y"]))
                track.append({"frame": frame, **detection})
            else:
                tracks.append([{"frame": frame, **detection}])
    moving = []
    for track in tracks:
        displacement = math.hypot(track[-1]["x"] - track[0]["x"],
                                  track[-1]["y"] - track[0]["y"])
        if len(track) >= 3 and displacement >= 12:
            moving.append({
                "frames": len(track),
                "displacement_px": round(displacement, 2),
                "max_confidence": round(max(item["confidence"] for item in track), 3),
            })
    return moving


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--model", default="vids/models/tennisball.pt")
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    targets = targets_from_csv(args.reviews)
    width, height, decoded, by_frame = detections(
        args.video, targets, args.model, args.confidence, args.image_size, args.radius
    )
    rows = [{"frame": frame, "moving_tracks": moving_tracks(by_frame, frame, args.radius),
             "status": "review_required"} for frame in targets]
    summary = {
        "source_video": args.video,
        "model": args.model,
        "decoded_frames": decoded,
        "video_size": [width, height],
        "window_radius_frames": args.radius,
        "review_event_count": len(rows),
        "not_bounce_truth": True,
        "status": "review_required",
        "events": rows,
        "notes": [
            "Moving YOLO tracks are evidence only; they may be tosses, racket contacts, or wrong associations.",
            "A production recovery path additionally needs a validated rebound and contact-exclusion test.",
        ],
    }
    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {args.output_json}: {len(rows)} reversal events, "
          f"{decoded} decoded frames")


if __name__ == "__main__":
    main()
