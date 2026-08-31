"""Export pre-roll trajectory cards for timeline review-priority frames.

The serve/contact crop sheet shows whether a strike happened. This artifact shows
the tracked ball trail before selected contacts, because that is the evidence
that separates a true serve from a mid-rally overhead.
"""

import argparse
import html
import json
import os

import cv2

from render_timeline_hypotheses import open_frame_reader


WINDOW_FRAMES = 60
OUTPUT_WIDTH = 420
BALL_OUTLINE = (0, 0, 0)
TRAIL_START = (36, 99, 235)
TRAIL_END = (239, 68, 68)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="LABEL=VIDEO=FRAME:KIND:NOTE[,FRAME:KIND:NOTE...]",
        help="Clip and priority frames. Repeat per clip.",
    )
    return parser.parse_args()


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def asset_dir_for(output_path):
    root, _ = os.path.splitext(output_path)
    return root + "_assets"


def clear_generated_assets(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.startswith("preroll_") and name.lower().endswith(".jpg"):
            os.remove(os.path.join(path, name))


def parse_clip(raw):
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--clip expects LABEL=VIDEO=FRAME:KIND:NOTE[,...]")
    label, video, raw_items = parts
    items = []
    for raw_item in raw_items.split(","):
        frame, kind, note = (raw_item.split(":", 2) + ["", ""])[:3]
        items.append({"frame": int(frame), "kind": kind, "note": note})
    return {"label": label, "video": video, "items": items}


def read_tracking_by_frame(path):
    if not path:
        return {}
    by_frame = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = row.get("frame")
            if frame is not None:
                by_frame[int(frame)] = row
    return by_frame


def ball_center(row):
    ball = (row or {}).get("ball") or {}
    center = ball.get("center")
    if center and len(center) == 2:
        return float(center[0]), float(center[1])
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def ball_size(row):
    ball = (row or {}).get("ball") or {}
    bbox = ball.get("bbox")
    if bbox and len(bbox) == 4:
        return max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    return None


def resize_frame(frame):
    height, width = frame.shape[:2]
    scale = OUTPUT_WIDTH / float(width)
    resized = cv2.resize(
        frame,
        (OUTPUT_WIDTH, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def trail_color(index, total):
    if total <= 1:
        ratio = 1.0
    else:
        ratio = index / float(total - 1)
    return tuple(
        int(round(TRAIL_START[channel] * (1.0 - ratio) + TRAIL_END[channel] * ratio))
        for channel in range(3)
    )


def draw_trail(image, points, scale):
    if not points:
        cv2.rectangle(image, (6, 6), (190, 29), TEXT_BG, -1)
        cv2.putText(
            image,
            "no tracked ball in window",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        return
    scaled = []
    for frame, center, size in points:
        radius = 3
        if size is not None:
            radius = max(2, int(round(size * scale / 2.0)))
        scaled.append((frame, int(round(center[0] * scale)), int(round(center[1] * scale)), radius))
    for index in range(1, len(scaled)):
        color = trail_color(index, len(scaled))
        _, x0, y0, _ = scaled[index - 1]
        _, x1, y1, _ = scaled[index]
        cv2.line(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    for index, (_, x, y, radius) in enumerate(scaled):
        color = trail_color(index, len(scaled))
        cv2.circle(image, (x, y), radius + 1, BALL_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    _, x, y, radius = scaled[-1]
    cv2.circle(image, (x, y), radius + 5, BALL_OUTLINE, 3, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 5, TRAIL_END, 2, cv2.LINE_AA)


def write_trail_frame(image, path, points):
    if image is None:
        raise RuntimeError(f"failed writing {path}: no image")
    resized, scale = resize_frame(image)
    draw_trail(resized, points, scale)
    if not cv2.imwrite(path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise RuntimeError(f"failed writing {path}")


def extract_clip_frames(video, jobs, by_frame):
    reader = open_frame_reader(video)
    jobs = sorted(jobs, key=lambda item: item["target"])
    current_frame = 0
    try:
        index = 0
        while index < len(jobs):
            target_frame = jobs[index]["target"]
            while current_frame < target_frame:
                ok, image = reader.read()
                current_frame += 1
                if not ok or image is None:
                    raise RuntimeError(f"failed extracting frame {target_frame}")
            while index < len(jobs) and jobs[index]["target"] == target_frame:
                job = jobs[index]
                write_trail_frame(image, job["output"], job["points"])
                index += 1
        while True:
            ok, _ = reader.read()
            if not ok:
                break
    finally:
        reader.release()


def asset_name(card_index, frame):
    return f"preroll_{card_index:03d}_f{frame}_trail.jpg"


def tracked_points(by_frame, start_frame, end_frame):
    points = []
    for frame in range(start_frame, end_frame + 1):
        row = by_frame.get(frame)
        center = ball_center(row)
        if center:
            points.append((frame, center, ball_size(row)))
    return points


def export_review(clips, output):
    ensure_dir(os.path.dirname(output) or ".")
    assets_dir = asset_dir_for(output)
    ensure_dir(assets_dir)
    clear_generated_assets(assets_dir)

    cards = []
    card_index = 0
    for clip in clips:
        jobs = []
        by_frame = read_tracking_by_frame(clip.get("jsonl"))
        for item in clip["items"]:
            card_index += 1
            frame = int(item["frame"])
            interval_source = "fixed 2-second window"
            start_frame = max(1, frame - WINDOW_FRAMES)
            if item.get("trail_start_frame") is not None:
                start_frame = max(1, int(item["trail_start_frame"]))
                interval_source = "previous accepted contact interval"
            points = tracked_points(by_frame, start_frame, frame)
            tracked = len(points)
            total = frame - start_frame + 1
            coverage = round(100.0 * tracked / float(max(1, total)), 1)
            name = asset_name(card_index, frame)
            rel = os.path.join(os.path.basename(assets_dir), name)
            jobs.append(
                {
                    "target": frame,
                    "output": os.path.join(assets_dir, name),
                    "points": points,
                }
            )
            cards.append(
                "<article>"
                f"<h2>{esc(clip['label'])} f{frame}</h2>"
                f"<p><strong>{esc(item.get('kind'))}</strong> — {esc(item.get('note'))}</p>"
                f"<p class=\"coverage\">trail interval: f{start_frame}-f{frame} "
                f"({esc(interval_source)}); tracked coverage: {tracked}/{total} frames ({coverage}%)</p>"
                "<figure class=\"trail\">"
                f"<img src=\"{esc(rel)}\" alt=\"{esc(clip['label'])} tracked-ball trail ending at frame {frame}\">"
                "<figcaption>blue = oldest tracked ball, red ring = latest tracked ball; dot size follows tracked bbox size</figcaption>"
                "</figure>"
                "</article>"
            )
        extract_clip_frames(clip["video"], jobs, by_frame)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Timeline Pre-Roll Review</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 28px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #17202a; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    .notice {{ border-left: 5px solid #d97706; background: #fff7ed; padding: 13px 15px; margin: 0 0 22px; max-width: 1060px; }}
    .grid {{ display: grid; gap: 18px; }}
    article {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
    h2 {{ font-size: 17px; margin: 12px 14px 6px; }}
    p {{ margin: 6px 14px 12px; color: #4b5663; }}
    .coverage {{ color: #111827; font-weight: 650; }}
    figure {{ margin: 0; position: relative; background: #111; }}
    figure.trail {{ border-top: 1px solid #111; }}
    img {{ width: 100%; display: block; background: #111; }}
    figcaption {{ position: absolute; left: 5px; bottom: 5px; background: rgba(0,0,0,.72); color: #fff; font-size: 11px; padding: 3px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Timeline Pre-Roll Review</h1>
  <div class="notice"><strong>Review artifact only.</strong> Contact posture alone cannot
  distinguish a serve from a rally overhead. Each card overlays an image-space tracked-ball trail
  on the contact frame. Controls with a known previous contact use that full interval; other cards
  fall back to the previous two seconds. The trail deliberately does not use ground-projected court
  side, because airborne balls make that projection unreliable. Read the tracked coverage line before
  trusting missing trail segments.</div>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>
"""
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"wrote {output} ({card_index} pre-roll trail cards)")


def main():
    args = parse_args()
    export_review([parse_clip(item) for item in args.clip], args.output)


if __name__ == "__main__":
    main()
