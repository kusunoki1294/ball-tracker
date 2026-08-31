"""Export pre-roll strips for timeline review-priority frames.

The serve/contact crop sheet shows whether a strike happened. This artifact shows
the two seconds before selected contacts, because that is the evidence that
separates a true serve from a mid-rally overhead.
"""

import argparse
import html
import json
import os

import cv2

from render_timeline_hypotheses import open_frame_reader


OFFSETS = (-60, -50, -40, -30, -20, -10, 0)
OUTPUT_WIDTH = 420
BALL_COLOR = (0, 255, 255)
BALL_OUTLINE = (0, 0, 0)
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


def resize_frame(frame):
    height, width = frame.shape[:2]
    scale = OUTPUT_WIDTH / float(width)
    resized = cv2.resize(
        frame,
        (OUTPUT_WIDTH, int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def draw_ball_marker(image, center, scale):
    if center:
        x = int(round(center[0] * scale))
        y = int(round(center[1] * scale))
        cv2.circle(image, (x, y), 9, BALL_OUTLINE, 4, cv2.LINE_AA)
        cv2.circle(image, (x, y), 9, BALL_COLOR, 2, cv2.LINE_AA)
        cv2.line(image, (x - 15, y), (x + 15, y), BALL_OUTLINE, 4, cv2.LINE_AA)
        cv2.line(image, (x, y - 15), (x, y + 15), BALL_OUTLINE, 4, cv2.LINE_AA)
        cv2.line(image, (x - 15, y), (x + 15, y), BALL_COLOR, 2, cv2.LINE_AA)
        cv2.line(image, (x, y - 15), (x, y + 15), BALL_COLOR, 2, cv2.LINE_AA)
        return
    cv2.rectangle(image, (6, 6), (150, 29), TEXT_BG, -1)
    cv2.putText(
        image,
        "ball not tracked",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def write_frame(image, path, center):
    if image is None:
        raise RuntimeError(f"failed writing {path}: no image")
    resized, scale = resize_frame(image)
    draw_ball_marker(resized, center, scale)
    if not cv2.imwrite(path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise RuntimeError(f"failed writing {path}")


def extract_clip_frames(video, jobs, by_frame):
    reader = open_frame_reader(video)
    jobs = sorted(jobs, key=lambda item: item[0])
    current_frame = 0
    try:
        index = 0
        while index < len(jobs):
            target_frame = jobs[index][0]
            while current_frame < target_frame:
                ok, image = reader.read()
                current_frame += 1
                if not ok or image is None:
                    raise RuntimeError(f"failed extracting frame {target_frame}")
            while index < len(jobs) and jobs[index][0] == target_frame:
                _, output_path = jobs[index]
                write_frame(image, output_path, ball_center(by_frame.get(target_frame)))
                index += 1
        while True:
            ok, _ = reader.read()
            if not ok:
                break
    finally:
        reader.release()


def asset_name(card_index, frame, offset):
    return f"preroll_{card_index:03d}_f{frame}_offset_{offset:+d}.jpg"


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
            strip = []
            for offset in OFFSETS:
                target = max(1, frame + offset)
                name = asset_name(card_index, frame, offset)
                jobs.append((target, os.path.join(assets_dir, name)))
                rel = os.path.join(os.path.basename(assets_dir), name)
                label = "contact" if offset == 0 else f"{offset:+d}f"
                strip.append(
                    "<figure>"
                    f"<img src=\"{esc(rel)}\" alt=\"{esc(clip['label'])} frame {target}\">"
                    f"<figcaption>{esc(label)} · f{target}</figcaption>"
                    "</figure>"
                )
            cards.append(
                "<article>"
                f"<h2>{esc(clip['label'])} f{frame}</h2>"
                f"<p><strong>{esc(item.get('kind'))}</strong> — {esc(item.get('note'))}</p>"
                f"<div class=\"strip\">{''.join(strip)}</div>"
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
    .strip {{ display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 1px; background: #111; }}
    figure {{ margin: 0; position: relative; background: #111; }}
    img {{ width: 100%; display: block; background: #111; }}
    figcaption {{ position: absolute; left: 5px; bottom: 5px; background: rgba(0,0,0,.72); color: #fff; font-size: 11px; padding: 3px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Timeline Pre-Roll Review</h1>
  <div class="notice"><strong>Review artifact only.</strong> Contact posture alone cannot
  distinguish a serve from a rally overhead. These strips show the two seconds before
  selected contacts so a reviewer can look for an incoming live ball. Yellow crosshairs
  mark tracked ball positions; "ball not tracked" means the tracker had no ball for that
  frame.</div>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>
"""
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(document)
    print(f"wrote {output} ({card_index} pre-roll strips)")


def main():
    args = parse_args()
    export_review([parse_clip(item) for item in args.clip], args.output)


if __name__ == "__main__":
    main()
