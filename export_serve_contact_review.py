"""Export a visual review sheet for serve-contact hypotheses.

This is a review artifact, not a scorer. It reads timeline hypothesis JSON and a
clean source video, extracts the listed contact frames, and writes an HTML page
with image strips and the evidence attached to each contact.
"""

import argparse
import html
import json
import os

import cv2

FRAME_OFFSETS = (-4, -2, 0, 2)
CROP_SCALE = 3.0
MIN_CROP_PX = 320
MAX_CROP_PX = 900
OUTPUT_CROP_PX = 520


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Clean input video.")
    parser.add_argument("--hypotheses", required=True, help="Timeline hypotheses JSON.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--jsonl",
        default="",
        help="Tracked JSONL log used to crop around the claimed server.",
    )
    parser.add_argument(
        "--include-suppressed",
        action="store_true",
        help="Include suppressed rally motions as separate review cards.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
        if name.startswith("contact_") and name.lower().endswith(".jpg"):
            os.remove(os.path.join(path, name))


def collect_contacts(data, include_suppressed):
    contacts = []
    for hypothesis in data.get("hypotheses") or []:
        for attempt in hypothesis.get("attempts") or []:
            contacts.append((hypothesis, attempt, "hypothesis"))
        if include_suppressed:
            for attempt in hypothesis.get("suppressed_rally_motions") or []:
                contacts.append((hypothesis, attempt, "suppressed"))
    return sorted(contacts, key=lambda item: int(item[1]["contact_frame"]))


def player_box_for(by_frame, frame, server):
    row = by_frame.get(int(frame)) or {}
    key = "player_near" if server == "near" else "player_far"
    box = row.get(key) or {}
    bbox = box.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return [float(value) for value in bbox]


def crop_rect(frame_shape, bbox):
    height, width = frame_shape[:2]
    if not bbox:
        return 0, 0, width, height
    x1, y1, x2, y2 = bbox
    box_h = max(1.0, y2 - y1)
    crop_size = int(round(max(MIN_CROP_PX, min(MAX_CROP_PX, box_h * CROP_SCALE))))
    crop_size = max(1, min(crop_size, width, height))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = int(round(center_x - crop_size / 2.0))
    top = int(round(center_y - crop_size / 2.0))
    left = max(0, min(width - crop_size, left))
    top = max(0, min(height - crop_size, top))
    return left, top, crop_size, crop_size


def extract_frame(capture, frame, output_path, bbox=None):
    # Timeline frame numbers are 1-based; OpenCV frame positions are 0-based.
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"failed extracting frame {frame}")
    left, top, width, height = crop_rect(image.shape, bbox)
    cropped = image[top : top + height, left : left + width]
    if bbox:
        cropped = cv2.resize(cropped, (OUTPUT_CROP_PX, OUTPUT_CROP_PX), interpolation=cv2.INTER_CUBIC)
    if not cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"failed writing {output_path}")


def landing_text(attempt):
    landing = attempt.get("landing") or {}
    reason = landing.get("reason") or "unknown"
    if landing.get("bounce_frame") is None:
        return f"hypothesised landing: {reason}"
    return (
        f"hypothesised landing: {landing.get('result', 'unknown')} "
        f"at f{landing.get('bounce_frame')} ({reason})"
    )


def review_text(attempt):
    reasons = attempt.get("review_reasons") or []
    return ", ".join(str(reason) for reason in reasons) if reasons else "none"


def contact_assets(index, frame):
    return [
        (offset, f"contact_{index:03d}_f{frame}_offset_{offset:+d}.jpg")
        for offset in FRAME_OFFSETS
    ]


def write_html(path, title, contacts, assets_dir, source_video, source_json):
    rel_assets = os.path.relpath(assets_dir, os.path.dirname(path) or ".")
    cards = []
    for index, (hypothesis, attempt, kind) in enumerate(contacts, start=1):
        frame = int(attempt["contact_frame"])
        strip = []
        for offset, image in contact_assets(index, frame):
            image_path = html.escape(os.path.join(rel_assets, image), quote=True)
            label = "contact" if offset == 0 else f"{offset:+d}f"
            strip.append(
                "<figure>"
                f"<img src=\"{image_path}\" alt=\"frame {frame + offset}\">"
                f"<figcaption>{esc(label)} · f{frame + offset}</figcaption>"
                "</figure>"
            )
        cards.append(
            "<article>"
            f"<h2>{esc(hypothesis.get('display_id') or hypothesis.get('id'))} · f{frame}</h2>"
            f"<p><strong>{esc(kind)}</strong> · claimed server {esc(attempt.get('server'))} · "
            f"{esc(attempt.get('source'))} · {esc(attempt.get('confidence'))}</p>"
            f"<div class=\"strip\">{''.join(strip)}</div>"
            f"<p>{esc(landing_text(attempt))}</p>"
            f"<p>Review: {esc(review_text(attempt))}</p>"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 28px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #17202a; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    .meta {{ color: #5a6673; margin: 0 0 16px; }}
    .notice {{ border-left: 5px solid #d97706; background: #fff7ed; padding: 13px 15px; margin: 0 0 22px; max-width: 1060px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
    article {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
    .strip {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: #111; }}
    figure {{ margin: 0; position: relative; background: #111; }}
    img {{ width: 100%; display: block; background: #111; }}
    figcaption {{ position: absolute; left: 6px; bottom: 6px; background: rgba(0,0,0,.72); color: #fff; font-size: 12px; padding: 3px 5px; border-radius: 4px; }}
    h2 {{ font-size: 17px; margin: 12px 14px 6px; }}
    p {{ margin: 6px 14px 12px; color: #4b5663; }}
    strong {{ color: #16202b; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p class="meta">Source video: {esc(source_video)}<br>Hypotheses: {esc(source_json)}</p>
  <div class="notice"><strong>Review artifact only.</strong> These cards show timeline hypotheses and suppressed serve-like motions. They do not score points, certify winners, or turn hypothesised landings into calls. Game 2 remains unverified until a person checks the strips.</div>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def export_review(video, hypotheses, output, jsonl="", include_suppressed=False):
    data = load_json(hypotheses)
    by_frame = read_tracking_by_frame(jsonl)
    contacts = collect_contacts(data, include_suppressed)
    ensure_dir(os.path.dirname(output) or ".")
    assets_dir = asset_dir_for(output)
    ensure_dir(assets_dir)
    clear_generated_assets(assets_dir)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"failed opening video {video}")
    for index, (_, attempt, _) in enumerate(contacts, start=1):
        frame = int(attempt["contact_frame"])
        bbox = player_box_for(by_frame, frame, attempt.get("server"))
        for offset, image in contact_assets(index, frame):
            extract_frame(
                capture,
                max(1, frame + offset),
                os.path.join(assets_dir, image),
                bbox,
            )
    capture.release()
    write_html(
        output,
        "Serve Contact Review",
        contacts,
        assets_dir,
        video,
        hypotheses,
    )
    print(f"wrote {output} ({len(contacts)} contacts)")


def main():
    args = parse_args()
    export_review(
        args.video,
        args.hypotheses,
        args.output,
        jsonl=args.jsonl,
        include_suppressed=args.include_suppressed,
    )


if __name__ == "__main__":
    main()
