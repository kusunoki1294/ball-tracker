"""Export a visual review sheet for serve-contact hypotheses.

This is a review artifact, not a scorer. It reads timeline hypothesis JSON and a
clean source video, extracts the listed contact frames, and writes an HTML page
with stills and the evidence attached to each contact.
"""

import argparse
import html
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Clean input video.")
    parser.add_argument("--hypotheses", required=True, help="Timeline hypotheses JSON.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--include-suppressed",
        action="store_true",
        help="Include suppressed rally motions as separate review cards.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def asset_dir_for(output_path):
    root, _ = os.path.splitext(output_path)
    return root + "_assets"


def collect_contacts(data, include_suppressed):
    contacts = []
    for hypothesis in data.get("hypotheses") or []:
        for attempt in hypothesis.get("attempts") or []:
            contacts.append((hypothesis, attempt, "hypothesis"))
        if include_suppressed:
            for attempt in hypothesis.get("suppressed_rally_motions") or []:
                contacts.append((hypothesis, attempt, "suppressed"))
    return sorted(contacts, key=lambda item: int(item[1]["contact_frame"]))


def extract_frame(video_path, frame, output_path):
    # Timeline frame numbers are 1-based; ffmpeg select's n is 0-based.
    selector = f"select=eq(n\\,{int(frame) - 1})"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            selector,
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting frame {frame}: {result.stderr.strip()}")


def landing_text(attempt):
    landing = attempt.get("landing") or {}
    reason = landing.get("reason") or "unknown"
    if landing.get("bounce_frame") is None:
        return reason
    return f"{landing.get('result', 'unknown')} at f{landing.get('bounce_frame')} ({reason})"


def review_text(attempt):
    reasons = attempt.get("review_reasons") or []
    return ", ".join(str(reason) for reason in reasons) if reasons else "none"


def write_html(path, title, contacts, assets_dir, source_video, source_json):
    rel_assets = os.path.relpath(assets_dir, os.path.dirname(path) or ".")
    cards = []
    for index, (hypothesis, attempt, kind) in enumerate(contacts, start=1):
        frame = int(attempt["contact_frame"])
        image = f"contact_{index:03d}_f{frame}.jpg"
        image_path = html.escape(os.path.join(rel_assets, image), quote=True)
        cards.append(
            "<article>"
            f"<img src=\"{image_path}\" alt=\"frame {frame}\">"
            f"<h2>{esc(hypothesis.get('id'))} · f{frame}</h2>"
            f"<p><strong>{esc(kind)}</strong> · server {esc(attempt.get('server'))} · "
            f"{esc(attempt.get('source'))} · {esc(attempt.get('confidence'))}</p>"
            f"<p>Landing: {esc(landing_text(attempt))}</p>"
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
    .meta {{ color: #5a6673; margin: 0 0 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    article {{ background: #fff; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; display: block; background: #111; }}
    h2 {{ font-size: 17px; margin: 12px 14px 6px; }}
    p {{ margin: 6px 14px 12px; color: #4b5663; }}
    strong {{ color: #16202b; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p class="meta">Source video: {esc(source_video)}<br>Hypotheses: {esc(source_json)}</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def main():
    args = parse_args()
    data = load_json(args.hypotheses)
    contacts = collect_contacts(data, args.include_suppressed)
    ensure_dir(os.path.dirname(args.output) or ".")
    assets_dir = asset_dir_for(args.output)
    ensure_dir(assets_dir)
    for index, (_, attempt, _) in enumerate(contacts, start=1):
        frame = int(attempt["contact_frame"])
        extract_frame(args.video, frame, os.path.join(assets_dir, f"contact_{index:03d}_f{frame}.jpg"))
    write_html(
        args.output,
        "Serve Contact Review",
        contacts,
        assets_dir,
        args.video,
        args.hypotheses,
    )
    print(f"wrote {args.output} ({len(contacts)} contacts)")


if __name__ == "__main__":
    main()
