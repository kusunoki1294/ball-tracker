"""Render timeline hypothesis overlays onto a clean video.

This consumes only `timeline_hypotheses.py` JSON. It deliberately does not read
analysis JSON and does not render scored points; the overlay is for reviewing
automated timeline hypotheses in context.
"""

import argparse
import json
import os
import subprocess
import tempfile

import cv2


PANEL_BG = (20, 24, 28)
PANEL_BORDER = (80, 90, 100)
TEXT = (245, 248, 250)
MUTED = (175, 184, 194)
HIGH = (90, 210, 115)
UNCERTAIN = (40, 180, 255)
REVIEW = (80, 120, 255)
SHADOW = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Clean input video.")
    parser.add_argument("--hypotheses", required=True, help="Timeline hypotheses JSON.")
    parser.add_argument("--output", required=True, help="Output annotated video.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for smoke renders.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def open_writer(path, width, height, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    codecs = ("MJPG", "mp4v", "avc1") if ext != ".mp4" else ("mp4v", "avc1", "MJPG")
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def render_target(output_path):
    if os.path.splitext(output_path)[1].lower() != ".mp4":
        return output_path, None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".avi",
        dir=os.path.dirname(output_path) or ".",
        delete=False,
    )
    handle.close()
    return handle.name, output_path


def transcode_to_mp4(source_path, output_path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            source_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}: {result.stderr.strip()}")


def draw_text(frame, text, origin, scale=0.58, color=TEXT, thickness=1):
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        SHADOW,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def confidence_color(confidence):
    return HIGH if confidence == "high" else UNCERTAIN


def active_hypothesis(hypotheses, frame):
    for item in hypotheses:
        if int(item["start_frame"]) <= frame <= int(item["end_frame"]):
            return item
    return None


def attempts_by_frame(hypotheses):
    result = {}
    for hypothesis in hypotheses:
        for attempt in hypothesis.get("attempts") or []:
            frame = attempt.get("contact_frame")
            if frame is None:
                continue
            result.setdefault(int(frame), []).append((hypothesis, attempt))
    return result


def draw_panel(frame, hypothesis, frame_index, total_frames):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (16, 16), (min(width - 16, 760), 164), PANEL_BG, -1)
    cv2.rectangle(frame, (16, 16), (min(width - 16, 760), 164), PANEL_BORDER, 1)
    draw_text(frame, "TIMELINE HYPOTHESES - NOT SCORING TRUTH", (28, 42), 0.62, REVIEW, 2)
    draw_text(frame, f"frame {frame_index}/{total_frames}", (28, 68), 0.52, MUTED, 1)
    if not hypothesis:
        draw_text(frame, "No active hypothesis on this frame", (28, 102), 0.62, TEXT, 1)
        return

    first = (hypothesis.get("attempts") or [{}])[0]
    landing = (first.get("landing") or {}).get("reason") or "unknown"
    line1 = (
        f"{hypothesis.get('id')}  {hypothesis.get('confidence')} "
        f"score={hypothesis.get('confidence_score')}  server={first.get('server')}"
    )
    line2 = (
        f"serves(hypothesis)={hypothesis.get('serve_count')}  "
        f"first_contact=f{first.get('contact_frame')}  landing={landing}"
    )
    review = ", ".join(hypothesis.get("review_reasons") or [])
    line3 = "review: " + (review if review else "none")
    draw_text(frame, fit_text(line1, 95), (28, 98), 0.58, confidence_color(hypothesis.get("confidence")), 2)
    draw_text(frame, fit_text(line2, 105), (28, 124), 0.52, TEXT, 1)
    draw_text(frame, fit_text(line3, 105), (28, 148), 0.46, MUTED if not review else REVIEW, 1)


def draw_timeline(frame, hypotheses, frame_index, total_frames):
    height, width = frame.shape[:2]
    left = 24
    right = width - 24
    y = height - 48
    cv2.line(frame, (left, y), (right, y), (90, 96, 104), 2)
    for hypothesis in hypotheses:
        start = int(hypothesis["start_frame"])
        end = int(hypothesis["end_frame"])
        x1 = left + int((right - left) * (start - 1) / max(1, total_frames - 1))
        x2 = left + int((right - left) * (end - 1) / max(1, total_frames - 1))
        color = confidence_color(hypothesis.get("confidence"))
        thickness = 8 if start <= frame_index <= end else 4
        cv2.line(frame, (x1, y), (max(x1 + 1, x2), y), color, thickness)
    cursor_x = left + int((right - left) * (frame_index - 1) / max(1, total_frames - 1))
    cv2.line(frame, (cursor_x, y - 18), (cursor_x, y + 18), (255, 255, 255), 2)
    draw_text(frame, "timeline hypotheses", (left, y - 24), 0.42, MUTED, 1)


def draw_contact_events(frame, events, frame_index):
    current = events.get(frame_index) or []
    for offset, (hypothesis, attempt) in enumerate(current):
        y = 194 + (offset * 26)
        text = (
            f"{hypothesis.get('id')} serve contact "
            f"attempt {attempt.get('attempt')} ({attempt.get('source')})"
        )
        draw_text(frame, fit_text(text, 90), (28, y), 0.55, confidence_color(hypothesis.get("confidence")), 2)


def main():
    args = parse_args()
    data = load_json(args.hypotheses)
    hypotheses = data.get("hypotheses") or []
    events = attempts_by_frame(hypotheses)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or data.get("fps") or 30.0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    hypothesis_frames = int(data.get("frame_range", {}).get("end_frame") or 0)
    total_frames = max(video_frames, hypothesis_frames, 1)
    writer_path, mp4_output = render_target(args.output)
    writer = open_writer(writer_path, width, height, fps)
    if writer is None:
        raise RuntimeError(f"Could not open output writer: {writer_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            hypothesis = active_hypothesis(hypotheses, frame_index)
            draw_panel(frame, hypothesis, frame_index, total_frames)
            draw_contact_events(frame, events, frame_index)
            draw_timeline(frame, hypotheses, frame_index, total_frames)
            writer.write(frame)
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
    if mp4_output:
        try:
            transcode_to_mp4(writer_path, mp4_output)
        finally:
            if os.path.exists(writer_path):
                os.remove(writer_path)
    print(f"wrote {args.output} ({frame_index} frames)")


if __name__ == "__main__":
    main()
