"""Evaluate whether existing racket boxes help serve-contact review.

Experiment only. This reads tracked JSONL plus timeline hypotheses and reports
whether YOLO `tennis racket` detections add useful evidence around accepted and
suppressed serve-motion contacts. It does not change analyzer decisions.
"""

import argparse
import csv
import html
import json
import os

OFFSETS = (-4, -2, 0, 2)
PLAYER_KEY = {"near": "player_near", "far": "player_far"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="LABEL=JSONL=HYPOTHESES",
        help="Clip inputs. Repeat as label=jsonl=hypotheses.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument(
        "--verified-contacts",
        action="append",
        default=[],
        metavar="LABEL=F1,F2,...",
        help="Optional verified contact frames for a clip label.",
    )
    return parser.parse_args()


def parse_clip(raw):
    parts = raw.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--clip expects LABEL=JSONL=HYPOTHESES, got {raw!r}")
    return parts[0], parts[1], parts[2]


def parse_verified(entries):
    result = {}
    for raw in entries:
        label, _, values = raw.partition("=")
        if not label or not values:
            raise ValueError(f"--verified-contacts expects LABEL=F1,F2,..., got {raw!r}")
        result[label] = {int(value) for value in values.split(",") if value.strip()}
    return result


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tracking_by_frame(path):
    by_frame = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_frame[int(row["frame"])] = row
    return by_frame


def distance_to_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    dx = max(float(x1) - x, 0.0, x - float(x2))
    dy = max(float(y1) - y, 0.0, y - float(y2))
    return (dx * dx + dy * dy) ** 0.5


def center(bbox):
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def box_height(bbox):
    return max(1.0, float(bbox[3]) - float(bbox[1]))


def racket_detections(row):
    return [
        detection
        for detection in (row.get("scene") or [])
        if detection.get("class_name") == "tennis racket" and detection.get("bbox")
    ]


def all_motions(hypotheses):
    rows = []
    for hypothesis in hypotheses:
        display_id = hypothesis.get("display_id") or hypothesis.get("id")
        for attempt in hypothesis.get("attempts") or []:
            rows.append((display_id, attempt, "accepted"))
        for attempt in hypothesis.get("suppressed_rally_motions") or []:
            rows.append((display_id, attempt, "suppressed"))
    return sorted(rows, key=lambda item: int(item[1]["contact_frame"]))


def cue_metrics(by_frame, attempt):
    contact_frame = int(attempt["contact_frame"])
    server = attempt.get("server")
    min_ball_to_racket = None
    min_server_to_racket = None
    best_racket_conf = None
    best_racket_above_top_frac = None
    frames_with_racket = 0
    frames_with_ball = 0
    frames_with_server = 0
    for offset in OFFSETS:
        row = by_frame.get(contact_frame + offset)
        if not row:
            continue
        server_box = (row.get(PLAYER_KEY.get(server, "")) or {}).get("bbox")
        ball_center = ((row.get("ball") or {}).get("center"))
        rackets = racket_detections(row)
        if server_box:
            frames_with_server += 1
        if ball_center:
            frames_with_ball += 1
        if rackets:
            frames_with_racket += 1
        for racket in rackets:
            bbox = racket["bbox"]
            racket_center = center(bbox)
            ball_distance = distance_to_bbox((float(ball_center[0]), float(ball_center[1])), bbox) if ball_center else None
            server_distance = distance_to_bbox(racket_center, server_box) if server_box else None
            if ball_distance is not None and (
                min_ball_to_racket is None or ball_distance < min_ball_to_racket
            ):
                min_ball_to_racket = ball_distance
                best_racket_conf = racket.get("conf")
            if server_distance is not None and (
                min_server_to_racket is None or server_distance < min_server_to_racket
            ):
                min_server_to_racket = server_distance
                best_racket_conf = racket.get("conf")
                if server_box:
                    _, top, _, _ = [float(value) for value in server_box]
                    best_racket_above_top_frac = (top - float(bbox[1])) / box_height(server_box)
    return {
        "frames_with_racket_in_strip": frames_with_racket,
        "frames_with_ball_in_strip": frames_with_ball,
        "frames_with_server_in_strip": frames_with_server,
        "min_ball_to_racket_px": round(min_ball_to_racket, 1) if min_ball_to_racket is not None else None,
        "min_server_to_racket_px": round(min_server_to_racket, 1) if min_server_to_racket is not None else None,
        "best_racket_conf": round(float(best_racket_conf), 3) if best_racket_conf is not None else None,
        "racket_above_server_top_frac": round(best_racket_above_top_frac, 3)
        if best_racket_above_top_frac is not None
        else None,
    }


def build_rows(clips, verified):
    output = []
    for label, jsonl_path, hypotheses_path in clips:
        by_frame = read_tracking_by_frame(jsonl_path)
        data = load_json(hypotheses_path)
        verified_frames = verified.get(label, set())
        for display_id, attempt, kind in all_motions(data.get("hypotheses") or []):
            contact_frame = int(attempt["contact_frame"])
            row = {
                "clip": label,
                "display_id": display_id,
                "kind": kind,
                "contact_frame": contact_frame,
                "verified_contact": contact_frame in verified_frames if verified_frames else "",
                "server": attempt.get("server"),
                "source": attempt.get("source"),
                "confidence": attempt.get("confidence"),
                "landing_reason": (attempt.get("landing") or {}).get("reason"),
                "review_reasons": ", ".join(attempt.get("review_reasons") or []),
            }
            row.update(cue_metrics(by_frame, attempt))
            output.append(row)
    return output


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "clip",
        "display_id",
        "kind",
        "contact_frame",
        "verified_contact",
        "server",
        "source",
        "confidence",
        "landing_reason",
        "frames_with_racket_in_strip",
        "frames_with_ball_in_strip",
        "frames_with_server_in_strip",
        "min_ball_to_racket_px",
        "min_server_to_racket_px",
        "racket_above_server_top_frac",
        "best_racket_conf",
        "review_reasons",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["clip"], row["kind"], row["verified_contact"])
        groups.setdefault(key, []).append(row)
    lines = []
    for key, items in sorted(groups.items()):
        with_racket = sum(1 for item in items if item["frames_with_racket_in_strip"])
        ball_racket = [
            item["min_ball_to_racket_px"]
            for item in items
            if item["min_ball_to_racket_px"] is not None
        ]
        server_racket = [
            item["min_server_to_racket_px"]
            for item in items
            if item["min_server_to_racket_px"] is not None
        ]
        lines.append(
            {
                "clip": key[0],
                "kind": key[1],
                "verified": key[2],
                "n": len(items),
                "with_racket": with_racket,
                "median_ball_to_racket": median(ball_racket),
                "median_server_to_racket": median(server_racket),
            }
        )
    return lines


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 1)


def write_html(path, rows):
    ensure_parent(path)
    summary_rows = []
    for row in summarize(rows):
        summary_rows.append(
            "<tr>"
            f"<td>{esc(row['clip'])}</td><td>{esc(row['kind'])}</td>"
            f"<td>{esc(row['verified'])}</td><td>{esc(row['n'])}</td>"
            f"<td>{esc(row['with_racket'])}</td>"
            f"<td>{esc(row['median_ball_to_racket'])}</td>"
            f"<td>{esc(row['median_server_to_racket'])}</td>"
            "</tr>"
        )
    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr>"
            f"<td>{esc(row['clip'])}</td><td>{esc(row['display_id'])}</td>"
            f"<td>{esc(row['kind'])}</td><td>f{esc(row['contact_frame'])}</td>"
            f"<td>{esc(row['verified_contact'])}</td><td>{esc(row['server'])}</td>"
            f"<td>{esc(row['source'])}</td><td>{esc(row['confidence'])}</td>"
            f"<td>{esc(row['landing_reason'])}</td>"
            f"<td>{esc(row['frames_with_racket_in_strip'])}</td>"
            f"<td>{esc(row['min_ball_to_racket_px'])}</td>"
            f"<td>{esc(row['min_server_to_racket_px'])}</td>"
            f"<td>{esc(row['racket_above_server_top_frac'])}</td>"
            f"<td>{esc(row['review_reasons'])}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Serve Racket Cue Evaluation</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:28px;background:#f6f7f9;color:#18202a}}
h1{{margin:0 0 8px}}p{{max-width:960px;color:#52606d}}table{{border-collapse:collapse;width:100%;background:#fff;margin:18px 0;border:1px solid #d8dee6}}
th,td{{border-bottom:1px solid #e5e9ef;padding:8px 10px;text-align:left;vertical-align:top;font-size:13px}}
th{{background:#eef2f6;font-size:12px;text-transform:uppercase;color:#3c4752}}
</style></head><body>
<h1>Serve Racket Cue Evaluation</h1>
<p>Experiment only. This measures existing YOLO racket-box evidence around accepted and
suppressed serve-motion contacts. It does not change scoring or serve decisions.</p>
<h2>Summary</h2>
<table><thead><tr><th>Clip</th><th>Kind</th><th>Verified</th><th>N</th><th>With Racket</th><th>Median Ball-Racket px</th><th>Median Server-Racket px</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>
<h2>Details</h2>
<table><thead><tr><th>Clip</th><th>Hypothesis</th><th>Kind</th><th>Frame</th><th>Verified</th><th>Server</th><th>Source</th><th>Confidence</th><th>Landing</th><th>Racket Frames</th><th>Ball-Racket px</th><th>Server-Racket px</th><th>Above Top frac</th><th>Review Reasons</th></tr></thead>
<tbody>{''.join(detail_rows)}</tbody></table>
</body></html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def main():
    args = parse_args()
    clips = [parse_clip(raw) for raw in args.clip]
    rows = build_rows(clips, parse_verified(args.verified_contacts))
    write_csv(args.output_csv, rows)
    write_html(args.output_html, rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_html}")
    for row in summarize(rows):
        print(
            f"{row['clip']} {row['kind']} verified={row['verified']} "
            f"n={row['n']} with_racket={row['with_racket']} "
            f"median_ball_to_racket={row['median_ball_to_racket']} "
            f"median_server_to_racket={row['median_server_to_racket']}"
        )


if __name__ == "__main__":
    main()
