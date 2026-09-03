"""Export tracker association steps worth a human's eye. Audit only.

A ball track sometimes jumps to a different object - a player's body, a ball in
someone's hand - and the tracker's prediction is then poisoned so real
detections are rejected behind it. That destroys the arc a bounce needs and is
the cause of the tennis9 bounce recall misses at f1147 and f1401. The remaining
f1446 miss is different: suppression keeps a nearby racket contact over the
ground bounce, so this audit is not expected to explain every miss.

This lists the steps where that MIGHT have happened. It deliberately does not
decide. Errors and recoveries are the same jump in opposite directions - at
tennis9 f1839 the flagged step is the tracker climbing back off the near
player's body onto the real ball - so any automatic verdict here would be wrong
about half the time. See docs/experiments/tennis9_association_labelled_set.md.

Reads tracking JSONL only. It does not import or influence track_ball_yolo.
"""

import argparse
import csv
import html
import json
import math
import os


DEFAULT_PREDICTION_ERROR_PX = 60.0
SIZE_RATIO_LARGE = 1.6
DEAD_TRACK_REASONS = {"far_jump_rejected", "no_candidates"}
DEAD_TRACK_MIN = 3
LOOKAHEAD = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", required=True,
                        help="label=path/to/tracking.jsonl (repeatable)")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-html")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-prediction-error", type=float,
                        default=DEFAULT_PREDICTION_ERROR_PX,
                        help="only report steps further than this from the tracker's "
                             "own prediction (default %(default)s px)")
    return parser.parse_args()


def ball_size(ball):
    bbox = (ball or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def size_class(before, after):
    """Which way the apparent size jumped.

    Named for the observation, not a verdict: moving onto a much larger blob
    often means the track has landed on a near-camera object, and moving off one
    often means it has climbed back onto the ball - but both also occur in
    ordinary fast motion, so neither is called an error here.
    """
    if before is None or after is None:
        return "unknown"
    if after >= before * SIZE_RATIO_LARGE:
        return "onto_larger_blob"
    if before >= after * SIZE_RATIO_LARGE:
        return "off_larger_blob"
    return "similar_size"


def track_outcome(rows, frame):
    reasons = []
    for offset in range(1, LOOKAHEAD + 1):
        selector = ((rows.get(frame + offset) or {}).get("ball_debug") or {}).get("selector") or {}
        reason = selector.get("reason")
        if reason:
            reasons.append(reason)
    dead = sum(1 for reason in reasons if reason in DEAD_TRACK_REASONS)
    return ("track_dies" if dead >= DEAD_TRACK_MIN else "track_continues"), reasons


def collect(label, path, fps, min_error):
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[int(record["frame"])] = record
    found = []
    for frame in sorted(rows):
        record = rows[frame]
        selector = (record.get("ball_debug") or {}).get("selector") or {}
        if selector.get("decision") != "selected":
            continue
        # Prediction distance only means anything while tracking is continuous.
        # After a gap the prediction is stale and routinely hundreds of px out,
        # so including those would bury the real candidates in noise.
        if (selector.get("missed_frames_before") or 0) != 0:
            continue
        predicted = selector.get("predicted_center")
        chosen = (selector.get("selected_candidate") or {}).get("center")
        if not predicted or not chosen:
            continue
        error = math.hypot(chosen[0] - predicted[0], chosen[1] - predicted[1])
        if error <= min_error:
            continue
        previous = (rows.get(frame - 1) or {}).get("ball")
        current = record.get("ball")
        jump = None
        if previous and current:
            jump = math.hypot(current["center"][0] - previous["center"][0],
                              current["center"][1] - previous["center"][1])
        before, after = ball_size(previous), ball_size(current)
        outcome, reasons = track_outcome(rows, frame)
        found.append({
            "clip": label,
            "frame": frame,
            "seconds": round(frame / fps, 2),
            "prediction_error_px": round(error, 1),
            "jump_px": None if jump is None else round(jump, 1),
            "size_before_px": before,
            "size_after_px": after,
            "size_ratio": None if not (before and after) else round(max(before, after) / max(1.0, min(before, after)), 2),
            "size_class": size_class(before, after),
            "track_outcome": outcome,
            "selector_reason": selector.get("reason"),
            "following_reasons": "|".join(reasons),
            "review_verdict": "",
        })
    return found


FIELDS = ["clip", "frame", "seconds", "prediction_error_px", "jump_px",
          "size_before_px", "size_after_px", "size_ratio", "size_class",
          "track_outcome", "selector_reason", "following_reasons", "review_verdict"]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def write_html(rows, path, min_error):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(row[field])}</td>" for field in FIELDS if field != "following_reasons") + "</tr>"
        for row in rows
    )
    document = f"""<title>Tracker Association Audit</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 24px; background: #f5f7f9; color: #16202b; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
 th, td {{ border: 1px solid #d5dce4; padding: 5px 8px; text-align: left; }}
 th {{ background: #eef2f6; }}
 .caveat {{ background: #fff5e6; border: 1px solid #f0c987; padding: 12px 14px;
            border-radius: 6px; margin-bottom: 18px; }}
</style>
<h1>Tracker Association Audit</h1>
<div class="caveat">
  <strong>Candidates for review, not findings.</strong> These are tracker steps that
  landed more than {min_error:.0f} px from the tracker's own prediction while tracking
  continuously. Some are the track jumping onto the wrong object; some are the track
  climbing back off a wrong object onto the ball; most are ordinary fast motion.
  <strong>This report cannot tell those apart</strong> — an error and a recovery are the
  same jump in opposite directions — so <em>size_class</em> and <em>track_outcome</em>
  describe what was observed and carry no verdict. Fill in <em>review_verdict</em> by
  looking at the frames.
</div>
<p>{len(rows)} step(s).</p>
<table><thead><tr>{''.join(f'<th>{esc(f)}</th>' for f in FIELDS if f != 'following_reasons')}</tr></thead>
<tbody>{body}</tbody></table>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def main():
    args = parse_args()
    rows = []
    for spec in args.clip:
        if "=" not in spec:
            raise SystemExit(f"--clip needs label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        rows.extend(collect(label, path, args.fps, args.min_prediction_error))
    write_csv(rows, args.output_csv)
    print(f"wrote {args.output_csv} ({len(rows)} steps)")
    if args.output_html:
        write_html(rows, args.output_html, args.min_prediction_error)
        print(f"wrote {args.output_html}")
    print("audit only: no tracker selection was changed and no step is labelled an error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
