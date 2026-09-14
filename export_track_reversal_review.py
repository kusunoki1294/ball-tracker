"""Export detector-independent ball-track reversals for human review.

This is an audit tool, not a bounce detector. A vertical reversal can be a
ground bounce, racket contact, toss, dead-ball handling, or track error. The
output therefore uses ``review_required`` status and never feeds scoring or
the normal bounce list.
"""

import argparse
import csv
import json

import numpy as np


def load_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observed_samples(rows):
    samples = []
    for row in rows:
        ball = row.get("ball") or {}
        center = ball.get("center")
        if not center or ball.get("interpolated") or ball.get("motion_gate") == "coast":
            continue
        samples.append((int(row["frame"]), float(center[0]), float(center[1])))
    return samples


def fit_velocity(samples, frame, before):
    points = [p for p in samples if (p[0] < frame if before else p[0] > frame)
              and abs(p[0] - frame) <= 10]
    if len(points) < 4:
        return None
    times = np.array([p[0] - frame for p in points], dtype=float)
    ys = np.array([p[2] for p in points], dtype=float)
    # Four samples support a quadratic while leaving enough samples to expose
    # a bad fit through the residual rather than silently accepting a slope.
    coeff = np.polyfit(times, ys, 2)
    predicted = np.polyval(coeff, times)
    residual = float(np.sqrt(np.mean((ys - predicted) ** 2)))
    return float(coeff[-2]), residual, len(points)


def reversals(rows, candidate_frames=(), cluster_frames=6,
              max_fit_residual_px=8.0):
    samples = observed_samples(rows)
    candidates = sorted(int(frame) for frame in candidate_frames)
    raw = []
    for frame, center_x, center_y in samples:
        before = fit_velocity(samples, frame, True)
        after = fit_velocity(samples, frame, False)
        if not before or not after:
            continue
        vy_before, residual_before, count_before = before
        vy_after, residual_after, count_after = after
        delta = vy_before - vy_after
        if (vy_before <= 0 or vy_after > 0 or delta < 8
                or max(residual_before, residual_after) > max_fit_residual_px):
            continue
        raw.append({
            "frame": frame,
            "image_x": round(center_x, 1),
            "image_y": round(center_y, 1),
            "vy_before_px_frame": round(vy_before, 3),
            "vy_after_px_frame": round(vy_after, 3),
            "direction_change_px_frame": round(delta, 3),
            "fit_residual_px": round(max(residual_before, residual_after), 3),
            "tracked_samples_before": count_before,
            "tracked_samples_after": count_after,
        })

    # Several adjacent observed frames can describe one reversal. Keep the
    # strongest member and mark whether the existing detector already has a
    # nearby candidate; this is a review index, never a new event count.
    grouped = []
    for item in raw:
        if grouped and item["frame"] - grouped[-1]["frame"] <= cluster_frames:
            if item["direction_change_px_frame"] > grouped[-1]["direction_change_px_frame"]:
                grouped[-1] = item
        else:
            grouped.append(item)
    for item in grouped:
        nearest = min((abs(item["frame"] - frame) for frame in candidates), default=None)
        item["near_existing_candidate"] = nearest is not None and nearest <= 4
        item["nearest_candidate_delta_frames"] = nearest if nearest is not None and nearest <= 4 else ""
        # Review ordering only: this measures how inspectable the reversal is
        # from the observed track, not whether it is a bounce.
        item["review_evidence_score"] = round(
            item["direction_change_px_frame"]
            / max(1.0, item["fit_residual_px"])
            * min(item["tracked_samples_before"], item["tracked_samples_after"]) / 10.0,
            3,
        )
        item["status"] = "review_required"
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--bounce-json", required=True,
                        help="Existing detector JSON containing a bounces list.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    with open(args.bounce_json) as handle:
        bounce_data = json.load(handle)
    existing = [int(item["frame"]) for item in bounce_data.get("bounces", [])]
    output = reversals(rows, existing)
    fields = [
        "frame", "image_x", "image_y", "near_existing_candidate",
        "nearest_candidate_delta_frames",
        "direction_change_px_frame", "vy_before_px_frame", "vy_after_px_frame",
        "fit_residual_px", "tracked_samples_before", "tracked_samples_after",
        "review_evidence_score", "status",
    ]
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    missing = sum(not item["near_existing_candidate"] for item in output)
    if args.summary_json:
        summary = {
            "source_jsonl": args.jsonl,
            "source_bounce_json": args.bounce_json,
            "review_event_count": len(output),
            "candidate_free_event_count": missing,
            "not_bounce_truth": True,
            "status": "review_required",
            "method": "observed_ball_track_vertical_reversal_with_quadratic_fit",
            "notes": [
                "Reversals may be bounces, racket contacts, dead-ball handling, or tracking errors.",
                "Candidate-free events require source-video review before any detector change.",
            ],
        }
        with open(args.summary_json, "w") as handle:
            json.dump(summary, handle, indent=2)
    print(f"wrote {args.output_csv}: {len(output)} reversal reviews, "
          f"{missing} without a nearby detector candidate")


if __name__ == "__main__":
    main()
