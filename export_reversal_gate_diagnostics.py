"""Report tracker context near candidate-free reversals.

This is an audit report only. It does not identify the causal rejection gate,
reinterpret a reversal as a bounce, or modify the tracking log.
"""

import argparse
import csv
import json


def load_reviews(path):
    with open(path, newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row.get("near_existing_candidate") != "True"]


def context_reason(row):
    debug = row.get("ball_debug") or {}
    moving = debug.get("moving_filter") or {}
    selector = debug.get("selector") or {}
    reason = selector.get("reason") or selector.get("decision") or "unknown"
    if moving.get("rejected_low_excursion"):
        return "moving_filter_low_excursion"
    if moving.get("rejected_short_history"):
        return "moving_filter_short_history"
    rejected = selector.get("rejected_candidates") or []
    if rejected:
        return "selector_" + str(rejected[0].get("rejection", {}).get("reason", reason))
    return str(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    reviews = load_reviews(args.reviews)
    rows = {}
    with open(args.jsonl) as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[int(item["frame"])] = item

    output = []
    for review in reviews:
        frame = int(review["frame"])
        item = rows.get(frame, {})
        debug = item.get("ball_debug") or {}
        selector = debug.get("selector") or {}
        output.append({
            "frame": frame,
            "image_x": float(review["image_x"]),
            "image_y": float(review["image_y"]),
            "direction_change_px_frame": float(review["direction_change_px_frame"]),
            "fit_residual_px": float(review["fit_residual_px"]),
            "ball_present_at_reversal": bool(item.get("ball")),
            "tracker_context_reason": context_reason(item),
            "selector_decision": selector.get("decision"),
            "moving_filter": debug.get("moving_filter", {}),
            "candidate_counts": debug.get("counts", {}),
            "not_bounce_truth": True,
        })
    summary = {
        "review_event_count": len(output),
        "candidate_free_event_count": len(output),
        "not_bounce_truth": True,
        "status": "review_required",
        "note": "Context at a reversal is not proof of which bounce-detector gate rejected it.",
        "events": output,
    }
    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {args.output_json}: {len(output)} candidate-free diagnostics")


if __name__ == "__main__":
    main()
