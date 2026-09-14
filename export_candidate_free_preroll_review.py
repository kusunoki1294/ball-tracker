"""Export image-space trajectory cards for candidate-free bounce events."""

import argparse
import csv

from export_timeline_preroll_review import export_review


def read_events(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "near_existing_candidate", "status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"review CSV missing fields: {sorted(missing)}")
        events = []
        for row in reader:
            if row["near_existing_candidate"] == "True":
                continue
            frame = int(row["frame"])
            events.append({
                "frame": frame,
                "kind": "candidate-free reversal",
                "note": (
                    f"direction change {row.get('direction_change_px_frame', '?')} px/frame; "
                    f"fit residual {row.get('fit_residual_px', '?')} px; "
                    "review only, not a bounce label"
                ),
                "trail_start_frame": max(1, frame - 60),
            })
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    events = read_events(args.reviews)
    if not events:
        raise SystemExit("review CSV contains no candidate-free events")
    export_review([{
        "label": "tennis11 candidate-free reversals",
        "video": args.video,
        "jsonl": args.jsonl,
        "items": events,
    }], args.output)


if __name__ == "__main__":
    main()
