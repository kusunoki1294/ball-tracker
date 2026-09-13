"""Evaluate bounce recall after candidate-free reversal events are reviewed.

The ordinary bounce labels describe detector proposals and therefore cannot
measure recall. This tool joins those labels with a separately reviewed
candidate-free inventory. Candidate-free labels are never consumed by the
analyzer; this module is evaluation-only.
"""

import argparse
import csv
import json
import sys


VALID_LABELS = {"live_bounce", "dead_bounce", "racket", "tracking_artifact", "ambiguous"}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_label(row, path, index):
    label = (row.get("label") or "").strip()
    if label not in VALID_LABELS:
        raise ValueError(
            f"{path} row {index}: label must be one of {sorted(VALID_LABELS)}, got {label!r}"
        )
    return label


def evaluate(detector_rows, candidate_free_rows):
    detector_labels = [require_label(row, "detector labels", i)
                       for i, row in enumerate(detector_rows, start=2)]
    candidate_labels = [require_label(row, "candidate-free labels", i)
                        for i, row in enumerate(candidate_free_rows, start=2)]
    detected_live = sum(label == "live_bounce" for label in detector_labels)
    candidate_free_live = sum(label == "live_bounce" for label in candidate_labels)
    known_live = detected_live + candidate_free_live
    return {
        "detector_proposals": len(detector_labels),
        "candidate_free_events": len(candidate_labels),
        "detected_live_bounces": detected_live,
        "candidate_free_live_bounces": candidate_free_live,
        "reviewed_live_bounces": known_live,
        "recall_before_retracking": detected_live / known_live if known_live else None,
        "detector_live_precision": detected_live / len(detector_labels) if detector_labels else None,
        "not_scoring_truth": True,
        "status": "reviewed_candidate_free_inventory",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-labels", required=True,
                        help="Existing labels for detector-proposed candidates.")
    parser.add_argument("--candidate-free-labels", required=True,
                        help="Separately reviewed candidate-free reversal CSV.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        summary = evaluate(read_rows(args.detector_labels),
                           read_rows(args.candidate_free_labels))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"bounce recall evaluation failed: {exc}", file=sys.stderr)
        return 1
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
