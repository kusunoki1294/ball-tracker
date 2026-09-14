"""Validate candidate-free pre-roll review input filtering."""

import csv
import tempfile

from export_candidate_free_preroll_review import read_events
from export_timeline_preroll_review import tracked_points


def main():
    with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "frame", "near_existing_candidate", "status",
            "direction_change_px_frame", "fit_residual_px", "review_evidence_score",
        ])
        writer.writeheader()
        writer.writerow({
            "frame": "10", "near_existing_candidate": "False", "status": "review_required",
            "direction_change_px_frame": "12", "fit_residual_px": "2", "review_evidence_score": "4.2",
        })
        writer.writerow({
            "frame": "20", "near_existing_candidate": "True", "status": "review_required",
            "direction_change_px_frame": "12", "fit_residual_px": "2", "review_evidence_score": "4.2",
        })
        handle.flush()
        events = read_events(handle.name)
    if len(events) != 1 or events[0]["frame"] != 10:
        print("pre-roll review must include only candidate-free events")
        return 1
    if "review only" not in events[0]["note"] or "bounce likelihood" not in events[0]["note"]:
        print("pre-roll note must preserve review-only semantics")
        return 1
    if events[0]["review_evidence_score"] <= 0:
        print("pre-roll review must expose a positive evidence-only rank")
        return 1
    rows = {
        1: {"ball": {"center": [10, 10], "bbox": [8, 8, 12, 12]}},
        2: {"ball": {"center": [20, 20], "interpolated": True}},
        3: {"ball": {"center": [30, 30], "motion_gate": "coast"}},
    }
    observed = tracked_points(rows, 1, 3)
    if [item[0] for item in observed] != [1]:
        print("pre-roll trails must exclude interpolated and coasted positions")
        return 1
    print("candidate-free pre-roll validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
