"""Validate candidate-free pre-roll review input filtering."""

import csv
import tempfile

from export_candidate_free_preroll_review import read_events


def main():
    with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "frame", "near_existing_candidate", "status",
            "direction_change_px_frame", "fit_residual_px",
        ])
        writer.writeheader()
        writer.writerow({
            "frame": "10", "near_existing_candidate": "False", "status": "review_required",
            "direction_change_px_frame": "12", "fit_residual_px": "2",
        })
        writer.writerow({
            "frame": "20", "near_existing_candidate": "True", "status": "review_required",
            "direction_change_px_frame": "12", "fit_residual_px": "2",
        })
        handle.flush()
        events = read_events(handle.name)
    if len(events) != 1 or events[0]["frame"] != 10:
        print("pre-roll review must include only candidate-free events")
        return 1
    if "review only" not in events[0]["note"]:
        print("pre-roll note must preserve review-only semantics")
        return 1
    print("candidate-free pre-roll validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
