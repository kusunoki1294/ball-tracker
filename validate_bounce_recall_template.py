"""Validate the candidate-free review template generator."""

import csv
import tempfile

from export_bounce_recall_template import SOURCE_FIELDS, export_template


def main():
    source_rows = [{
        "frame": "123", "image_x": "10", "image_y": "20",
        "direction_change_px_frame": "12", "fit_residual_px": "1",
        "tracked_samples_before": "4", "tracked_samples_after": "4",
        "near_existing_candidate": "False",
    }, {
        "frame": "456", "image_x": "10", "image_y": "20",
        "direction_change_px_frame": "12", "fit_residual_px": "1",
        "tracked_samples_before": "4", "tracked_samples_after": "4",
        "near_existing_candidate": "True",
    }]
    with tempfile.TemporaryDirectory() as directory:
        source = f"{directory}/source.csv"
        output = f"{directory}/output.csv"
        with open(source, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*SOURCE_FIELDS, "near_existing_candidate"])
            writer.writeheader()
            writer.writerows(source_rows)
        if export_template(source, output) != 1:
            print("template must preserve only candidate-free input rows")
            return 1
        with open(output, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if (rows[0]["frame"] != "123"
                or rows[0]["review_scope"] != "candidate_free_reversal"
                or rows[0]["label"] or rows[0]["reviewer_confidence"]):
            print("template must preserve event data and leave review fields blank")
            return 1
    print("bounce recall template validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
