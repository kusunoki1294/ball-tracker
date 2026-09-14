"""Create a blank, review-only label sheet for candidate-free reversals."""

import argparse
import csv


SOURCE_FIELDS = [
    "frame", "image_x", "image_y", "direction_change_px_frame",
    "fit_residual_px", "tracked_samples_before", "tracked_samples_after",
]
FIELDS = [
    "frame", "review_scope", *SOURCE_FIELDS[1:],
    "label", "reviewer_confidence", "note",
]
LABEL_HELP = "live_bounce|dead_bounce|racket|tracking_artifact|ambiguous"


def export_template(input_csv, output_csv):
    with open(input_csv, newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        missing = {field for field in [*SOURCE_FIELDS, "near_existing_candidate"]
                   if field not in (rows.fieldnames or [])}
        if missing:
            raise ValueError(f"review CSV missing fields: {sorted(missing)}")
        with open(output_csv, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            count = 0
            for row in rows:
                if row.get("near_existing_candidate") == "True":
                    continue
                writer.writerow({
                    "frame": row["frame"],
                    "review_scope": "candidate_free_reversal",
                    **{field: row.get(field, "") for field in SOURCE_FIELDS[1:]},
                    "label": "",
                    "reviewer_confidence": "",
                    "note": "",
                })
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Candidate-free reversal review CSV.")
    parser.add_argument("--output", required=True,
                        help=f"Blank label CSV; label values: {LABEL_HELP}.")
    args = parser.parse_args()
    count = export_template(args.input, args.output)
    print(f"wrote {args.output}: {count} review rows; labels remain blank until human review")


if __name__ == "__main__":
    main()
