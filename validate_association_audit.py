"""Check the tracker association audit reports observations, not verdicts.

The audit's value is that it declines to decide. An error and a recovery are the
same jump in opposite directions, so a row that asserted "mis-association" would
be wrong about half the time. These checks pin that contract as much as the
numbers.
"""

import csv
import os
import subprocess
import sys
import tempfile


CLIPS = [
    "tennis9=yoloVids/outputs/tennis9/play_segments/ai9.3.jsonl",
    "tennis11=yoloVids/outputs/tennis11/ai11.1.jsonl",
]

# Hand-checked against the source frames; see
# docs/experiments/tennis9_association_labelled_set.md.
EXPECTED_ROWS = {
    ("tennis9", 1150): ("similar_size", "track_dies"),
    ("tennis9", 1405): ("onto_larger_blob", "track_dies"),
    ("tennis9", 1591): ("off_larger_blob", "track_dies"),
    ("tennis9", 1839): ("off_larger_blob", "track_continues"),
}

VERDICT_WORDS = ("mis-association", "misassociation", "error", "bug", "wrong_object")


def python_bin():
    venv = os.path.join(".venv", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def main():
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "association_audit.csv")
        html_path = os.path.join(tmp, "association_audit.html")
        command = [python_bin(), "export_association_audit.py",
                   "--output-csv", csv_path, "--output-html", html_path]
        for clip in CLIPS:
            command += ["--clip", clip]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            print("association audit failed to run:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            errors.append("association audit produced no rows")
        found = {(row["clip"], int(row["frame"])): row for row in rows}
        for key, (size_class, outcome) in EXPECTED_ROWS.items():
            row = found.get(key)
            if not row:
                errors.append(f"{key}: hand-checked step missing from the audit")
                continue
            if row["size_class"] != size_class:
                errors.append(f"{key}: size_class {row['size_class']!r}, expected {size_class!r}")
            if row["track_outcome"] != outcome:
                errors.append(f"{key}: track_outcome {row['track_outcome']!r}, expected {outcome!r}")
        # The contract: the audit must not decide for the reviewer.
        for row in rows:
            if (row.get("review_verdict") or "").strip():
                errors.append(f"{row['clip']} f{row['frame']}: review_verdict must ship empty")
                break
        for row in rows:
            blob = " ".join(str(v).lower() for k, v in row.items() if k != "following_reasons")
            hit = [w for w in VERDICT_WORDS if w in blob]
            if hit:
                errors.append(f"{row['clip']} f{row['frame']}: row asserts a verdict ({hit[0]})")
                break
        with open(html_path, encoding="utf-8") as handle:
            page = handle.read()
        for phrase in ("Candidates for review, not findings",
                       "cannot tell those apart",
                       "carry no verdict"):
            if phrase not in page:
                errors.append(f"audit HTML missing its caveat: {phrase!r}")
    if errors:
        print("association audit validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"association audit validation passed ({len(rows)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
