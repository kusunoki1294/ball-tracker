"""Validate known bounce-detector limits stay explicit.

This does not ask the detector to get better. It pins the current explanation
for a documented miss so a future change cannot quietly turn the diagnosis into
stale prose.
"""

import json
import os
import sys

from bounce_detect import detect_bounces
from eval_bounce_detect import TENNIS9, load_rows


def find_frame(bounces, frame):
    return next((bounce for bounce in bounces if bounce["frame"] == frame), None)


def main():
    errors = []
    with open(TENNIS9["calib"], encoding="utf-8") as handle:
        calib = json.load(handle)["points"]
    rows = load_rows(TENNIS9["jsonl"])

    normal = detect_bounces(rows, calib)
    unsuppressed = detect_bounces(rows, calib, {"suppress": 0})

    # f1446's validated bounce is represented by f1445 in the offline detector:
    # it exists before non-maximum suppression, but the nearby racket contact at
    # f1456 ranks higher and suppresses it. See
    # docs/experiments/tennis9_f1446_suppression.md.
    pre_bounce = find_frame(unsuppressed, 1445)
    pre_contact = find_frame(unsuppressed, 1456)
    kept_bounce = find_frame(normal, 1445)
    kept_contact = find_frame(normal, 1456)

    if pre_bounce is None:
        errors.append("tennis9 f1445 ground-bounce candidate must remain visible before NMS")
    elif pre_bounce.get("shape_confidence") != "high":
        errors.append(
            f"tennis9 f1445 pre-NMS shape {pre_bounce.get('shape_confidence')!r}, expected 'high'"
        )
    if pre_contact is None:
        errors.append("tennis9 f1456 racket-contact candidate must remain visible before NMS")
    elif pre_contact.get("shape_confidence") != "high":
        errors.append(
            f"tennis9 f1456 pre-NMS shape {pre_contact.get('shape_confidence')!r}, expected 'high'"
        )
    if kept_bounce is not None:
        errors.append("tennis9 f1445 is no longer suppressed; update the f1446 diagnosis")
    if kept_contact is None:
        errors.append("tennis9 f1456 no longer survives NMS; update the f1446 diagnosis")
    elif kept_contact.get("suppressed") != 3:
        errors.append(
            f"tennis9 f1456 suppressed count {kept_contact.get('suppressed')!r}, expected 3"
        )

    doc = "docs/experiments/tennis9_f1446_suppression.md"
    if not os.path.exists(doc):
        errors.append(f"missing diagnostic doc {doc}")
    else:
        with open(doc, encoding="utf-8") as handle:
            text = handle.read()
        for phrase in (
            "non-maximum suppression",
            "f1456",
            "f1445",
            "shadow remains a rejected signal",
        ):
            if phrase not in text:
                errors.append(f"{doc} missing diagnostic phrase {phrase!r}")

    if errors:
        print("bounce known-limits validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("bounce known-limits validation passed (f1446 NMS miss guarded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
