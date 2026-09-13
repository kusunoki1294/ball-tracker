"""Validate known bounce-detector limits stay explicit.

This does not ask the detector to get better. It pins the current explanation
for a documented miss so a future change cannot quietly turn the diagnosis into
stale prose.
"""

import json
import os
import sys

from bounce_detect import candidate_rank_key, detect_bounces
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
    # These two fire when the f1446 bug is FIXED, which is the outcome we want.
    # The message has to make that unmistakable: arriving as a plain red failure,
    # the cheapest way out is reverting the fix.
    if kept_bounce is not None:
        errors.append(
            "EXPECTED TO FAIL ONCE f1446 IS FIXED: tennis9 f1445 now survives NMS. "
            "This check pins a KNOWN BUG, not desired behaviour. If you fixed it, "
            "update docs/experiments/tennis9_f1446_suppression.md and delete this check."
        )
    if kept_contact is None:
        errors.append(
            "EXPECTED TO FAIL ONCE f1446 IS FIXED: tennis9 f1456 no longer survives NMS. "
            "This check pins a KNOWN BUG, not desired behaviour. If you fixed it, "
            "update docs/experiments/tennis9_f1446_suppression.md and delete this check."
        )
    elif not kept_contact.get("suppressed"):
        # The exact count was pinned at 3 and is incidental: it counts f1444,
        # f1445 and f1455 clustering nearby, so any unrelated detector change
        # touching one of them broke the assertion with no diagnostic value.
        # What the diagnosis needs is that f1456 suppressed something at all.
        errors.append("tennis9 f1456 suppressed nothing; the f1446 diagnosis no longer holds")

    # The mechanism, not just its current outcome. If f1456 stopped outranking
    # f1445 while f1445 stayed suppressed, the diagnosis would be stale and every
    # check above would still pass.
    if pre_bounce is not None and pre_contact is not None:
        if candidate_rank_key(pre_contact) >= candidate_rank_key(pre_bounce):
            errors.append(
                "tennis9 f1456 no longer outranks f1445 under the detector's own "
                "suppression order; the f1446 diagnosis names the wrong mechanism"
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
