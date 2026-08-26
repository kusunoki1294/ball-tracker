"""Score bounce_detect.py against what we can actually check.

There is no exhaustive hand-labelled bounce set for either clip, so this uses
the two references that exist, and is explicit about what each one proves.

* tennis9's 23 validated bounces (the reviewed known-good output of the
  in-tracker detector) measure RECALL only. They are not a complete ground
  truth, so a detection outside that set is reported as "extra", never as a
  false positive: several verified extras are real ball bounces the old detector
  missed, mostly players dribbling the ball before serving.
* tennis11's serve contact frames, read off the video, measure whether the
  highest-value bounces are found. Two metrics are reported. The LOOSE one only
  asks for a bounce 5-40 frames after contact. The STRICT one additionally
  requires it on the receiver's side of the net and flagged
  `serve_landing_precondition`, because a loose hit can be satisfied by an
  unrelated low-confidence event. It deliberately does NOT require
  `rally_scoring_eligible`: that contract excludes bounces near a player, and a
  serve landing beside the waiting receiver is exactly the case it would reject.
  STRICT is a recall check, not a precision estimate -- the precondition is near
  universal on tennis11 (41/41), so passing it proves little on its own.

Precision is deliberately NOT claimed. `--review-csv` exports the tennis9
detections that fall outside the validated set, plus EVERY tennis11 detection
(that clip has no truth set to be outside of), for hand labelling. Labelling
that file is what would let precision be scored honestly.
"""
import argparse
import csv
import json
import sys

import numpy as np

from bounce_detect import detect_bounces

TENNIS9 = {
    "jsonl": "yoloVids/outputs/tennis9/play_segments/ai9.3.jsonl",
    "calib": "yoloVids/calibration/court_calib_tennis7.json",
    "analysis": "yoloVids/outputs/tennis9/play_segments/ai9.5.analysis.json",
    "fps": 30.0,
}
TENNIS11 = {
    "jsonl": "yoloVids/outputs/tennis11/ai11.1.jsonl",
    "calib": "yoloVids/calibration/court_calib_tennis11.json",
    "analysis": "yoloVids/outputs/tennis11/ai11.2.analysis.json",
    "fps": 30.0,
}
# (point, serve contact frame, receiver side). Contact frames come from
# serve_detect.py's verified set, which superseded an earlier eyeballed list --
# P3's two serves and P4/P5 moved by up to 12 frames. In tennis11 game 1 the near
# player serves throughout, so every serve must land far.
TENNIS11_SERVES = [(1, 160, "far"), (2, 635, "far"), (3, 1485, "far"), (3, 1659, "far"),
                   (4, 2091, "far"), (5, 2432, "far"), (6, 2952, "far")]
TENNIS11_POINTS = [(1, 445), (446, 1353), (1354, 1796), (1797, 2367), (2368, 2632), (2633, 3210)]
MATCH_TOLERANCE = 4
SERVE_WINDOW = (5, 40)
# A serve crossing ~60ft cannot land sooner than this, even at high pace;
# confirmed landings on tennis11 sit at +17 to +26. Anything detected between the
# strike and this floor is the racket contact itself, not a landing. The serve
# metric's window starts at +5 and so cannot see them, which is exactly how a
# +1-frame contact reached Nadal's anchoring path unnoticed -- report them
# separately rather than relying on someone thinking to ask.
SERVE_MIN_FLIGHT_FRAMES = 10


def load_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(config):
    with open(config["calib"]) as handle:
        calib = json.load(handle)["points"]
    return detect_bounces(load_rows(config["jsonl"]), calib)


def recall_report(bounces, truth):
    frames = [b["frame"] for b in bounces]
    used, matched, missed = set(), [], []
    for want in truth:
        hit = next((f for f in frames if abs(f - want) <= MATCH_TOLERANCE and f not in used), None)
        if hit is None:
            missed.append(want)
        else:
            used.add(hit)
            matched.append(hit)
    extra = [b for b in bounces if b["frame"] not in used]
    return matched, missed, extra


def distributions(bounces, label):
    """Djokovic's ask: see the normalised numbers per clip before trusting them."""
    if not bounces:
        return
    for key in ("dvy_ft", "residual_px", "join_error_px", "join_error_frames"):
        values = np.array([b[key] for b in bounces if b.get(key) is not None], dtype=float)
        if not len(values):
            continue
        print(f"         {label} {key:>14}: p10={np.percentile(values,10):>7.2f} "
              f"p50={np.percentile(values,50):>7.2f} p90={np.percentile(values,90):>7.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--review-csv", help="Export unmatched detections for hand labelling.")
    args = parser.parse_args()

    bounces9 = run(TENNIS9)
    with open(TENNIS9["analysis"]) as handle:
        truth = [b["frame"] for b in json.load(handle)["bounces"]]
    matched, missed, extra = recall_report(bounces9, truth)
    print(f"tennis9  recall {len(matched)}/{len(truth)}  detected {len(bounces9)}  "
          f"extra {len(extra)} (unlabelled, not counted as errors)")
    if missed:
        print(f"         missed frames: {missed}")
    distributions(bounces9, "t9")

    bounces11 = run(TENNIS11)
    with open(TENNIS11["analysis"]) as handle:
        previous = len(json.load(handle)["bounces"])
    loose = strict = 0
    for point, contact, receiver in TENNIS11_SERVES:
        lo, hi = SERVE_WINDOW
        window = [b for b in bounces11 if lo <= b["frame"] - contact <= hi]
        # A serve landing is expected to be near the waiting receiver, so this
        # asks for sound trajectory evidence on the receiver's side rather than
        # rally-scoring eligibility, which excludes anything near a player.
        good = next((b for b in window
                     if b["side"] == receiver and b["serve_landing_precondition"]), None)
        # Report the strict match when there is one: printing the first loose
        # candidate would misdescribe which detection satisfied the metric.
        hit = good or (window[0] if window else None)
        loose += 1 if window else 0
        strict += 1 if good else 0
        if args.verbose:
            if hit:
                detail = (f"f{hit['frame']} (+{hit['frame'] - contact}) side={hit['side']} "
                          f"shape={hit['shape_confidence']}"
                          f"{' near_player' if hit['near_player'] else ''} "
                          f"world=({hit['world_point'][0]:.1f},{hit['world_point'][1]:.1f}) "
                          f"{hit['provenance']}")
            else:
                detail = "none"
            print(f"         pt{point} serve f{contact}: {detail}"
                  f"  [{len(window)} in window]{'' if good else '   [fails strict]'}")
    print(f"tennis11 bounces {previous} -> {len(bounces11)}   "
          f"serve bounces loose {loose}/{len(TENNIS11_SERVES)}  strict {strict}/{len(TENNIS11_SERVES)}")

    # Anchoring hazard: a consumer that takes the FIRST bounce after the strike
    # will take a contact-adjacent detection instead of the real landing.
    hazards = []
    for point, contact, _ in TENNIS11_SERVES:
        early = [b for b in bounces11
                 if 1 <= b["frame"] - contact < SERVE_MIN_FLIGHT_FRAMES]
        for bounce in early:
            hazards.append((point, contact, bounce))
    if hazards:
        print(f"         ANCHORING HAZARD: {len(hazards)} detection(s) within "
              f"{SERVE_MIN_FLIGHT_FRAMES} frames of a serve strike -- too soon to be a "
              f"landing, so a first-bounce-after-strike consumer would take these:")
        for point, contact, bounce in hazards:
            print(f"           pt{point} strike f{contact}: f{bounce['frame']} "
                  f"(+{bounce['frame'] - contact}) world=("
                  f"{bounce['world_point'][0]:.1f},{bounce['world_point'][1]:.1f}) "
                  f"near_player={bounce['near_player']}")
    else:
        print(f"         anchoring hazard check: none within "
              f"{SERVE_MIN_FLIGHT_FRAMES} frames of a strike")
    print("         note: strict adds only receiver-side + shape-not-low, and "
          "shape-not-low is near-universal here, so strict is a recall check, "
          "NOT a precision estimate. Precision needs the review CSV labelled.")
    split = {}
    for bounce in bounces11:
        key = bounce["confidence"] + ("/near_player" if bounce["near_player"] else "")
        split[key] = split.get(key, 0) + 1
    eligible = sum(1 for b in bounces11 if b["rally_scoring_eligible"])
    serve_ok = sum(1 for b in bounces11 if b["serve_landing_precondition"])
    print(f"         confidence split: {split}")
    print(f"         rally-scoring-eligible {eligible}/{len(bounces11)}, "
          f"serve-landing precondition {serve_ok}/{len(bounces11)}")
    distributions(bounces11, "t11")

    if args.review_csv:
        with open(args.review_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["clip", "frame", "seconds", "side", "world_x", "world_y",
                             "shape_confidence", "confidence", "near_player",
                             "rally_scoring_eligible", "serve_landing_precondition",
                             "provenance", "dvy_ft", "residual_px", "join_error_px",
                             "join_error_frames", "suppressed", "label"])
            for clip, items, fps in (("tennis9", extra, TENNIS9["fps"]),
                                     ("tennis11", bounces11, TENNIS11["fps"])):
                for b in items:
                    writer.writerow([clip, b["frame"], round(b["frame"] / fps, 2), b["side"],
                                     b["world_point"][0], b["world_point"][1],
                                     b["shape_confidence"], b["confidence"], b["near_player"],
                                     b["rally_scoring_eligible"], b["serve_landing_precondition"],
                                     b["provenance"], b["dvy_ft"], b["residual_px"],
                                     b["join_error_px"], b["join_error_frames"],
                                     b.get("suppressed", 0), ""])
        print(f"wrote {args.review_csv} "
              f"(label column: live_bounce | dead_bounce | racket | tracking_artifact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
