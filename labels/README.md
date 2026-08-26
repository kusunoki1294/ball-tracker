# labels/

Hand-labelled ground truth. Small, and the only ground truth this project has.

## tennis11_game1_bounce_labels.csv

Every `bounce_detect.py` detection on tennis11 game 1 (44), labelled from
5-frame video crops centred on each detection's own recorded pixel, taken from
the clean slice `yoloVids/inputs/tennis11_game1.mp4` so the tracker's own
overlay could not influence the call.

Labels: `live_bounce` (real, in play), `dead_bounce` (real bounce, ball not in
play - pre-serve dribbles, post-point), `racket` (player contact: strike, catch,
toss, or ball carried in hand), `tracking_artifact` (no ball visible at the
marker), `ambiguous` (deliberately not forced - see `note`).

What it establishes, as of 2026-08:

- Only 3 of 44 are artifacts. The detector rarely fires on nothing; it
  over-reports REAL ball events that are not rally bounces. That makes this a
  classification problem, not a detection one.
- `rally_scoring_eligible` contains zero racket contacts and zero dead balls,
  which is the property rally scoring depends on. Live precision 66.7% inside
  it, 12.5% outside.
- `near_player` separates almost all of it: 69.2% live precision when False,
  9.7% when True.
- `shape_confidence` does NOT predict event type. It grades trajectory quality,
  not what the event is; do not use it as a proxy for correctness.
- `court_margin_ft=3.0` is optimal against these labels. Tightening to 0 loses
  two real out-of-court bounces including a serve fault landing; loosening to 10
  keeps fewer real bounces (NMS suppression) and adds unlabelled detections.

Regenerate the detections with `eval_bounce_detect.py --review-csv`; the `label`
and `note` columns are hand-added and are not reproducible from code.
