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

What it establishes, as of 2026-09-01:

- Label counts after re-checking the post-`aa8f69b` detector output:
  11 `live_bounce`, 14 `racket`, 9 `dead_bounce`, 5 `tracking_artifact`, and 5
  `ambiguous`.
- Only 5 of 44 are artifacts. Most detections still fire on a real ball event;
  the detector's main failure mode is over-reporting real events that are not
  rally bounces. That makes this primarily a classification problem, not a
  hallucination problem.
- `rally_scoring_eligible` contains zero racket contacts and zero dead balls,
  and zero tracking artifacts, which is the property rally scoring depends on.
  Live precision is 72.7% inside it, 9.1% outside.
- `near_player` separates almost all of it: 75.0% live precision when False,
  6.2% when True.
- `shape_confidence` does NOT predict event type. It grades trajectory quality,
  not what the event is; do not use it as a proxy for correctness.
- `court_margin_ft=3.0` is optimal against these labels. Tightening to 0 loses
  two real out-of-court bounces including a serve fault landing; loosening to 10
  keeps fewer real bounces (NMS suppression) and adds unlabelled detections.

Regenerate the detections with `eval_bounce_detect.py --review-csv`; the `label`
and `note` columns are hand-added and are not reproducible from code.

When editing a label CSV, re-derive any quoted counts in this README and in any
`docs/experiments/` file that cites the labels in the same commit. The label
frames are guarded by `eval_bounce_detect.py --check-labels`; prose summaries are
not, so stale counts have to be caught by review discipline.
