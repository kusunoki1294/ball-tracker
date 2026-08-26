# Gap interpolation on the ball track — closed, negative result

2026-08-26. Experiment only; no code changed as a result.

## Question

tennis11 point 5's serve landing is lost because the ball track has an 11-frame
hole and the bounce falls inside it. Could the positions inside such holes be
reconstructed cheaply — by fitting the surrounding ballistic arc — so that
`bounce_detect.py` can find those bounces without re-running the tracker?

This mattered because the answer decides whether ball-track recall is a cheap
problem (post-processing) or an expensive one (detector/model work).

## Result: no. Baseline wins on every axis.

Measured against `labels/tennis11_game1_bounce_labels.csv`, where "live" is the
12 hand-labelled `live_bounce` detections.

| variant | detections | live recall | new unlabelled | serve windows | P5 |
| --- | --- | --- | --- | --- | --- |
| A baseline, tracker output as-is | 44 | **12/12** | 0 | 6/7 | none |
| B coasted/held frames removed | 41 | 10/12 | 5 | 5/7 | none |
| C interpolation added, gap<=6 | 39 | 10/12 | 0 | 5/7 | none |
| C interpolation added, gap<=12 | 39 | 10/12 | 4 | 6/7 | [2452] |
| D strip coast, then interp gap<=6 | 38 | 10/12 | 3 | 5/7 | none |
| D strip coast, then interp gap<=12 | 37 | 10/12 | 6 | 6/7 | none |

Every variant loses two real bounces.

## P5 was not recovered; the one apparent hit is a seam artifact

C at gap<=12 produces a candidate at frame 2452. It is not a recovery. Tracing
the fill: real samples descend through 2450 (y=315) and 2451 (y=322), then the
interpolated curve must bend upward to meet the post-gap samples at 2464
(y=264). The detected "bounce" is the seam where the fitted curve turns, not
evidence that the ball did. It costs two genuine bounces to buy.

## Why this cannot work, structurally

A bounce **is** a discontinuity in the trajectory. Fitting a smooth arc across a
hole models the ball as though nothing happened inside it, which erases exactly
the signature the detector looks for. Any scheme that reconstructs positions by
smoothing is structurally incapable of recovering a bounce that occurred inside
the gap. This is not a tuning problem.

## The tracker already interpolates — and that changes the headline number

`track_ball_yolo.py` emits held-over positions flagged
`"motion_gate": "coast"`, `"interpolated": true`. On tennis11 game 1:

| | frames | of 3330 |
| --- | --- | --- |
| frames with a ball | 2090 | 62.8% |
| carrying the tracker's interpolated flag | 222 | |
| exact position repeat of the previous frame | 199 | |
| **distinct real observations** | **1891** | **56.8%** |

So "62.8% of frames have a ball" overstates real observation by six points, and
roughly one in ten "tracked" frames is a held position with zero motion being
fed to the arc fits as though it were data. **Use 56.8% when describing tennis11
track quality.**

Variant B is the telling one: removing the coasted frames makes results *worse*
(12/12 -> 10/12 live, +5 spurious). The tracker's coasting is already doing the
useful part of gap filling, which is why stacking another layer on top only
degrades — it interpolates already-interpolated data.

## Recommendation

Close this line. Do not revisit interpolation unless the INPUT changes: a
detector that actually observes the ball through these moments (higher far-court
resolution, a better ball model, or temporal detection). P5-class holes are a
detection problem, not a reconstruction problem.

## Reproducing

The harness is not in the repo (it produced no shipped code). It re-ran
`detect_bounces` over the logged JSONL with positions filled by a quadratic fit
to the surrounding *original* observations, and scored against the labels file.
The load-bearing inputs are `yoloVids/outputs/tennis11/ai11.1.jsonl` and
`labels/tennis11_game1_bounce_labels.csv`, both of which are preserved.
