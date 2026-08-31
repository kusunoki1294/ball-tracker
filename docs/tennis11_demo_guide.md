# Tennis11 Demo Guide

Date: 2026-08-30

## Open This First

Use:

```bash
open yoloVids/outputs/tennis11/timeline/timeline_demo.html
```

Shareable bundle:

```text
yoloVids/outputs/tennis11/timeline/tennis11_timeline_demo.zip
```

The zip is the portable artifact. It includes the demo page, both review MP4s,
the detailed audit, contact sheets, racket-cue audit, and the hypothesis JSON.
Do not send `timeline_demo.html` by itself; its video and audit links are
relative files beside it.

## What To Say

This is an automated serve-motion timeline review, not automated scoring truth.
It finds serve-like contact moments from the tracked video and keeps uncertainty
visible instead of pretending point boundaries and winners are known.

Good current claim:

- Game 1 has 6 serve-motion hypotheses matching all 6 real point starts.
- Game 1 has 7 verified serve contacts found, including the double fault's
  second serve.
- Game 1 has zero spurious accepted serve-motion hypotheses after suppression.
- Game 2 is unverified but useful as a harder generalization clip: worse ball
  observation, 9 accepted serve-motion hypotheses, and 8 suppressed motions.
- A source-video label pass says every real game-2 serve is from the near end:
  accepted far-side contacts are rally shots, while four real near-side serves
  (`f623`, `f2517`, `f3506`, `f4183`) sit in the suppressed set.
- Game 2 also has a tracker anomaly absent from game 1: 159 large ball
  detections in the top image band, consistent with the f183 phantom-ball
  diagnosis. Treat game-2 ball-derived landings as review evidence, not truth.
- The front-page review priorities show both failure classes: verified
  suppressed serves and far-side non-serve controls.

Do not claim:

- Full automatic scoring.
- Verified game-2 point boundaries.
- Verified game-2 source timestamps. Game 2's clip start is recorded from the
  cut command, but unlike game 1 it has not yet been pinned to a hand-verified
  contact in the full source video.
- That game-2 ball-track evidence is clean. The demo now surfaces top-band
  large-ball anomalies precisely because the tracker can lock onto non-ball
  objects in that clip.
- That the colored timeline bars are point extents. They are padded review
  windows around serve-motion hypotheses.
- That racket cues solve serve detection. The racket audit says they are useful
  diagnostics, not a safe gate.

## Files Worth Opening

- `timeline_demo.html`: show-and-review entry page with embedded videos.
- `game1_timeline_hypotheses.mp4`: best verified demo video.
- `game2_timeline_hypotheses.mp4`: harder, unverified generalization video.
- `timeline_audit.html`: detailed table with confidence, landing, review, and
  contact metrics.
- `game1_serve_contact_review.html`: crop strips for all 7 verified game-1
  contacts.
- `game2_serve_contact_review.html`: crop strips for 9 accepted and 8 suppressed
  game-2 contacts.
- `timeline_preroll_review.html`: full-court pre-roll trail cards for f786 plus
  the game-2 suppressed serves and far-side non-serve controls. The trail is
  drawn in image space, not ground-projected court space, and each card reports
  how much of the previous two seconds actually had a tracked ball.
- `serve_racket_cue_eval.html`: why current YOLO racket boxes should stay audit
  evidence only.

## Regenerate

```bash
.venv/bin/python run_timeline_pipeline.py \
  --config timeline_configs/tennis11_games1_2.json \
  --render-videos \
  --bundle-demo
```

Validation:

```bash
.venv/bin/python validate_timeline_pipeline.py
.venv/bin/python validate_serve_detection.py
.venv/bin/python validate_tennis9_regression.py
```
