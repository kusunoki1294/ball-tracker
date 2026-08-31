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
  observation, mixed near/far serves, and 9 accepted serve-motion hypotheses.
- Game 2 has four front-page review priorities: one known game-1 suppression
  control (`f786`) and three game-2 suppressed serve candidates (`f623`,
  `f3506`, `f4183`) that need human eyes.

Do not claim:

- Full automatic scoring.
- Verified game-2 point boundaries.
- Verified game-2 source timestamps. Game 2's clip start is recorded from the
  cut command, but unlike game 1 it has not yet been pinned to a hand-verified
  contact in the full source video.
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
- `timeline_preroll_review.html`: full-court pre-roll strips for f786 plus the
  three game-2 suppressed serve candidates. Yellow crosshairs mark tracked ball
  positions; "ball not tracked" is shown when the tracker has no ball for that
  frame.
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
