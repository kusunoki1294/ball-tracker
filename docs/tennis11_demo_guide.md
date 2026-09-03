# Tennis11 Demo Guide

Date: 2026-09-03

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
  observation, 5 accepted serve-motion hypotheses, and 12 suppressed motions.
- A source-video label pass says the four suppressed near-side contacts
  (`f623`, `f2517`, `f3506`, `f4183`) are real serves. The far-side non-serve
  reads are lower confidence because the far player is small in frame.
- The detailed audit reports this as contact-label evidence: 5 of 5 accepted
  game-2 contacts are labelled serves, and 4 labelled serves are suppressed.
- The front-page review priorities show both failure classes: verified
  suppressed serves and far-side non-serve controls.
- The f786 game-1 control also carries a stuck-track warning in the detailed
  JSON: the suppressor's rally evidence is correct on outcome but weak on
  mechanism, so that metric should not be tuned blindly.
- That warning does not explain the four verified game-2 serves that remain in
  the suppressed set; those are still a separate suppression/recall problem.

Do not claim:

- Full automatic scoring.
- Verified game-2 point boundaries.
- Verified game-2 source timestamps. Game 2's clip start is recorded from the
  cut command, but unlike game 1 it has not yet been pinned to a hand-verified
  contact in the full source video.
- That game-2 ball-track evidence is clean. Ball-derived landings are still
  review evidence, and the contact labels are the stronger source for whether a
  frame is a serve.
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
- `game2_serve_contact_review.html`: crop strips for 5 accepted and 12 suppressed
  game-2 contacts.
- `timeline_preroll_review.html`: full-court pre-roll trail cards for f786 plus
  the game-2 suppressed serves and far-side reviewer reads. The trail is drawn
  in image space, not ground-projected court space, dots preserve tracked bbox
  size, and each card reports how much of the interval had a tracked ball.
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
.venv/bin/python validate_project.py
.venv/bin/python validate_demo_artifacts.py
.venv/bin/python validate_timeline_pipeline.py
.venv/bin/python validate_serve_detection.py
.venv/bin/python validate_tennis9_regression.py
```

Run `validate_demo_artifacts.py` immediately before showing or sending the
bundle. It checks the files currently on disk, not just temp regenerated output.
Run `validate_project.py --full` when you also want the slower regenerated
timeline-pipeline check.
