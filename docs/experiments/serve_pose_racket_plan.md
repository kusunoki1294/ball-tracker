# Serve Detection With Racket And Pose Cues

Date: 2026-08-30

## Product Question

Can serve detection be strengthened with arms or racket position, instead of relying only on
ball toss plus player-box reach?

## Current Answer

Use the existing racket boxes first. The YOLO scene pass already logs `tennis racket`
detections; adding a pose model is a larger dependency and should be treated as a second-tier
experiment, not the next default path.

Measured on `yoloVids/outputs/tennis11/ai11.1.jsonl`:

| clip | frames | frames with racket | racket detections |
| --- | ---: | ---: | ---: |
| tennis11 game 1 | 3,330 | 2,139 (64.2%) | 2,924 |

That is enough coverage to test whether racket proximity and racket height can improve serve
contact confidence without re-tracking or downloading a new model.

## Cheap Experiment

Inputs:

- `yoloVids/outputs/tennis11/ai11.1.jsonl`
- `yoloVids/outputs/tennis11/ai11.g2.jsonl`
- `yoloVids/outputs/tennis11/timeline/game_1_hypotheses.json`
- `yoloVids/outputs/tennis11/timeline/game_2_hypotheses.json`
- existing game-1 verified serve contacts: `159, 635, 1485, 1659, 2091, 2432, 2952`

For every accepted and suppressed serve motion, compute:

- nearest racket-box distance to the tracked ball around `contact_frame -4,-2,0,+2`
- racket-box vertical position relative to the claimed server's player box
- whether a racket box overlaps or sits near the server's extended hand area
- whether the cue agrees with the claimed server side

Report by bucket:

- game-1 verified contacts
- game-1 suppressed false positives
- game-2 accepted motions
- game-2 suppressed motions

Success criterion:

- preserve game-1 verified contact recall at `7/7`
- separate game-1 verified contacts from suppressed false positives better than the current
  source/confidence rules
- do not make far-side game-2 output look more certain than the footage supports

## Pose Model Tier

A pose model could help if it provides reliable wrist/elbow/shoulder landmarks at the far end.
There is no local pose dependency in the project right now (`mediapipe` is not installed), and
the current local YOLO assets do not include a dedicated pose or racket-contact model.

Do not add pose to the production path without first measuring it on crops:

- near side: can it locate the racket-side wrist within a few frames of contact?
- far side: can it locate arm posture when the player is roughly 85 px tall?
- does it add information beyond the existing player-box top and racket-box detections?

## Recommendation

Next implementation slice: `serve_racket_cue_eval.py`, experiment-only. It should read existing
JSONL and timeline outputs, write a CSV/HTML audit, and make no analyzer decision changes until
the cue proves it separates verified contacts from known false positives.

## First Experiment Result

Implemented `serve_racket_cue_eval.py` and ran it on game 1 and game 2:

```bash
.venv/bin/python serve_racket_cue_eval.py \
  --clip "game 1=yoloVids/outputs/tennis11/ai11.1.jsonl=yoloVids/outputs/tennis11/timeline/game_1_hypotheses.json" \
  --clip "game 2=yoloVids/outputs/tennis11/ai11.g2.jsonl=yoloVids/outputs/tennis11/timeline/game_2_hypotheses.json" \
  --verified-contacts "game 1=159,635,1485,1659,2091,2432,2952" \
  --output-csv yoloVids/outputs/tennis11/timeline/serve_racket_cue_eval.csv \
  --output-html yoloVids/outputs/tennis11/timeline/serve_racket_cue_eval.html
```

Summary:

| bucket | n | with racket box | median ball-racket px | median server-racket px |
| --- | ---: | ---: | ---: | ---: |
| game 1 accepted, verified true | 7 | 7 | 23.0 | 11.6 |
| game 1 suppressed, verified false | 6 | 5 | 40.0 | 15.5 |
| game 2 accepted, unverified | 9 | 9 | 332.9 | 19.5 |
| game 2 suppressed, unverified | 8 | 8 | 260.0 | 0.0 |

Interpretation:

- Racket boxes are present often enough to use as review/audit evidence.
- They are not usable reliably enough to gate serve detection. Independent manual review
  found that on 2 of 7 verified game-1 serve contacts, a racket box exists in the frame
  but is not the server's ball-contact racket: the nearest racket sits 227.7 px and
  286.7 px from the ball. Any server-racket-required gate would reject 29% of real serves
  before thresholds enter the picture.
- They do not cleanly separate verified serves from known false positives on game 1 even
  when present. A 30 px ball-racket gate would keep only 4/7 verified contacts while still
  keeping 2/5 measured suppressed false positives with ball-racket distances.
- Normalizing the posture cue does not rescue it:

  | cue | verified true | known false positives |
  | --- | --- | --- |
  | `racket_above_server_top_frac` | -0.181, -0.169, -0.087, -0.009, 0.069, 0.166, 0.167 | -4.713, -0.184, -0.086, -0.069, 0.143 |

  Keeping all verified contacts would admit all measured false positives, so the failure is
  not just raw-pixel scale dependence.
- Racket-above-head posture is useful as an audit flag, but cannot distinguish a serve from
  a rally overhead. The known f786 rally overhead has the same racket-above-head geometry as
  verified serves; only prior rally evidence separates it.
- Far-side ball-racket distance is not comparable to near-side distance. On game 2 the
  ball is often too small or unresolved, so the metric can report hundreds of pixels even
  when the motion looks serve-shaped in the contact strip.

Conclusion: keep racket cues as diagnostics for now. Do not wire them into serve detection
or scoring until they are validated against labelled game-2 contacts and a larger false-positive
set. This experiment is enough to disqualify racket gating on the current evidence; it does
not prove that racket cues carry no information at all.
