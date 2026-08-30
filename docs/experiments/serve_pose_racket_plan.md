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
