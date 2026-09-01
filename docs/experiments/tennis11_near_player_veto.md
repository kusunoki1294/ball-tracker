# The Near-Player Veto Is Correctly Calibrated

Date: 2026-09-01
Author: Federer

## Question

`bounce_detect` collapses confidence to `low` whenever a bounce is near a player
(`confidence = "low" if near_player else shape_confidence`), which makes it
ineligible for rally scoring. That veto hides three real bounces in tennis11
game 1. Can it be relaxed without letting racket contacts back in?

## Answer: no, and the veto should be left alone

It rejects 26 things correctly and costs 3:

| label | n | near_player=True |
| --- | ---: | ---: |
| racket | 14 | **14** |
| dead_bounce | 10 | **10** |
| tracking_artifact | 3 | 2 |
| live_bounce | 12 | 3 |
| ambiguous | 5 | 2 |

The three real bounces it costs are `f1075` (deep near-court bounce), `f2253`
(near-court bounce, shadow confirms ground) and `f2114` (P4 serve landing).
`f2114` survives anyway through the serve-landing exemption, so the true loss is
two.

## Why the obvious relaxation fails

"Downgrade one level instead of flooring at low", so a near-player bounce with
`shape_confidence=high` becomes `medium` and stays eligible:

**+2 real bounces recovered, +18 false positives admitted.** 9:1 against.

The reason is visible in the shape grades of near-player detections:

| label | shape=high | shape=medium |
| --- | ---: | ---: |
| live_bounce | 2 | 1 |
| racket | 7 | 7 |
| dead_bounce | 10 | 0 |

**A ball coming off a racket produces just as clean a trajectory reversal as a
ball coming off the ground.** Every dead bounce scores high too. Shape carries
no information about *what* the ball bounced off, which is precisely why the
positional veto exists.

## A discriminator I tried and rejected

Compare the ball's contact point to the nearest player's feet: a ground bounce
should sit at or below them, a racket contact above.

| label | median gap (player box heights) | range |
| --- | ---: | --- |
| racket | -0.579 | [-1.186, -0.407] |
| live_bounce | +0.613 | [-1.070, 1.616] |

Promising until the two hidden bounces are checked: `f1075` is -0.670 and
`f2253` is -1.070 — both sit *inside* the racket range.

The measure is confounded by depth. Comparing image rows only means something
when the two objects are at the same distance, and a ball bouncing deep behind a
player who is close to the camera appears above that player's feet while being
firmly on the ground. This is the same image-row-as-depth error that produced
the retracted phantom-ball claim and the broken far-side test; it does not
become valid by being applied to players instead of the net.

## Where the signal actually has to come from

Recovering these needs a *new* observation, not a re-weighting of existing ones.
The most promising candidate is in the hand label itself: "shadow confirms
ground". A ball touching the court meets its own shadow; a ball on a racket does
not. That is a real physical difference, visible in this footage, and unlike
every measure above it does not depend on knowing depth.

Not attempted here. Recorded because it is the first idea in this area that is
not another way of rearranging the same three numbers.

## Status

No change made. The veto stands as written.
