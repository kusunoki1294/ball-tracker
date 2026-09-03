# The Near-Player Veto Is Correctly Calibrated

Date: 2026-09-01
Author: Federer

## Question

`bounce_detect` collapses confidence to `low` whenever a bounce is near a player
(`confidence = "low" if near_player else shape_confidence`), which makes it
ineligible for rally scoring. That veto hides two real bounces in tennis11
game 1. Can it be relaxed without letting racket contacts back in?

## Answer: no, and the veto should be left alone

It rejects 28 things correctly and costs 2:

| label | n | near_player=True |
| --- | ---: | ---: |
| racket | 14 | **14** |
| dead_bounce | 9 | **9** |
| tracking_artifact | 5 | **5** |
| live_bounce | 11 | 2 |
| ambiguous | 5 | 2 |

(Counts re-measured 2026-09-01 after the labels were re-aligned to post-`aa8f69b`
detector output, then corrected after the shadow experiment showed `f1075` was a
racket contact, not a ground bounce. The earlier draft cited 14/10/3 against the
stale keying, then 13/9/5 with 3 live bounces lost before the `f1075` relabel.
The conclusion is unchanged and stronger: 28 correct rejections, not 26, and
only 2 real bounces lost.)

The two real bounces it costs are `f2253` (near-court bounce, shadow confirms
ground) and `f2114` (P4 serve landing). `f2114` survives anyway through the
serve-landing exemption, so the true rally-scoring loss is one.

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

Promising until the hidden bounces are checked: `f2253` is -1.070, inside the
racket range. The other case that originally made this look worse, `f1075`
(-0.670), was later re-checked and relabelled as a racket contact, so it does
not support relaxing the veto.

The measure is confounded by depth. Comparing image rows only means something
when the two objects are at the same distance, and a ball bouncing deep behind a
player who is close to the camera appears above that player's feet while being
firmly on the ground. This is the same image-row-as-depth error that produced
the retracted phantom-ball claim and the broken far-side test; it does not
become valid by being applied to players instead of the net.

## Where the signal actually has to come from

Recovering these needs a *new* observation, not a re-weighting of existing ones.
The most promising candidate was in the hand label itself: "shadow confirms
ground". That has now been checked in `tennis11_ball_shadow_signal.md`. The
signal is real on the near court, but it is a gap-minimum signal rather than a
"touching" test, and it does not survive far-court scale. It remains an
experiment, not a gate.

## Status

No change made. The veto stands as written.
