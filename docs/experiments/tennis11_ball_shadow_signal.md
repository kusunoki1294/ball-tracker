# Does The Ball's Shadow Separate A Bounce From A Racket Contact?

Date: 2026-09-01
Author: Federer
Scope: experiment only. No gate built, no behaviour changed.

## Why

A ball touching the court meets its own shadow; a ball on a racket does not.
Unlike every other candidate tried for this problem, it does not require knowing
depth — which is what killed the contact-height measure and the far-side test.

## Findings

### 1. The signal is real on the near court

At `f2253` the ball and a distinct dark shadow are both plainly visible, and
tracking frames f2245–f2262 shows the gap closing as the ball descends and
reopening as it rises. The minimum sits at the detected bounce.

### 2. But ball and shadow never actually touch

Even at a real bounce the smallest gap is roughly one ball diameter. Two causes,
both structural: at 30 fps the contact instant usually falls between frames, and
the sun is off to one side so the shadow is laterally offset rather than
directly beneath.

**So a "do they touch" test would fail on real bounces.** The usable form is the
*magnitude* of the gap, or its minimum across a window — not contact.

### 3. It does not survive far-court scale

| frame | where | ball size | shadow |
| --- | --- | ---: | --- |
| 2253 | near court | 14 px | clear, distinct |
| 1013 | far service box | 10 px | none resolvable |
| 1104 | far service box | 11 px | none; ball is against the net |
| 186 | far, serve landing | 11 px | none; ball is against the net |

At the far end the ball is 10–11 px and frequently silhouetted against the net
or a line. There is nothing to measure. Note the three far cases are exactly the
ones the hand labels call "ball too small to adjudicate" — the human could not do
it either, so this is a property of the footage, not of the method.

**Verdict: a near-court-only signal.** Useful for the near-player veto problem,
which is a near-court problem, but it cannot become a general bounce test.

## A mislabel found on the way

`f1075` was labelled `live_bounce` ("deep near-court bounce"). The sequence
f1069–f1081 shows the ball arriving from the upper left and sitting **inside the
racket hoop** at the detected frame. It is a racket contact.

Corrected in `labels/tennis11_game1_bounce_labels.csv`. Consequence: the
near-player veto costs **2 live bounces, not 3**, so it is better calibrated than
`tennis11_near_player_veto.md` reported, and the case for relaxing it is weaker
still.

`f2253`, the other hidden bounce, is genuine — but note the ball is already
airborne at the detected frame, with a visible shadow gap, so the true ground
contact is a frame or two earlier.

## What would have to be true to use this

Gap in units of ball diameter, measured over a window rather than at one frame,
near court only, with an explicit abstain when no shadow is resolvable. That
abstain matters: without it the far court would read as "no shadow, therefore not
a bounce", which inverts the signal exactly where it has no information.

Not built. This records what the signal is and is not.
