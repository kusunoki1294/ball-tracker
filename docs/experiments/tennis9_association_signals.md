# Can Size Or Prediction Distance Catch The Tracker's Bad Associations?

Date: 2026-09-02
Author: Federer
Evidence only. No tracker selection changed.

## Question

`f1147` and `f1401` are both lost to the same failure: the tracker accepts a jump
to a different object, its prediction is poisoned, and real detections are then
rejected. Can either apparent-size discontinuity or distance-from-prediction
identify that bad association safely?

Measured over 3,164 consecutive-frame accepted steps pooled from tennis9 and
tennis11.

## Signal 1: size discontinuity — catches one of two, and is confounded

| | jump | size ratio |
| --- | ---: | ---: |
| f1405 (bad) | 77.8 px | **2.33** |
| f1150 (bad) | 72.9 px | **1.10** |
| all steps | p90 = 30 px | p50 = 1.07, p90 = 1.21 |

`f1405` stands out clearly. **`f1150` does not stand out at all** — a 1.10 ratio
is the population median.

It is also confounded: a legitimately fast ball grows, presumably from motion
blur. Median size ratio rises with jump distance — 1.07 below 25 px, 1.12 at
50–80 px, 1.43 above 150 px, where p90 reaches 3.49.

A `jump > 50 px AND ratio >= 2.0` rule fires on 7 of 3,164 steps. Checking each
for the poisoned-prediction signature: 3 look like real mis-associations
(f1405, and two previously unknown — **f1591 and f1839**), 4 continue tracking
normally. Roughly half its firings are real.

## Signal 2: distance from the tracker's own prediction — catches both, but overlaps

Prediction distance is only meaningful while tracking is continuous. Splitting on
`missed_frames_before` shows two different regimes:

| regime | n | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| continuous | 3,164 | 2.0 px | 11.0 px | 777 px |
| after a gap | 134 | 26.1 px | 527.9 px | 1,479 px |

Restricted to continuous tracking, both misses are conspicuous: `f1150` at
67.7 px and `f1405` at 70.6 px, each above 96% of accepted steps.

**But the threshold that catches them is inside the legitimate range.** At real
bounces — where the ball reverses — prediction error stays mostly small
(tennis9 p50 5.1, p90 31.2) but reaches a **maximum of 68.2 px**. The two bad
associations are 67.7 and 70.6. They overlap.

So a `> 60 px` rule fires on 82 of 3,164 continuous steps (2.6%), catches both
misses, and would also reject a genuine bounce association.

## Verdict

**Neither is safe as an automatic rejection rule on this evidence.** Size misses
half the cases and is confounded by speed. Prediction distance catches both but
cannot be thresholded without cutting into real bounce behaviour, and cutting
there is the worst possible place — a rejected association at a bounce destroys
the arc the bounce detector needs, which is the failure we set out to fix.

Both are usable as **audit signals**, where a firing means "a human should look",
not "reject this track".

## The actual bottleneck

Two labelled failures is too thin to fit an association rule to. This experiment
produced two more candidates worth labelling — `f1591` and `f1839` — by their
poisoned-prediction signature. Getting a labelled set of mis-associations, rather
than a cleverer statistic over two of them, is what would move this.

One encouraging measurement for later: the predictor handles real bounces better
than expected. p90 error at a bounce is 31 px against 12 px elsewhere, not the
blow-up a naive linear extrapolation would give. Whatever separates a bad
association from a bounce, it is not simply that bounces are unpredictable.
