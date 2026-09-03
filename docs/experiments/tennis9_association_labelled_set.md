# Labelling The Tracker's High-Prediction-Error Steps

Date: 2026-09-03
Author: Federer
Evidence only. Recommendation at the end; no tracker change.

## The two new suspects are both RECOVERIES, not errors

Checked against the source frames, not just the track.

**f1839.** At f1835 the tracker is on the real ball (size 8, high in frame near
the far player). At f1837–f1838 it is on the **near player's body** (size 17–20,
bottom of frame). At f1839 it jumps back to the real ball (size 8) and tracks on
smoothly.

**f1591.** Same shape: the track sits on a large near-camera blob (size 18–20) at
f1589–f1590, then jumps to a small distant object (size 10) at f1591, which is
where it re-seeds six frames later.

In both cases **the step my size rule flagged is the tracker recovering**, not
the tracker erring. A rule that rejected those associations would have kept the
track on the player.

That is the opposite of the intended effect, and it is not visible from the
statistics — only from looking at which object each end of the jump is on.

## All 82 continuous high-prediction-error steps

Classified by whether the step moves onto a much larger blob (toward a
near-camera object), off one (back to the ball), or neither; and by whether the
track survives.

| | tennis9 | tennis11 |
| --- | ---: | ---: |
| similar size, track continues | 39 | 26 |
| similar size, track dies | 5 | 0 |
| into larger blob, track dies | 2 | 0 |
| into larger blob, track continues | 1 | 3 |
| out of larger blob, track continues | 2 | 2 |
| out of larger blob, track dies | 1 | 1 |

Calibration against the known cases:

| frame | class | outcome | what it is |
| --- | --- | --- | --- |
| f1150 | similar size | dies | error |
| f1405 | into larger blob | dies | error |
| f1591 | out of larger blob | dies | recovery |
| f1839 | out of larger blob | continues | recovery |

**Roughly 9 of 82 firings (11%) are associated with a dying track.** The other
73 continue tracking, and the sampled ones are legitimate fast motion.

## Recommendation: audit report, not a tracker feature

A rejection rule at this threshold would disturb about 73 healthy associations to
address 9 problematic ones, and among the 9 it cannot distinguish an error from a
recovery — rejecting the recoveries would entrench exactly the failure we are
trying to remove. Roughly eight-to-one against, with the errors and the fixes
sharing a signature.

An audit report is the right shape: list continuous steps above the threshold
with their size transition and whether the track survived, and let a person
label them. That also grows the labelled set, which is the actual bottleneck.

## What would change the answer

A signal that separates "moved onto a player-sized blob" from "moved back to a
ball-sized object" **with the court scale taken into account**, so that size is
read against expected ball size at that depth rather than against the previous
frame. Every attempt so far has compared a detection to its own neighbour, which
is why errors and recoveries look identical — they are the same jump in opposite
directions.

Not attempted. Noted because it is the first framing that would distinguish the
two, and because it needs the perspective scale that already exists in
`bounce_detect.PerspectiveScale`.
