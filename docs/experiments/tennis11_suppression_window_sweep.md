# The Rally-Continuation Window Is Accidentally Right and Should Be Left Alone

Date: 2026-09-03
Author: Nadal

## Question

Four verified game-2 serves — f623, f2517, f3506, f4183 — are suppressed as
rally continuations instead of being promoted to point starts. The suppressor
fires when `gap <= RALLY_CONTINUATION_SUPPRESS_SECONDS` (15.0) and the ball is
measured as having returned over the net in between. Their gaps are 6.5-13.4s,
all inside the window. Can a shorter window recover them?

## Answer: no, and the window should not be changed

No value satisfies the controls. Game 1's zero-spurious requirement fails at
every value below 15s.

| window | g1 hyps | g1 verified | g1 spurious | f786 | f1659 | g2 accepted | g2 recovered | g2 false+ |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| **15.0** | **6** | **7/7** | **0** | suppressed | promoted | 5 | 0/3 | **0** |
| 12.0 | 8 | 7/7 | 2 | suppressed | promoted | 6 | 2/3 | 0 |
| 10.0 | 8 | 7/7 | 2 | suppressed | promoted | 8 | 3/3 | 1 |
| 9.0 | 8 | 7/7 | 2 | suppressed | promoted | 9 | 3/3 | 2 |
| 8.0 | 8 | 7/7 | 2 | suppressed | promoted | 9 | 3/3 | 2 |
| 7.0 | 8 | 7/7 | 2 | suppressed | promoted | 9 | 3/3 | 2 |
| 6.0 | 8 | 7/7 | 2 | suppressed | promoted | 11 | 3/3 | 2 |

f786 stays suppressed and f1659 stays promoted at every value, so the window is
not what protects those two. It is protecting game 1 precision.

## The evidence is healthy; the question is wrong

Nothing about the four is degraded:

| frame | gap | return fraction | tracked frames | stuck dominates |
| ---: | ---: | ---: | ---: | --- |
| f623 | 12.2s | 0.482 | 168 | no (9%) |
| f2517 | 13.4s | 0.425 | 153 | no (9%) |
| f3506 | 11.7s | 0.458 | 120 | no (10%) |
| f4183 | 6.5s | 0.741 | 170 | no (27%) |

Plenty of track, no stuck runs, fractions 3-5x the 0.15 threshold. The ball
really did cross the net repeatedly in each interval — because a whole point was
played, ended, and the server set up and served again. "Did the ball come back
over the net between these two strikes" is trivially yes over 11-13 seconds, and
says nothing about whether the earlier point is still live.

The rule's premise — nothing returned, therefore this is a second serve of the
same point — only holds when the two strikes could plausibly belong to one
point. The cases it gets right sit at ~5s: f786 suppressed at 5.0s, f1659
promoted at 5.8s. The failures sit at 6.5-13.4s. The rule works in the regime it
was designed for and is being applied outside it.

## Why 15s works at all, which is an accident

The two spurious game-1 hypotheses that appear below 15s are f1040 and f1079,
both inside point 2's 30-second rally. The window is long enough to cover that
rally, so it masks them. It is tuned to the longest rally in the data, not to a
second-serve interval. That is why shortening it exposes them immediately, and
why the current value should not be read as a calibrated choice.

## Two fixes tested, both rejected

**Gate on motion source.** Only `ball_toss` motions escape suppression. Fixes
game 1 — both spurious detections are `peak_reach` — and does nothing for game
2, whose two false positives at an 8s window (f918, f1931) are both `ball_toss`.

**Gate on reach prominence.** On game 2 this looks decisive:

| | prominence / box height |
| --- | --- |
| serves (n=9) | 0.19 - 0.28 |
| non-serves (n=6) | 0.00 - 0.14 |
| the one `ambiguous` label | 0.19, exactly on the boundary |

Then game 1 collapses it. f786 scores **0.25**, inside the game-1 serve range of
0.24-0.31. f786 is a genuine overhead — full extension above the head — and no
posture cue separates a serve from a smash. Game 2's non-serves are all
waist/shoulder-height groundstrokes, which is the easy problem; game 1's f786 is
the hard one.

This is the trap worth recording: a threshold fitted on game 2 alone would have
looked like a clean discriminator with a comfortable gap, and would have broken
the game-1 control on the very next clip.

## Decision

No production threshold change. 15.0s is the best of the tested options under
the stated controls, and the four missed serves are its price.

The next lever is not a better threshold. The elapsed-time proxy answers "was
there ball traffic recently" when the question is "did the previous point end".
Nothing currently computes that. Until something does, recovering these four
costs game 1 precision, which is not a trade worth making on a clip whose
headline claim is zero spurious accepted hypotheses.

## Reproducing

`RALLY_CONTINUATION_SUPPRESS_SECONDS` in `timeline_hypotheses.py` is the swept
value. Call `build_hypotheses` directly per clip rather than running the full
pipeline; no rendering is needed and the sweep takes seconds. Controls are the
7 verified contacts in `validate_serve_detection.EXPECTED_SERVES` for game 1 and
`labels/tennis11_game2_contact_labels.csv` for game 2.
