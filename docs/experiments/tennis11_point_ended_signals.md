# Point-Ended Evidence: What Separates A Live Rally From A Serve About To Happen

> **RETRACTED 2026-09-04.** The receiver-travel result below does not hold. A
> player-box stability audit showed the metric is ~99% a detector-swap counter:
> median per-frame receiver movement is identical between rally and serve windows
> (0.006 vs 0.006), and the separation came from the three controls happening to
> sit in swap-heavy windows. Neither a confidence-filtered version nor mean
> detection confidence generalises. Do not implement the veto described here.
> See `tennis11_player_box_stability.md`. The sections on signals that do NOT
> separate remain valid.

Date: 2026-09-03
Author: Federer
Experiment only. No promotion rule built, no threshold changed.

## Result in one line

**Receiver stillness is the only signal that separates the controls, and it is
necessary but not sufficient**: it reliably says "a rally is in progress", and it
cannot say "a point is starting".

## Signals that do not separate

Measured over the gap between the previous accepted serve and the candidate:

| signal | live-rally controls | verified point starts |
| --- | --- | --- |
| gap seconds | 5.0, 13.5, 14.8 | 14.7, 13.4, 11.7, 6.5 |
| ball tracked fraction | 0.51, 0.77, 0.78 | 0.54, 0.38, 0.34, 0.87 |
| net crossings | 4, 8, 9 | 8, 13, 4, 2 |
| longest dead stretch | 2.43, 2.43, 2.43 | 2.77, 2.2, 5.3, 0.83 |

All four overlap. They describe the whole interval, when the question is about
its final moment.

## The signal that does separate

Receiver travel in the second before contact, measured as summed box-centre
displacement in units of the receiver's own box height — **scale-invariant, no
homography, so no perspective amplification**:

| case | travel | truth |
| --- | ---: | --- |
| g1 f786 | 42.41 | rally live |
| g1 f1040 | 36.09 | rally live |
| g1 f1079 | 80.35 | rally live |
| g1 f1659 | 5.09 | second serve, same point |
| g2 f623 | 0.27 | point start |
| g2 f2517 | 0.42 | point start |
| g2 f3506 | 0.29 | point start |
| g2 f4183 | **0.14** | point start (the hard positive) |

An ~85x margin, and `f4183` — the hard positive, gap only 6.5 s — is the
*cleanest* case, because the signal does not use the gap at all.

It also respects the stated constraints: it reads only the receiver's box, never
stuck tracks, net-line dead balls or tracking holes. And it does not distinguish
a first serve from a second, so `f1659` scores with the serves and second-serve
grouping stays a separate concern.

## Why it is not sufficient

Tested beyond the controls, across all 16 verified serve contacts and 239 random
non-serve frames:

- 14 of 16 serves score below 1.0
- **but 49% of random non-serve frames also score below 1.0**

A still receiver happens constantly — between points, during dead ball, whenever
someone is standing. So a low score cannot establish that a point is starting. A
*high* score is the informative direction: the receiver is visibly in motion, so
a rally is live and the motion must not be promoted.

**Use it as a veto, not a promoter.**

## The one serve that scores like a rally is a different bug

`g1 f1485` is a verified serve scoring 77.86. The cause is not the metric:
`player_far` alternates between two objects, a stable box at (856, 258) h=91
conf 0.80 and a spurious one at (458, 255) h=46 conf 0.26, flipping every few
frames.

That is a **player-detection mis-association**, the same class as the ball
association failures in `tennis9_association_signals.md`, one level up. It has an
obvious signature: alternation between a high-confidence large box and a
low-confidence small one. Worth its own audit; this metric is only as stable as
the player boxes it reads.

## Recommendation

Use receiver motion as a veto on promotion, alongside whatever positive evidence
is chosen, and expect it to be unreliable wherever player detection alternates.
Fixing the player-box alternation would likely raise its value more than tuning
its threshold.
