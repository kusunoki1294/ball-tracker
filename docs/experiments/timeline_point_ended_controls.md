# Controls a Point-Ended Signal Must Satisfy

Date: 2026-09-03
Author: Nadal

A point-ended signal would replace the elapsed-time proxy in
`RALLY_CONTINUATION_SUPPRESS_SECONDS`, which is accidentally rather than
deliberately correct — see `tennis11_suppression_window_sweep.md`. These are the
controls any candidate must pass before it feeds `timeline_hypotheses`.

## Warning: f1659 proves second-serve grouping must stay separate

f1659 is the second serve of point 3's double fault. The first serve faulted, so
the point has **not** ended — yet f1659 must still be grouped as attempt 2.

A point-ended signal therefore cannot be the sole gate for promotion. If
promotion requires `point_ended=True`, the double fault breaks. Second-serve
grouping must remain a separate path that a point-ended signal does not veto.
Build this in from the start rather than discovering it when f1659 disappears.

## Must report STILL LIVE (strike stays suppressed)

| case | why it is hard |
| --- | --- |
| g1 f786 | Genuine rally overhead, 5.0s after the previous contact. `ball_toss`/high, and reach prominence 0.25 sits **inside** the g1 serve range of 0.24-0.31. No posture or toss cue rejects it. |
| g1 f1040, f1079 | `peak_reach` false positives inside point 2's 30-second rally. These are the precision controls: they are exactly what surfaces when the current window is shortened. |

## Must report ENDED (strike promoted to a point start)

| case | gap | note |
| --- | ---: | --- |
| g2 f623 | 12.2s | previous point genuinely over |
| g2 f2517 | 13.4s | previous point genuinely over |
| g2 f3506 | 11.7s | previous point genuinely over |
| g2 f4183 | 6.5s | **hardest positive.** Its gap sits in the legitimate rally-continuation range, so it is the case that shows whether the signal discriminates or has merely re-encoded elapsed time. |

## Must not be fooled by

Each of these has already produced a wrong answer somewhere in this system.

- **Stuck tracks.** f786's interval is 70% a stationary ball resting at the net
  (33 of 77 tracked frames, f652-f684). Any signal keying on "the ball stopped
  moving" fires here and reports ENDED on the one case that must say LIVE.
  Consume `serve_detect.ball_return_evidence`'s `stuck_track_run_frames` /
  `stuck_track_dominates` rather than re-deriving them.
- **Net-line contacts.** A serve clipping the cord dies near the net line (g1
  f1682, 1.08ft). "Ball went dead near the net" must not read as a point end —
  it can be a let or a net-cord fault mid-point.
- **Tracking holes are not dead time.** Measured longest untracked runs: the
  game-2 misses span 0.83-5.30s, f786 is 2.43s. They overlap, so "no ball for N
  seconds" cannot be the signal. Already disproven; do not re-test it.

## Must abstain rather than decide

- Ball track in the interval is stuck-dominated or too sparse. Reuse the
  existing `enough_ball_track` / `stuck_track_dominates` thresholds so that
  abstention means the same thing across modules.
- The previous serve's landing was never observed, so whether a second serve is
  even due is unknown. g1 P2 and P5 are live examples.
- No bounces at all in the interval.

Abstention must resolve to **suppress**, not promote. The reason must be
distinguishable from a positive "still live" verdict, in the same way
`serve_bounce_not_detected` is distinguishable from
`serve_bounce_net_line_contact`. "I could not tell" and "I checked, the point
was live" are different states and downstream must see which one it got.

## Bias

The costs are not symmetric. A false ENDED invents a point and appears in the
demo headline as a spurious hypothesis. A false STILL LIVE loses a serve, which
is the documented status quo. Bias toward STILL LIVE under uncertainty.

## Acceptance bar

All of the following, together:

- all three g1 negatives (f786, f1040, f1079) suppressed
- f1659 still grouped as the double fault's second attempt
- g1 still 6 hypotheses, 7/7 verified contacts, zero spurious accepted
- at least f623, f2517 and f3506 recovered
- no new g2 false positives against `labels/tennis11_game2_contact_labels.csv`

Anything short of this is not an improvement on the 15s window it would replace.
