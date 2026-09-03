# tennis9 f1446: Suppression Keeps The Racket Contact And Discards The Bounce

Date: 2026-09-03
Author: Federer
Diagnosis only. No thresholds or suppression changed.

## Verdict: non-maximum suppression

A third, distinct cause. `f1147` and `f1401` are tracker mis-associations; this
one is not. The track here is clean and continuous through the whole event, the
detector finds the bounce, and then throws it away.

## What happens

With suppression disabled, four candidates exist in the window:

| frame | shape | score | world |
| --- | --- | ---: | --- |
| 1444 | medium | 4.53 | (23.3, 62.5) |
| **1445** | **high** | **8.08** | (23.5, 66.7) |
| 1455 | high | 9.85 | (28.4, 71.0) |
| **1456** | **high** | **12.07** | (28.9, 67.4) |

Suppression ranks by shape grade then score, and discards anything within 18
frames of a kept candidate. `f1456` wins on score, and `f1445` is 11 frames
away, so it is suppressed. Only `f1456` survives.

## Why that is the wrong survivor

From the source frames:

- **f1445 is the ground bounce.** The ball is low on the court beside its own
  shadow, and the player is still winding up.
- **f1456 is a racket contact.** The ball is in the strings; the player has swung.

So suppression kept the racket contact and discarded the bounce. The 18-frame
window — chosen so that genuine double bounces about 20 frames apart stay
distinct — also spans the gap between a bounce and the shot that follows it,
which here is 11 frames.

`f1447` also passes every gate but fails on residual by 0.3 px (18.3 against a
limit of 18.0), so a marginal threshold is adjacent to this too. It is not the
cause: even passing, it would be suppressed by `f1456` for the same reason.

## Scope of the damage

Both candidates are `near_player=True` and `confidence=low`, so neither is
rally-scoring eligible. **This costs bounce recall, not scoring correctness.**

## What would fix it, and what would not

Shrinking the suppression window is the obvious move and is wrong: it exists to
keep real double bounces separate, and a bounce followed by a shot can be closer
together than two bounces are.

The ranking is the actual problem — score is not a proxy for "is a bounce". A
racket contact can outscore a ground bounce because both are clean trajectory
reversals, which is the same result measured in
`tennis11_near_player_veto.md`: shape says a reversal happened, never what it
bounced off.

This is a concrete case where the shadow signal would decide it. `f1445` sits
beside its shadow, `f1456` is at racket height with the shadow well below. Both
are near-court, which is exactly where
`tennis11_ball_shadow_signal.md` measured the signal to work. That is now two
independent problems — the near-player veto and this suppression ranking — that
the same unbuilt signal would address.
