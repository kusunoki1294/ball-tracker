# RETRACTED: "A Phantom Ball Track Corrupts Game 2's Serve Evidence"

Date: 2026-08-31 (retracted same day)
Author: Federer

**The claim in this document was wrong. The detection at `f183` is a real tennis
ball.** The file is kept because the reasoning error is worth not repeating.

## What I claimed

That from f180–f196 the tracker was locked onto a background object above the far
court, because the detection sat 144 px above the far player's head at 23 px
across — "a third of that player's entire body height", therefore impossible.

## Why it was wrong

I zoomed into the raw frames at the reported coordinates. The ball is plainly
there: bright, well-resolved, against the trees at f181/f183 and against the dark
wall at f187/f191. The box is offset a few pixels low and right, but it is a
genuine detection of a genuine ball.

It is the **near player's service toss**. It appears high in the frame because it
was tossed high, and it stays 23 px across because it is close to the camera.
Checked against the player's own scale: his box is 462 px for ~5.8 ft, so ~80
px/ft; the ball sits ~331 px above his head, about 4 ft — an ordinary toss.

**The error was comparing a near-court ball to the far player's head because they
were adjacent in image rows.** That is treating image row as depth — the exact
mistake I had flagged in the bounce thresholds, in the racket-cue metric, and in
my own far-ball side test the day before. I made it while writing the argument
against it.

## What does not survive

- The claim that the real ball was rejected and a phantom kept. `ball_debug` does
  show two candidates with one rejected as low-excursion, but the one that was
  **kept is the real ball**.
- The claim that `f183`'s "fault" landing was computed from a phantom.
- The claim that the far-half counts in `554c5f8` were corrupted by phantoms.
- The reading of the y<150 size band as physically impossible. Those large,
  high-in-frame detections are near-player service tosses, which are legitimately
  both.

## What does survive, from elsewhere

The separate defect in `tennis11_game2_suppressed_serves.md` is real and
independently confirmed: comparing ball image row against the net's image row is
not a side test for an airborne ball, because a near toss appears above the net
line. That is the same physical fact as this retraction — read from the correct
direction.

The size-cap result there still holds and is strengthened by this: discarding
oversized "far" detections works precisely *because* those are near tosses.

## Lesson

Apparent size constrains depth only together with a depth estimate. A large ball
high in the frame is near and lofted; a small ball high in the frame is far. I
had the rule and still applied it backwards, because I anchored to the nearest
object in image space rather than to the player who actually hit the ball.
