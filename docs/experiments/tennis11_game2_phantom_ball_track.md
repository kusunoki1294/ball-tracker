# A Phantom Ball Track Corrupts Game 2's Serve Evidence

Date: 2026-08-31
Author: Federer

## What I found

While testing whether game 2's far-attributed contacts are serves, I checked the
raw ball track instead of trusting derived fields. From **f180 to roughly f196**
the tracked "ball" is not the ball.

At `f183` — the moment of the near player's serve contact — the tracker reports:

| | |
| --- | --- |
| tracked ball | centre `(957, 71)`, bbox 23 px, confidence 0.78 |
| far player | box `[1151, 215, 1180, 289]` — whole body 74 px tall, head at y 215 |
| near player | box `[795, 402, 935, 864]`, head at y 402, x centre 865 |

The tracked object sits **144 px above the far player's head** and is **23 px
across, roughly a third of that player's entire body height**. No tennis ball at
that depth can be a third of a person. A real ball there would be 3–8 px.

It is not the near player's toss either: it is 92 px right of his head and 330 px
above it, which would mean a toss several metres over his head.

It is a background object above the far court, tracked at high confidence for
~2.5 s while drifting smoothly downward — smooth motion is exactly what lets it
pass the motion gate.

## Why the real ball was dropped

`ball_debug` at f183 shows the mechanism:

```
moving_filter: input_count 2, passed_excursion 1, rejected_low_excursion 1
counts:        stationary_count 2, moving_count 1
```

Two candidates. The one that moved a lot was kept; the one that moved little was
rejected as stationary. A ball at the top of a service toss is nearly stationary.
**The gate that suppresses static false positives also discards the toss at its
apex**, and here the phantom won the selection.

## Population-level symptom

Apparent ball size against image row, whole clip (n = 2351):

| image y | n | median size |
| --- | ---: | ---: |
| 0–150 | 182 | **22.0 px** |
| 150–300 | 854 | 11.0 px |
| 300–450 | 692 | 13.0 px |
| 450–600 | 194 | 18.0 px |
| 600–750 | 386 | 23.0 px |
| 750–1080 | 41 | 63.0 px |

From y=150 downward the progression is exactly perspective: 11 → 13 → 18 → 23 →
63. The top band breaks it — detections up there are as large as balls beside the
near player, which is physically impossible. That band is ~7.7% of all tracked
detections in the clip.

## What this invalidates

- **`f183`'s "fault" landing.** It was computed from the phantom, not the serve.
  The far player then plays the ball at `f257`, which is consistent with the
  serve having gone in.
- **My far-half ball counts in `554c5f8`.** Those counted tracked detections in
  the far half of the image. Where the track is a phantom sitting in the trees,
  the count is meaningless. The measured 3/7 and 0/6 rates on game 1 need
  re-checking against a clean track before they are relied on.
- **My `near_half = 0` result for the five far contacts.** Same defect. It looked
  like clean separation and was partly an artifact.

The *visual* labels in `tennis11_game2_contact_verification.md` are unaffected —
they came from watching the swing, not the ball track.

## Suggested fix, with a caveat

A size-versus-depth consistency gate would reject this class outright: at the
depth implied by the far player's scale, a 23 px ball is impossible.

The caveat matters. Image row is **not** depth for an airborne ball — a near
player's toss appears high in the frame while staying close and therefore large.
A naive "high in frame must be small" rule would throw away real serve tosses,
which is the opposite of what we want. Any gate needs a depth estimate that does
not assume the ball is on the ground, or should be anchored to a nearby player's
box scale rather than raw image row.

## Status

Diagnosis only. No tracker or analyzer change made.
