# Are Game 2's Suppressed Motions Real Serves?

Date: 2026-08-30
Author: Federer

## Question

Game 2 (the 186–330s slice) accepts 9 serve motions and suppresses 8 others as rally
motions. All 8 suppressed are high confidence with source `ball_toss`. If any of them is
a real serve, the slice contains more point starts than the accepted set implies, which
bears directly on the open question of where game 2 ends and game 3 begins.

This is an attempt to narrow that question from available evidence. It is **not** ground
truth and does not license changing any suppression decision.

## Method

Two independent readings of each suppressed motion:

1. **Posture**, read off the contact strips in `game2_serve_contact_review.html`.
2. **Prior far-side ball**: count tracked ball detections in the far half of the court
   (image `y` above the calibrated net line) in frames `[c-150, c-45]`. A serve begins a
   point, so no ball has come across; an overhead or smash is a reply to a ball that did.

## Calibrating the second reading

Run against labelled game-1 data before trusting it on game 2.

| control set | n | test says "point start" |
| --- | ---: | ---: |
| verified serves (`159, 635, 1485, 1659, 2091, 2432, 2952`) | 7 | 3 |
| verified non-serves (`786, 941, 1040, 1079, 2491, 2867`) | 6 | 0 |

So on labelled data the test is **low recall (3/7) but did not fire on any known
non-serve (0/6)**. It is a high-precision, low-sensitivity indicator: "no far-side ball"
is evidence for a point start; "far-side ball present" proves nothing, because 4 of 7
real serves also show far-side activity — between points the previous ball is still being
tracked, and for a second serve the faulted first serve is in the far half.

Sample caveat: 0/6 is a weak specificity estimate. With n=6 the 95% upper bound on the
false-positive rate is roughly 50%, so this narrows the candidate set; it does not settle it.

## Result on game 2's suppressed motions

| frame | far-half detections | posture in strip | read |
| ---: | ---: | --- | --- |
| 408 | 33 / 55 | far player, too small to judge | no call |
| 623 | 0 / 10 | racket overhead, extended | **serve candidate** |
| 918 | 75 / 79 | racket overhead | reply |
| 1795 | 24 / 70 | racket overhead, extended | ambiguous |
| 1931 | 97 / 102 | racket low, not a contact | not a serve |
| 2517 | 6 / 25 | racket overhead, extended | ambiguous |
| 3506 | 0 / 19 | racket overhead, ball at racket | **serve candidate** |
| 4183 | 0 / 80 | racket overhead, ball at racket | **serve candidate** |

`f4183` is the strongest: 80 tracked ball detections in the preceding 3.5 s, none of them
in the far half, with a clear overhead contact in the strip.

A separate check — vertical ball travel in `[c-45, c-3]`, negative when the ball is rising
into a toss — does **not** discriminate: all 7 verified serves, all 9 accepted motions and
7 of the 8 suppressed motions are negative. It confirms these are overhead swings; it does
not separate a serve from any other overhead.

## Conclusion

Narrowed from 8 unexplained suppressions to **3 strong candidates: f623, f3506, f4183**.
In the source video these are approximately **3:27, 5:03 and 5:25** (clip starts at 186 s,
30 fps).

If any one of them is a real serve, the slice holds at least 10 point starts, which —
together with the accepted set already splitting 4 far / 5 near — makes it very unlikely
that 186–330s is a single game.

Not changing any behaviour on this evidence. The three frames above are what a human should
look at first.

---

## Correction (2026-08-31): the far-ball test had a units defect

The "far half" test above used image row versus the net's image row. **That is
not a side test for an airborne ball.** A ball lofted near the camera appears
high in the frame while still being on the near side, so the server's own toss
gets counted as a far-side ball.

That is not hypothetical — it is the mechanism behind the test's low recall. For
three of the four verified game-1 serves the test wrongly called "reply", the
far-labelled detections are large, i.e. near-depth:

| serve | far-labelled detections | median size |
| --- | ---: | ---: |
| f635 | 6 | 17.0 px |
| f2091 | 11 | 14.0 px |
| f2952 | 24 | 14.0 px |
| f1659 | 75 | 10.0 px (genuinely far — this is the second serve after a fault) |

Reference for the same clip: median ball size is 13 px at y 200–402 and 22 px at
y 600–900. So the 14–18 px detections above the net line are near-depth balls —
the toss — not balls across the net.

### Size-aware version

Discarding "far" detections too large for that depth:

| size cap | serves flagged | false positives |
| --- | ---: | ---: |
| none | 3/7 | 0/6 |
| ≤ 20 px | 3/7 | 0/6 |
| **≤ 16 px** | **4/7** | **0/6** |
| ≤ 14 px | 4/7 | 0/6 |
| ≤ 12 px | 5/7 | 1/6 |

A ≤16 px cap improves recall 3/7 → 4/7 at no cost in false positives. It does
**not** rescue the test: on game 2 it still flags exactly `f623, f3506, f4183`
and still misses `f2517`, which visual review confirmed is a serve.

### Standing

Precision remains the strong property — 0 false positives across both control
sets, and all three game-2 flags are confirmed serves. Recall is about half.
Treat it as a candidate surfacer, never a gate, and read the front-page
priorities as *incomplete*, not wrong.

See also `tennis11_game2_phantom_ball_track.md`: some tracked detections in game
2 are not balls at all. Game 1 looks far less affected (n=36 above y=200, median
12 px), which is why the control numbers here are still worth something.
