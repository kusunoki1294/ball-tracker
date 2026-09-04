# Shadow As A Bounce-Ranking Signal: Measured And Rejected

Date: 2026-09-04
Author: Federer
Evidence only. Nothing built, nothing committed to the detector.

## Question

`f1446` is lost because non-maximum suppression ranks a racket contact above a
ground bounce 11 frames earlier, and score cannot tell them apart — both are
equally clean reversals. A ball touching the court meets its own shadow and a
ball on a racket does not, so can shadow proximity rank them?

Measured as the distance from the ball to the nearest ball-sized dark blob, in
ball diameters. No resolvable blob means **abstain**.

## It works on the pair it was built for

| frame | what it is | gap |
| --- | --- | ---: |
| tennis9 f1445 | ground bounce (the suppressed one) | 1.18 |
| tennis9 f1446 | same bounce, next frame | 1.18 |
| tennis9 f1455 | racket approach | **abstain** |
| tennis9 f1456 | racket contact (the one kept) | 2.36 |
| tennis11 f2251 | descending, still airborne | 2.95 |
| tennis11 f2253 | hidden ground bounce | 1.43 |

Bounces cluster at 1.2–1.4, non-ground at 2.4–3.0. Exactly the hoped-for story.

## It does not survive the labelled population

All 44 hand-labelled tennis11 game-1 cases:

| label | n | abstain | resolved | median gap | range |
| --- | ---: | ---: | ---: | ---: | --- |
| live_bounce | 10 | 1 | 9 | 0.96 | 0.40–5.19 |
| dead_bounce | 9 | 5 | 4 | 0.88 | 0.76–4.64 |
| racket | 14 | 8 | 6 | 1.17 | 0.27–3.32 |
| tracking_artifact | 2 | 0 | 2 | 1.83 | 1.55–2.12 |
| ambiguous | 4 | 0 | 4 | 1.91 | 0.38–3.40 |

`live_bounce` 0.96 against `racket` 1.17, ranges overlapping almost entirely.
**No separation.**

## The abstain rate disqualifies it independently

| region | n | abstain |
| --- | ---: | ---: |
| near court (ball ≥ 12 px) | 29 | 12 (41%) |
| far court (ball < 12 px) | 10 | 2 (20%) |

A ranking input that declines on 41% of near-court cases cannot rank. It already
abstains on `f1455`, one of the four frames in the pair it was built for.

## Far court does not abstain — it resolves, confidently and wrongly

The abstain rates look backwards: 20% far court against 41% near. An earlier doc
concluded far court has nothing to measure, so it should abstain *more*. Both
cannot be true.

Inspecting the 8 resolved far-court cases settles it:

| frame | ball | gap | blob | blob/ball area | label |
| --- | ---: | ---: | ---: | ---: | --- |
| f186 | 11 px | 0.62 | 27 | 0.29 | live_bounce |
| f345 | 11 px | 0.33 | 22 | 0.23 | **racket** |
| f1013 | 10 px | 0.43 | 14 | 0.18 | ambiguous |
| f1104 | 11 px | 0.38 | 10 | 0.11 | ambiguous |
| f1511 | 8 px | 1.84 | 18 | 0.36 | live_bounce |
| f2114 | 11 px | 0.66 | 52 | 0.55 | live_bounce |
| f2976 | 10 px | 0.40 | 14 | 0.18 | live_bounce |
| f3166 | 10 px | 2.12 | 121 | 1.55 | **tracking_artifact** |

Gaps cluster at 0.33–0.66, *tighter and smaller* than any genuine near-court
bounce (1.18–1.43). A gap of 0.33 ball diameters means the "shadow" is sitting on
top of the ball — that is the detector latching onto the ball's own dark edge, a
court line, or net shadow, not a separated ground shadow. A real shadow under an
8–11 px ball would be a few pixels and not separable at all.

**This is worse than missing information.** Under any threshold calibrated on the
near-court pair, every far-court case reads as a ground contact — including the
racket contact at `f345` and the tracking artifact at `f3166`. The signal is not
silent there; it is confidently wrong in a single direction.

Hence the constraint below is a hard not-applicable by court region, not a
softer "absence must not penalise".

## The threshold was tuned in the signal's own favour

The first implementation used 18% of local median as the darkness threshold and
found nothing anywhere. Inspecting `f2253` showed the shadow is only ~15 grey
levels below the court and covers ~40–65 px against a 153 px ball footprint — a
shadow is smaller than the ball on screen because it lies flat.

The threshold was then set from that frame, which is one of the six pair cases.
**That makes the negative result stronger, not weaker**: tuned in its favour on
its own best case, it still does not generalise.

## Decision

**Do not build shadow ranking or gating without new labelled near-court
ground-contact data.** Nine resolved live bounces is too thin to conclude in
either direction, and the current evidence is against.

If it is ever revisited, the constraints from review stand:

- a **separate shadow field only** — it must not feed `detector_confidence` or
  `detector_shape_confidence`
- no shadow means **not applicable / abstain**, never weak evidence
- **hard not-applicable by court region.** Far court must be excluded by region,
  not merely un-penalised: it resolves on lines and net shadow at gaps smaller
  than genuine bounces, so it reads every far-court event as a ground contact
- **net-line contacts remain governed by `NET_LINE_CONTACT_BAND_FT` geometry**;
  shadow must not reclassify them

## "Ranking only" is not automatically safe

Raised in review: NMS survivor selection happens inside the 10–45 frame serve
flight window, so a shadow-driven re-rank could change which candidate becomes a
serve landing — and therefore change an in/fault verdict. Any such change would
require re-verifying `f186` in, `f1511` fault, `f2114` fault, `f2976` in, plus
the net-line rejections.

Measured, because the scope matters:

| landing | verdict | rivals within the 18-frame window |
| --- | --- | ---: |
| f186 | in | 0 |
| f1511 | fault | 0 |
| f2114 | fault | 0 |
| f2976 | in | **1** (f2974, medium, score 2.20, ranks below today) |

So the principle is right and the current exposure is **one landing with one
rival**, not four. Three of the four serve landings are uncontested inside the
suppression window and no re-ranking could displace them on this clip.

That is still enough to make the point: a change sold as "ranking only" reaches
serve adjudication, so it needs the landing verdicts re-verified rather than
assumed. It is simply a narrower re-verification than four.

## Why this is recorded rather than dropped

This is the third signal measured and rejected for the bounce problem, after the
racket cue and the near-player veto relaxation. The population check ran *before*
the report this time rather than after, which is what two earlier retractions
cost. Six hand-picked frames separated cleanly and 44 labelled ones did not — the
same shape both retractions had.
