# Player-Box Association Stability — And A Correction To The Point-Ended Result

Date: 2026-09-04
Author: Federer
Audit only. Contains a retraction of my own `c46e394` finding.

## 1. The swaps are real and concentrated on the far player

A "swap" here is a frame-to-frame box-centre move of more than one box height.
A person cannot travel a body length in 1/30 s, so every one is a detection
change, not motion.

| clip | player | frames | swaps | rate |
| --- | --- | ---: | ---: | ---: |
| game1 | near | 3190 | 14 | 0.4% |
| game1 | **far** | 3330 | **231** | **6.9%** |
| game2 | near | 3940 | 4 | 0.1% |
| game2 | far | 4244 | 43 | 1.0% |

Signature, as seen at `g1 f1485`: the box alternates between a stable
high-confidence one and a spurious low-confidence one. Median size ratio at a
swap is 1.82 (game1) and 4.18 (game2); median confidence of the weaker box is
0.26–0.33.

## 2. Decision impact is small and localised

Far-player swaps inside the 1 s window before a verified serve contact:

- **2 of 16 serves affected**: `g1 f1485` (9 swaps) and `g1 f1659` (2)
- both belong to the same point — game 1, point 3
- all 14 other serves have zero

Swaps cluster during rallies, when the far player is moving, and are absent when
they stand still to receive. So the direct damage to serve/timeline decisions is
confined to one point in one clip.

## 3. Retraction: the receiver-motion signal is a swap detector

`c46e394` reported receiver travel separating live rallies from point starts by
~85x, and called it "receiver stillness". **That description is wrong.**
Decomposing the same windows:

| case | total | median step | swaps | total from swaps | truth |
| --- | ---: | ---: | ---: | ---: | --- |
| f786 | 42.41 | 0.006 | 5 | **42.26** | rally live |
| f1040 | 36.09 | 0.030 | 5 | **35.40** | rally live |
| f1079 | 80.35 | 0.011 | 8 | **80.10** | rally live |
| f159 | 0.25 | 0.006 | 0 | 0.00 | serve |
| f2952 | 0.14 | 0.006 | 0 | 0.00 | serve |

**99% of the metric's value comes from swap steps.** Median per-frame movement
is essentially identical between rally and serve windows (0.006 vs 0.006). The
metric counts detector instability; it does not measure whether the receiver is
moving.

It separated the three controls because those three windows happen to be
swap-heavy, not because a rally was live.

### Neither replacement survives a broad test

- **Travel with low-confidence boxes filtered out**: the rally controls collapse
  to 0.11–1.09, overlapping the serves at 0.04–0.29.
- **Mean far-player confidence**: separates the three controls (0.41–0.51 versus
  0.66–0.83) but not in general — 62% of 300 random non-serve frames exceed 0.65,
  and two verified serves fall below it (`f635` 0.30, `f1485` 0.38).

## 4. What this means for the point-ended work

The receiver-motion veto is **not supported by evidence beyond the three
controls**, and the README line and the implementation guardrails built on it
rest on a metric that is measuring the wrong thing. It should not be implemented
as described.

What survives: a *swap-heavy* window is genuine evidence that something is wrong
with the boxes there, which is a reason to **abstain**, not to conclude a rally
is live. That is a weaker and differently-shaped claim than the one I made.

The lesson is the one this project keeps re-teaching: three hand-picked controls
can separate on a coincidence. The generalisation test is what tells you whether
a signal exists, and it has now overturned two of my results in two days.
