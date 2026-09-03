# Why tennis9 Misses f1147 And f1401

Date: 2026-09-02
Author: Federer
Diagnosis only. No thresholds changed.

## Verdict

Neither miss is a bounce-detector threshold problem, and neither is a label
problem. **Both are the same tracker mis-association**, and the bounce detector
is behaving correctly on corrupted input.

The failure runs in three steps:

1. the tracker accepts a large jump to a different object, scoring it positively
2. the motion prediction is then poisoned, extrapolating away from the real ball
3. real detections are rejected as `far_jump_rejected` against that prediction

## f1147 — the poisoned prediction opens a hole

| frame | tracked centre | note |
| --- | --- | --- |
| 1149 | (948, 336) | real ball |
| 1150 | (883, 303) | **72 px jump accepted**, `reason: tracked_ai`, score 20.2, against a prediction of (947, 325) |
| 1151+ | none | prediction now (818, 270); candidates at (948, 317) and (883, 304) are `far_jump_rejected` |

The result is a 13-frame hole starting immediately after the bounce. The
detector then has 3 samples on the after-arc where it requires 4, so the
candidate at f1147 is dropped for `AFTER has 3 samples, needs 4`.

The ball is plainly visible in frames f1152–f1158 — I checked the pixels. It is
not a detection failure; the detections exist and are being rejected.

## f1401 — the same jump, poisoning a residual instead of a hole

| frame | tracked centre | size | note |
| --- | --- | ---: | --- |
| 1404 | (944, 334) | 9 px | prediction (944, 333), score 195.1 |
| 1405 | (968, 260) | **21 px** | **70 px jump accepted**, score 39.0 |
| 1406 | none | | prediction now (992, 186); real candidate (967, 263) rejected |

Here the bad sample stays inside the fit window rather than opening a hole, and
poisons the after-arc:

| after-arc at f1401 | samples | residual |
| --- | ---: | ---: |
| as tracked | 6 | **21.5 px** — rejected, `max_residual_px` is 18 |
| with f1405 removed | 5 | **0.3 px** |

One bad sample is the entire difference between a clean bounce and no bounce.

## Why this must not be fixed by tuning

Raising `max_residual_px` past 21.5 would recover f1401 and admit genuinely bad
fits everywhere else — the honest 0.3 px fit is what the data supports, not a
21.5 px one. Lowering `min_samples` to 3 would recover f1147 by fitting a
quadratic to the minimum possible number of points, which is a degeneracy, not
evidence. Both rejections are the detector correctly declining to assert a
bounce from corrupted input.

## Where a fix would belong

Upstream, in the tracker's association step. One concrete signal is already
visible in the evidence: at f1405 the accepted candidate is **21 px where its
neighbours are 9–10 px**. A jump that also changes apparent size sharply is far
more likely to be a different object than the same ball moving fast. Size
inconsistency is the same cue that settled the tennis11 phantom question, and it
does not require knowing depth.

Not attempted. Recorded so the next person does not start by reaching for the
thresholds.
