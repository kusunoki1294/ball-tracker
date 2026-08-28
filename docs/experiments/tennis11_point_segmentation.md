# Automatic point segmentation from tracking logs — experiment

2026-08-28. Experiment; no production code changed.

## Question

The project goal moved to fully automated analysis: no hand-entered "game 1 is
this slice" or "point 1 winner is X". That requires `point_frames` to come from
the tracking log rather than a human. Can it?

## Approach

Points are anchored on **serve motions**, because a point begins with a serve.
Everything else is supporting evidence:

1. serve motions from `serve_detect`, both sides, over ball-activity spans
2. a landing bounce shortly after the strike, corroborating a real serve
3. ball activity spans, to close a point's end
4. a change of serving side, marking a game boundary

No manifest, no point winners, no human input.

## The three rules that did the real work

Two are plain tennis, and without them the method fails outright.

**A point has at most two serves.** Without this, six points chained into three.
Consecutive points routinely start inside the 15s second-serve window, so time
alone cannot separate a fault from the next point's first serve.

**A second serve only follows a fault.** If the previous serve landed in the
box, the point was live, so a later strike cannot be its second serve. This is
what stops a mid-rally false positive being quietly absorbed as a phantom second
serve rather than being exposed as a spurious detection.

**Rally evidence between two strikes means the earlier point finished.** The
same test the analyzer already applies when deciding whether a second serve is
genuine.

## Game 1 result — against the 7 verified serve contacts

| configuration | proposals | point-start recall | precision |
| --- | --- | --- | --- |
| require a corroborating landing | 7 | 5/6 | 5/7 |
| no requirement | 9 | **6/6** | 6/9 |

Requiring a landing costs point 5, whose landing falls in the known 11-frame
track hole. So the trade is 6/6 recall at 67% precision, or 5/6 at 71%.

False positives at f786, f941, f2491, f2867 — mid-point tosses and reaches that
resemble a serve, two of them with a bounce shortly after.

## Game 2 — over-segmentation

13–14 proposals. Game 1, a six-point game, produced 7–9. Game 2 has **16**
activity spans against game 1's **7**, and the over-segmentation tracks that
fragmentation directly: a broken ball track splits one point into several, and
each fragment can attract a spurious serve candidate.

Game 2's true point count is unknown, so no precision figure is claimed. But
13–14 points in a single game is implausible on its face.

## Caveat on the range-agreement number

Mean IoU against the game-1 manifest ranges is 0.53. **Do not read that as 47%
error, and do not quote it as accuracy.**

- Proposals start late and end early *by construction* — anchored at contact
  minus 45 frames and closed at the last activity — whereas the manifest ranges
  were padded to the midpoints between points. Much of the gap is convention.
- More importantly, **the manifest ranges are not independent ground truth.** I
  built them myself from activity spans; the user supplied only the point count
  and the winners. The serve *contacts* are verified; the *boundaries* are not.
- There is therefore **no independent ground truth for point ends at all**. This
  experiment can report agreement, not accuracy, for ends.

## Product answer: no

Hard `point_frames` cannot be auto-generated reliably enough to feed
`analyze_tennis_events` as though they were ground truth, and more tuning is not
expected to close it.

The reason is the failure mode rather than the headline rate. **One in three
proposed points is spurious on the better-tracked game.** A spurious point does
not degrade the output gracefully — it invents a point that never happened and
shifts every subsequent game score. That is silent corruption, and it is worse
than an obvious failure because nothing downstream can detect it.

**Point ranges must be uncertain hypotheses carrying confidence, with the
ability to abstain.** This is the same move that made point-winner inference
trustworthy: the system improved when it started abstaining instead of
asserting, going from 2 correct / 4 wrong to 3 correct / 0 wrong / 3 abstained.

## Confidence fields the evidence supports

All are already computed by the experiment:

| field | meaning |
| --- | --- |
| `serve_corroborated` | a landing bounce exists in the flight window |
| `landing_in_box` | the landing was inside the receiver's service box |
| `serve_count` | 1 or 2, and whether the second followed a fault |
| `isolated_by_deadtime` | dead frames precede the strike, i.e. a real point opening |
| `local_fragmentation` | activity spans per minute nearby — the best single predictor of over-splitting, and what separates game 1 from game 2 |
| `ends_have_no_truth` | flags that the point END is inferred, never observed |

## Recommendation for the next production step

Build a **`timeline_hypotheses` layer** that emits candidate point starts and
ranges with confidence and reasons.

It must **not** be a manifest replacement that pretends certainty. The manifest
format asserts that a point runs from frame A to frame B; this evidence cannot
support that assertion, and dressing hypotheses in manifest clothing would hide
exactly the uncertainty that matters. The layer should:

- emit candidates with the confidence fields above and a human-readable reason
- allow several competing hypotheses over the same span rather than forcing one
- let the analyzer consume high-confidence candidates, and abstain on the rest
  instead of scoring them
- keep a hand-authored manifest as an override for known-good clips, not as the
  normal path

## Reproducing

The harness lives outside the repo, since it produced no shipped code. Inputs
are `yoloVids/outputs/tennis11/ai11.1.jsonl` and `ai11.g2.jsonl`, the shared
calibration, and the verified serve contacts recorded in
`tennis11_game2_tier_a.md`.
