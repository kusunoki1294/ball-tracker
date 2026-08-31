# Game 2 Contact Verification Pass

Date: 2026-08-31
Author: Federer
Labels: `labels/tennis11_game2_contact_labels.csv`

## Scope

All 17 game-2 serve-motion contacts (9 accepted, 8 suppressed), labelled
serve / non-serve / ambiguous.

**Evidence source: every label came from a 5-frame sequence (offsets -15, -8, 0,
+6, +12) cut from `yoloVids/inputs/tennis11_game2.mp4` and cropped to the claimed
server's tracked box — not from the 4-frame contact strip alone.** A single
contact frame cannot separate a serve from an overhead; the swing shape can.

This is my visual reading of the source footage. It is stronger than the strip,
and it is still not the player's own confirmation.

## Result

| label | n | frames |
| --- | ---: | --- |
| serve | 10 | 183, 623, 819, 1595, 2116, 2517, 2990, 3506, 3987, 4183 |
| non-serve | 2 | 918, 1931 |
| ambiguous | 5 | 257, 408, 1795, 2629, 3154 |

### Four suppressed motions are serves

`f623, f2517, f3506, f4183` show the complete service action — toss, trophy
position, full extension, overhead contact, follow-through into the court —
and are near-indistinguishable from accepted serves in the same clip. `f4183`
and accepted `f3987` are the same motion 6.5 s apart.

These are suppression errors, not detection errors: the detector found them at
high confidence with source `ball_toss` and then discarded them as rally motion.

### Two suppressions are correct

`f918` takes a ball arriving from his left at shoulder height, from 5 ft behind
the baseline. `f1931` never raises the racket overhead.

### The heuristic from 554c5f8 missed one

The far-ball test flagged `f623, f3506, f4183` — all three confirmed serves, no
false positives. It did **not** flag `f2517`, which is a serve. That is a true
miss and matches the 3/7 recall measured on game 1: the test is precise and
insensitive, exactly as characterised.

## Pre-roll resolves the anomaly - and reverses the conclusion

I built the same pre-roll evidence (offsets -75 to +15, cropped to the far
player) for every contact attributed to the far end. **None of them is a serve.**

| frame | t | what the pre-roll shows |
| ---: | ---: | --- |
| 257 | 194.6 s | the near player serves at -75; the far player then moves laterally and contacts at waist height. A **return**. |
| 408 | 199.6 s | waist/chest contact while moving. Rally shot. |
| 2116 | 256.5 s | walks into position, contacts at waist height moving laterally. Rally shot. |
| 2629 | 273.6 s | lateral movement, waist-height contact. Rally shot. |
| 3154 | 291.1 s | largely static, waist-height contact, no service motion. Leaning rally shot. |

No toss, no trophy position, no overhead extension in any of them. Compare the
verified near serves, where all three are unmistakable.

So the earlier "knock-back" theory was also wrong. These are ordinary rally
shots, and `f257` in particular is the *return of the near player's serve* -
which also means the serve at `f183` was played, not the fault the landing
model called it.

## Revised result

| label | n | frames |
| --- | ---: | --- |
| serve | 9 | 183, 623, 819, 1595, 2517, 2990, 3506, 3987, 4183 |
| non-serve | 7 | 257, 408, 918, 1931, 2116, 2629, 3154 |
| ambiguous | 1 | 1795 |

**Every serve in game 2 is from the near end.** Four of the nine accepted
hypotheses (`257, 2116, 2629, 3154`) are false positives, and four real serves
(`623, 2517, 3506, 4183`) sit in the suppressed set.

## What this means for the boundary question

The mixed near/far split was the main evidence that 186-330 s spans more than
one game. That evidence does not survive: there are no far serves. Nine serves
from one end over 144 s is entirely consistent with **a single game**, roughly
the shape of game 1 (7 serves, 6 points).

I argued the opposite earlier, twice, and both times on weaker evidence than
this. The far-server hypotheses that drove it are misclassified rally shots.

This also bears on configuration: `single_server` was removed from game 2 partly
on my report that real far serves were being suppressed. That report was wrong -
the far motions are not serves. Whether to restore it is a call for the owner of
that config, on this evidence.

Caveat: the far player is roughly 85 px tall, so far-side labels here are
medium-to-low confidence individually. The pattern across all five is what
carries the argument, not any single one.

## Not done here

No suppression rule and no config changed. These are labels.
