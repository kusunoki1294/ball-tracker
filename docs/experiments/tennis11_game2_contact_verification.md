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

## The anomaly that blocks a boundary conclusion

Three times, a near serve is followed within a few seconds by an accepted **far**
serve:

| near serve | far serve | gap |
| --- | --- | ---: |
| f183 192.1 s (fault) | f257 194.6 s | 2.5 s |
| f2517 269.9 s (fault) | f2629 273.6 s | 3.7 s |
| f2990 285.7 s (fault) | f3154 291.1 s | 5.5 s |

Two players cannot serve from opposite ends 2.5 s apart. In the f183/f257 case
the near player is standing mid-baseline with his racket down through the whole
window — he is not receiving. The most likely reading is that these "far serves"
are the receiver knocking back a faulted serve, and that the far-server
hypotheses are partly an artifact of that.

**This revises my earlier claim.** I previously argued the mixed near/far serve
pattern made it unlikely that 186–330 s is a single game. If the far serves are
knock-backs, that argument does not hold, and the near/far split is not evidence
of a game change. I do not have enough to settle it either way.

What survives: at least 10 serve motions in the slice, 4 of them wrongly
suppressed. The boundary question stays open and still needs the player.

## Not done here

No suppression rule was changed. These labels are evidence for a future decision,
not a decision.
