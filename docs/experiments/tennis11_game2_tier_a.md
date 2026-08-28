# tennis11 game 2 — Tier A generalization result

2026-08-27. Detection-level only; needed no ground truth for game 2. No
production code changed.

## Verdict

The perception stack **generalizes**. Serve detection, calibration reuse and the
bounce caller contracts all hold on a second game, with different players on
different ends.

What does **not** hold constant is input quality. Game 2 tracks materially worse
than game 1, which means **every threshold and label in this project was tuned
on the easier of the two games**. Recall work should be validated against game 2
from here, not game 1.

## Runtime

| | |
| --- | --- |
| planned / idle estimate | ~85 min |
| observed while contended | 3.52 s/frame at load avg 17.6, tracker ~150% CPU |
| after macOS scans ended | 1.87 s/frame |
| **actual wall clock** | **86 min for 4,321 frames = 1.19 s/frame mean** |

The early 4-hour projection did not materialize. Contention was transient and
the average landed on the idle benchmark.

Budget ~85 min/game for games 3-8, but expect up to 3-4h if the machine is
loaded, and check load before promising a window. A single 60-second spot rate
is not a forecast — that mistake produced an alarming ETA that was wrong by 3x.

## Track quality — the headline

| | game 1 | game 2 |
| --- | --- | --- |
| frames | 3,330 | 4,321 |
| with a ball | 2,090 (62.8%) | 2,351 (54.4%) |
| coasted / held | 222 | 245 |
| exact position repeats | 199 | 177 |
| **distinct real observations** | **1,891 (56.8%)** | **2,174 (50.3%)** |

Game 2 tracks **6.5 points worse**.

## Fragmentation

Game 1 breaks into **7** activity spans; game 2 into **16**, for a comparable
length of play. Worse tracking splits points mid-rally, which directly degrades
automatic point segmentation — the mechanism by which `cut_play_segments.py`
would find points without human input.

## Bounces

| | game 1 | game 2 |
| --- | --- | --- |
| bounces | 44 (23.8/min) | 45 (18.7/min) |
| rally_scoring_eligible | 12 | 19 |
| high shape confidence | 9 | 15 |
| side split | near 29 / far 15 | near 27 / far 18 |

Per-minute rate is down 21%, tracking the observation drop. But rally-eligible
detections nearly doubled and high-confidence rose. Game 2 yields **fewer but
cleaner** detections: the contracts are not degrading, the input is.

## Serve detection — the strongest result

| | game 1 | game 2 |
| --- | --- | --- |
| motions detected | 9 | 12 |
| vs known ground truth | **7/7 matched**, 2 false positives (f786, f2491) | no ground truth |
| landings anchored | 7/9 (78%) | 9/12 (75%) |

Four of game 2's twelve contacts (f183, f1595, f2990, f3987) were checked
against video: all four are genuine serves, near player at full extension behind
the baseline. Anchoring rate is essentially unchanged across games.

## Calibration reuse — confirmed

Game 1's calibration was accepted on game 2 with no warning in the run log, so
the plausibility and frame-margin checks both passed. The static-camera claim
holds, and games 3-8 therefore cost tracker time only, with no calibration work.

## Near/far inversion — real, but narrower than first claimed

The ends **did** change: game 1's near player is navy-shirted and bare-headed
(S); game 2's is black-shirted in a white cap (K), confirmed from side-by-side
crops. Player identity is inverted and the stack handled it.

The serving **side** is not inverted. Service alternates, so K serves game 2
from the near side exactly as S served game 1 from the near side. **A far-side
server has still never been exercised**; that requires game 3.

## Hazards

| | game 1 | game 2 |
| --- | --- | --- |
| contact-adjacent detections (<10 frames after a serve strike) | 3 | 1 |
| detections within 2ft of the net line | 2 | 3 |

Both in the same range across games. The net-line cases are handled at the
consumer by `NET_LINE_CONTACT_BAND_FT`.

## Method correction — `detect_serve_motions` needs per-point ranges

The first analysis pass called `detect_serve_motions` with a **single range
spanning the whole clip** and reported 1 motion for game 1. That function is
per-point by design and does not enumerate serves across an arbitrary span.

Re-running with derived activity spans as point ranges gives 9 motions for game
1 and recovers all 7 known contacts. Anyone measuring serve recall on a clip
without a manifest must segment first.

This was caught only because game 1's known answer disagreed with the output.
**Carry a clip with ground truth alongside any new one**; without game 1 in the
same run, "1 serve motion detected" would have been reported as a generalization
failure.

## Open

Whether 12 serve motions is correct for game 2 is unknown without its point
count. Tier B (scoring) still needs point winners.
