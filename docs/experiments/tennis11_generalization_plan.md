# tennis11 generalization pass — scoping plan

2026-08-26. Scoping/decision document. No code changes. **The tracking run is
not approved yet** and must not be started without sign-off.

## Why

Everything validated so far comes from **game 1 only** — 2 minutes of a
28-minute set. The perception stack, the caller contracts, the thresholds and
the labels were all measured against that single game. We do not know whether
any of it generalizes, and tuning ball-track recall against one possibly
unrepresentative game risks overfitting. This pass closes that unknown before
any tracker/model investment.

## 1. Asset inventory

| kind | file | covers |
| --- | --- | --- |
| raw | `yoloVids/inputs/tennis11.MOV` | full 6-2 set, 1.9GB, 28m41s, 51,649 frames |
| cut | `yoloVids/inputs/tennis11_game1.mp4` | game 1, source 40–151s, 3,330 frames |
| cut | `yoloVids/inputs/tennis11_game1_clean.avi` | clean render base, same span |
| cut | `yoloVids/inputs/tennis11_input_340_460.mp4` | source 340–460s. Cut before game 1's location was known; **video only, never tracked** |
| log | `yoloVids/outputs/tennis11/ai11.1.jsonl` | **game 1 only**, 3,330 frames, 8MB |
| analysis | `ai11.2.analysis.json`, audit CSV/summary, `ai11.2.court_map.png`, `match_report.html`, `point_debug/`, `tennis11_game1_annotated.mp4` | game 1 |
| manifest | `manifests/tennis11_game1_manifest.json` | game 1 |
| labels | `labels/tennis11_game1_bounce_labels.csv` | game 1 |
| calibration | `yoloVids/calibration/court_calib_tennis11.json` | **the entire set** |

**The calibration is reusable for all 28 minutes.** The camera is static for the
whole recording, so no per-game recalibration is needed. It was verified against
two features the fit never used — projected net-post bases and baseline centre
marks.

## 2. What can be analyzed now vs what needs tracking

Analysable immediately: **game 1 only.** Everything downstream of
`ai11.1.jsonl` can be re-derived for free.

Requires tracking: **everything else**, including the existing 340–460s slice.
There is no tracking log for 26 of the 28 minutes, so generalization cannot be
tested at all without at least one new tracker run.

## 3. Smallest run that tests generalization: game 2

Game 2 was located by a 1fps visual scan rather than assumed. The ends change
after game 1, and the white-capped player (K, previously far) is on the **near**
side from ~186s with a serve motion visible at ~190s. **Game 2 starts ~186–190s.**
Its end is not established: there is no changeover after an even game, so the
ends cue does not work twice — it needs a point count or the user.

The test splits in two tiers because the first needs no human input.

### Tier A — detection generalization (runnable without ground truth)

Track ~186–330s and compare the perception layer against game 1:

- distinct ball observation rate (game 1 baseline: **56.8%**, not 62.8% — see
  `tennis11_gap_interpolation.md`)
- bounces found, and their confidence/near_player split
- serve motions found vs visually confirmed
- serve landings anchored within the flight window

This answers the actual open question. It also exercises something never tested:
**the ends have swapped**, so K is near and S is far, and every near/far
assumption in the stack is stressed for the first time.

### Tier B — scoring generalization (needs ground truth)

Requires `point_frames` and point winners from the user, exactly as supplied for
game 1. Without winners, auto-scoring cannot be scored and the manifest cannot
be completed honestly.

Recommendation: run Tier A first, and request game 2 winners in parallel so B
can follow without a second wait.

## 4. Runtime and storage

Game 2 ≈ 144s ≈ 4,300 frames. At the measured 1.2s/frame with OpenMP threads
set: **~85 minutes**, one-off, background.

Storage: JSONL ~10MB, clean base ~150MB. **Skip the annotated tracker AVI** —
`--output` is optional, and the analysis render now uses a clean base, so the
tracker's annotated video is dead weight. That saves ~2GB per game (game 1's is
1.5GB) and some write time.

```bash
# 1. clean base for rendering and any later labelling
ffmpeg -ss 186 -t 144 -i yoloVids/inputs/tennis11.MOV -map 0:v:0 -an \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 30 \
  yoloVids/inputs/tennis11_game2.mp4

# 2. track. Note: no --output, and the game 1 calibration is reused as-is.
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
.venv/bin/python track_ball_yolo.py \
  --video yoloVids/inputs/tennis11_game2.mp4 \
  --court-calib-file yoloVids/calibration/court_calib_tennis11.json \
  --court-calib-margin-px 300 --no-ball-court-filter --device cpu \
  --log-jsonl yoloVids/outputs/tennis11/ai11.g2.jsonl --headless

# 3. detection-level comparison; no manifest or winners needed
.venv/bin/python bounce_detect.py \
  --jsonl yoloVids/outputs/tennis11/ai11.g2.jsonl \
  --court-calib-file yoloVids/calibration/court_calib_tennis11.json
```

## 5. Repo changes this would need

**Genuinely needed to score a second game**

- `eval_bounce_detect.py` hardcodes `TENNIS9` and `TENNIS11` config dicts plus
  tennis11's serve contact frames. Needs parameterising, or a third entry.
- `validate_serve_detection.py` defaults to game 1's analysis path; a second
  clip needs the path as an argument.

**Probably not needed**

- Multi-game manifests. The analyzer already tracks games, changeovers and
  `side_to_player` across a point sequence, so one manifest spanning several
  games should work if `point_frames` and winners cover them. Untested — try
  that before adding structure.

**Ticket-level note, not an implementation task**

- `near_handedness` / `far_handedness` are fixed per SIDE in the manifest, but
  the ends swap at changeovers, so after game 1 they describe the wrong player.
  Both players in tennis11 are right-handed, so it cannot bite on this video. It
  will on any match where the two players differ. Record it; do not fix it as
  part of this pass.
