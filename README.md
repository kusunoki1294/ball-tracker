Ball Tracker

This project started as an OpenCV tennis-ball tracker and now also includes a separate YOLO-only pipeline for tracking the ball, players, and other court objects.

Current status
- `track_ball.py` is the original hybrid tracker. It uses HSV/color, motion, optional court logic, and optional YOLO ball detection.
- `track_ball_yolo.py` is the newer YOLO-only tracker. This is the main experimental path for the tennis project right now.
- The YOLO-only tracker works reasonably well for:
  - ball tracking
  - near/far player labeling
  - other object boxes such as rackets and cars
- The tennis9 post-analysis pipeline now adds:
  - serve-state diagnostics
  - point-ending diagnostics
  - point/game/set scoring
  - audit CSV/summary exports
  - per-point debug court-map PNGs
  - static HTML match report
  - red-mark and clean overlay renders
- The YOLO-only analysis path is still experimental for:
  - fully automated point timeline hypotheses
  - far-side serve detection
  - separating true in-play bounces from other real ball events without caller context
- Second worked example (2026-08): tennis11 game 1
  - manifest: `manifests/tennis11_game1_manifest.json`
  - input slice: `yoloVids/inputs/tennis11_game1.mp4` (source 40s-151s of a full
    6-2 set; game 1 runs from the first serve at ~41s to ~2:30)
  - calibration: `yoloVids/calibration/court_calib_tennis11.json` (automatic,
    line-model fit, 0.4px)
  - tracking: `yoloVids/outputs/tennis11/ai11.1.{avi,jsonl}`
  - annotated render: `yoloVids/outputs/tennis11/tennis11_game1_annotated.mp4`
  - automated timeline demo: `yoloVids/outputs/tennis11/timeline/timeline_demo.html`
  - hypothesis audit: `yoloVids/outputs/tennis11/timeline/timeline_audit.html`
  - hypothesis review videos:
    `yoloVids/outputs/tennis11/timeline/game1_timeline_hypotheses.mp4`,
    `yoloVids/outputs/tennis11/timeline/game2_timeline_hypotheses.mp4`
  - Scoring reproduces the real game exactly (0-15, 15-15, 15-30, 30-30, 40-30,
    game) from manual `point_winners`. Offline bounce detection and serve-motion
    detection are now integrated for this clip; automatic winner inference is
    improved but still abstains where the ball track drops critical events. The
    fully automated timeline path is deliberately hypothesis-only and does not
    feed scoring.

Current known-good analysis target:
  - manifest: `manifests/tennis9_analysis_manifest.json`
  - analysis JSON: `yoloVids/outputs/tennis9/play_segments/ai9.5.analysis.json`
  - red-mark render: `yoloVids/outputs/tennis9/play_segments/ai9.5.avi`
  - clean render: `yoloVids/outputs/tennis9/play_segments/ai9.6.avi`

Making it work on any video (2026 update)
The goal of this round of work was to make the system work on arbitrary
behind-the-baseline footage, not just the tennis9 clip. A second test video was
added (tennis10: a SwingVision singles match with the same camera angle,
`yoloVids/inputs/tennis10_input.mp4`, downloaded and normalized to 1080p/30fps).
Testing on it surfaced and fixed the main things that made the system brittle
across videos.

1. The stale-calibration footgun (fixed)
   - `track_ball_yolo.py` defaults `--court-calib-file` to `court_calib.json`,
     and only checked that the points fell inside the frame. A stale ~1280x720
     tennis6/7 calibration therefore passed that check on 1080p footage while
     mapping the court to the wrong place, so the court filter silently rejected
     most real ball detections.
   - Effect on tennis10: in-rally ball recall was only 41% because ~44% of real
     ball detections were being thrown away by the court filter (the ball model
     itself detects the ball in ~94% of rally frames).
   - Fix: `calibration_plausible_for_frame()` rejects a calibration that clearly
     does not match this video (court must span a large vertical slice of the
     frame, near baseline low in the frame, near baseline wider than far). On a
     mismatch the tracker warns and runs WITHOUT court filtering instead of
     silently degrading. tennis10 recall went 41% -> 78% with no override.
   - Net result: ball, player, and object tracking now work out of the box on
     any behind-baseline clip, with no per-video setup.

2. Automatic court calibration (new: court_ai/)
   - Precise court geometry is still needed for bounce localization, the mini
     court overlay, side classification, and scoring. Hand calibration does not
     scale to "any video", and classical line detection could not reliably find
     the far baseline (faint and occluded by the net).
   - `court_ai/` trains a small CNN (CourtNet) to regress the 4 doubles-court
     corners. It is trained on 100% synthetic data (the court geometry rendered
     through random behind-baseline homographies with net bands, player
     occlusions, and clutter), so no manual labeling is needed.
   - Sim2real transfer only worked after switching the model input to a
     domain-invariant "top-hat line map" (thin bright lines isolated regardless
     of court/background brightness). With RGB input the model failed on real
     courts (417px corner error); with the line map it produces accurate
     calibrations that visually trace the real tennis9 and tennis10 courts
     (balls project to world coords centered on the court, median x=17.4 with
     court center at 18).
   - See `court_ai/README.md` for the pipeline and reproduction steps.

3. Key finding: an accurate calibration LOWERS raw ball recall
   - Feeding the accurate model calibration to the tracker dropped raw recall to
     64.7% vs 81% with no court filter. This is NOT a calibration error: a ball
     in the air projects through the ground homography to a world point outside
     the court (measured world y down to -103, well past the far baseline), so
     the court filter rejects it.
   - Fixed (2026-08): `--no-ball-court-filter` decouples the two uses. Ball
     candidates are no longer gated on the court bounds (recall stays high, as
     the guardrail already does when no calibration is present) while the
     calibration still drives bounce localization, the mini court, side
     classification, and scoring. Use it whenever you pass a calibration.
   - Confirmed on tennis11 over the same 350 rally frames, with an accurate
     calibration supplied either way: 73.1% ball recall with the court filter,
     82.9% without it. The filter discarded 219 real candidates in those frames.

4. Calibration accuracy: CourtNet alone is an initialization, not a calibration
   - On tennis11 the raw CourtNet corners were ~8px off on the interior lines and
     ~25px off on the far baseline, which is metres of world error at the far end
     - exactly where bounce in/out calls matter.
   - Snapping only the four outer edges to the nearest bright line made it WORSE
     (27px out on the far service line). The far baseline is short and faint with
     the net tape and the court/apron edge running parallel to it, so a snapped
     edge slides onto a neighbour and still looks self-consistent.
   - `court_ai/fit_lines.py` fits the WHOLE court line model instead (baselines,
     doubles and singles sidelines, both service lines, centre service line).
     No wrong homography can light all of those up at once. Mean offset went
     0.4px, verified against two features the objective never sees: the projected
     net-post bases and the baseline centre marks both land on the real ones.
   - This runs by default in `court_ai/infer.py` (disable with `--no-refine`).

5. Cameras that crop the near doubles corners
   - A correct calibration can place the near corners outside the frame (tennis11:
     x=-69 and x=2185 on a 1920-wide frame). `calibration_fits_frame`'s ±24px
     default rejected that correct calibration, so `--court-calib-margin-px` now
     sets the tolerance. `court_ai/infer.py` prints the exact value needed.

Files
- `track_ball.py`: original tracker with HSV/motion and optional YOLO support
- `track_ball_yolo.py`: YOLO-only tracker for ball, players, other objects, experimental bounce markers, and a mini court overlay with mapped ball/bounce points. Now rejects a mismatched court calibration (`calibration_plausible_for_frame`) instead of silently degrading.
- `court_ai/`: automatic court-calibration model. Trains a CNN on synthetic courts to predict the 4 corners from a real frame; writes a `track_ball_yolo` calibration JSON. See `court_ai/README.md`.
- `court_calib.json`: saved court calibration data (legacy 1280x720; do not rely on it for 1080p footage)
- `vids/`: older sample videos, model files, helper files
- `yoloVids/`: newer YOLO workflow videos and renders (gitignored). Inputs live in `yoloVids/inputs/`, calibrations in `yoloVids/calibration/`.
- `yolo.txt`: YOLO reference links

Requirements
- Python 3
- OpenCV (`opencv-python`)
- `ultralytics` for the YOLO-only tracker
- Optional: `ffmpeg` and `ffprobe` for the original script's video fallback paths

Install
1. Create and activate a virtual environment if desired.
2. Install the base dependencies:

   pip install -r vids/requirements.txt

3. Install YOLO support:

   pip install ultralytics

Models
- Ball model:

  `vids/models/tennisball.pt`

- Scene model:

  default is `yolov8n.pt`

Notes
- `yolov8n.pt` may download automatically the first time you run the YOLO-only tracker.
- The project was tested on iPhone footage shot at `0.8x` zoom, which makes the far-court ball much smaller and harder to detect.

Original tracker usage
Basic run:

python track_ball.py --video vids/tennis.MOV

Save annotated output:

python track_ball.py --video vids/tennis.MOV --output output.mp4

Use a preset:

python track_ball.py --video vids/annotated2/annotated2.MOV --preset annotated2

Run with YOLO ball detection:

python track_ball.py --video vids/tennis.MOV --yolo-model vids/models/tennisball.pt

YOLO-only tracker usage
Basic run:

python track_ball_yolo.py --video yoloVids/tennis6.MOV --output yoloVids/annotated6.avi --headless

Tennis analysis pipeline
Run the tracked tennis9 analysis, audit exports, and configured overlay renders:

python run_tennis_pipeline.py --manifest manifests/tennis9_analysis_manifest.json

This writes per-point debug PNGs to `yoloVids/outputs/tennis9/play_segments/point_debug/`.
Magenta `B?` markers are conservative trajectory-based missed-bounce candidates for review; they are not counted as live scoring bounces.
It also writes a browser-readable report to `yoloVids/outputs/tennis9/play_segments/match_report.html` and compact structured data to `match_report_data.json`.

After regenerating tennis9 outputs, validate the known-good scoring and point-ending behavior:

python validate_tennis9_regression.py

Use a custom scene model:

python track_ball_yolo.py --video yoloVids/tennis6.MOV --scene-model /path/to/model.pt --output yoloVids/annotated6.avi --headless

Write JSONL logs:

python track_ball_yolo.py --video yoloVids/tennis6.MOV --output yoloVids/annotated6.avi --log-jsonl yoloVids/annotated6.jsonl --headless

Mac output note
- On this machine, writing `.mp4` output was unreliable.
- Writing `.avi` output worked.
- In practice, use `.avi` for the YOLO-only tracker unless video writing has been fixed later.

What was added to `track_ball_yolo.py`
1. Two-model YOLO pipeline
   - one YOLO model for the tennis ball
   - one YOLO model for players and other scene objects

2. YOLO-only tracking
   - removed dependence on HSV/motion logic from this script
   - uses a simple in-script tracker instead of Ultralytics `track()` because `lap` was missing in the local environment

3. Player labeling
   - labels the higher player in frame as `Player Far`
   - labels the lower player in frame as `Player Near`

4. Ball trail
   - draws a yellow trail behind the tracked ball

5. Stationary-ball filtering
   - suppresses detections that look like a ball but stay fixed
   - added because the tracker was locking onto things like a speck on the net or a ball left on the ground

6. Moving-ball filtering
   - requires a ball track to show actual travel before it is treated as the active ball

7. Far-court ball pass
   - runs a second higher-resolution YOLO pass over the top portion of the frame
   - added because the iPhone `0.8x` footage makes the far-court ball very small

8. Physical jump rejection
   - rejects active ball candidates that move impossibly far in one frame
   - rejects candidates that are too far from the predicted next ball position
   - rejects huge near/far court side flips
   - tries the next plausible candidate before dropping the ball

9. Experimental bounce markers
   - current event work is bounce-only for now
   - hit markers were removed temporarily because they were too noisy

10. Side-aware bounce filtering
   - uses `court_calib.json` when available
   - tries to treat near-side and far-side bounce candidates differently
   - rejects candidates that project outside the singles court

11. Mini singles-court overlay
   - draws a small singles court in the top-right corner
   - plots the current ball position and recent bounce markers when calibration is available

Important YOLO-only options
Main detection options
- `--court-calib-file`: calibration file for side-aware bounce rules
- `--no-ball-court-filter`: do not reject ball candidates that project outside
  the court. Recommended whenever a calibration is supplied: an in-flight ball
  projects through the ground homography to a world point past the baselines, so
  the court filter drops real detections. The calibration is still used for
  bounces, the mini court, side classification, and scoring.
- `--court-calib-margin-px`: how far outside the frame a calibration point may
  fall before the calibration is rejected (default 24). Cameras that crop the
  near doubles corners need a larger value; `court_ai/infer.py` prints it.
- `--ball-model`: ball detector path
- `--scene-model`: scene detector path
- `--ball-conf`: confidence threshold for the ball detector
- `--scene-conf`: confidence threshold for the scene detector
- `--imgsz`: main inference size
- `--device`: YOLO device, such as `cpu`, `mps`, or `0`

Tracking options
- `--ball-max-distance`: max frame-to-frame match distance for ball IDs
- `--ball-max-jump-px`: reject active ball candidates that move too far from the last accepted ball in one frame
- `--ball-max-prediction-error-px`: reject active ball candidates that are too far from the predicted next ball position
- `--ball-max-side-flip-jump-px`: reject impossible near/far court side flips
- `--object-max-distance`: max frame-to-frame match distance for player/object IDs
- `--trail`: ball trail length

Stationary/moving ball filters
- `--ball-stationary-px`
- `--ball-stationary-frames`
- `--ball-motion-history`
- `--ball-min-travel`

Far-court ball options
- `--far-ball-roi-height`
- `--far-ball-roi-width`
- `--far-ball-conf`
- `--far-ball-imgsz`

Experimental bounce options
- `--bounce-min-vertical-change`
- `--bounce-min-gap-frames`
- `--bounce-x-margin-ratio`
- `--bounce-y-margin-ratio`
- `--bounce-min-y-ratio`
- `--player-hit-margin-px`
- `--racket-hit-margin-px`
- `--event-min-travel`
- `--court-calib-file`

Unused legacy tuning options still present in the script
- `--player-hit-upper-body-ratio`
- `--hit-min-gap-frames`
- `--hit-min-angle-change-deg`
- `--hit-min-speed-change-ratio`

Display/output options
- `--hide-other-objects`
- `--headless`
- `--log-jsonl`
- `--no-court-overlay`
- `--court-overlay-size`
- `--court-overlay-margin`

Known issues
YOLO-only tracker
- Ball tracking is decent, but still not perfect, especially on the far side of the court.
- The far-court improvement made detection better, but also made the script slower.
- Bounce marking is still experimental.
- Far-side bounces are still harder than near-side bounces.
- Some player-contact points can still be mislabeled as bounces.
- The script is not currently trying to show hit markers.
- The mini court overlay depends on court calibration quality; bad calibration will place mapped points incorrectly.
- The missed-bounce recovery layer is currently review-only. It surfaces magenta `B?` candidates but does not count them as live scoring bounces.
- The physical jump gate affects future `track_ball_yolo.py` runs. Existing renders based on an old JSONL need the tracking log regenerated before they benefit from it.

Latest tennis9 analysis notes
- Point 1 is flagged as `serve_unobserved`: its serve bounce was never detected, so the first bounce on record is a mid-rally one on the server's own side. (It used to read `played_out_after_geometric_fault` - geometry said double fault while the players continued - which was the same miss seen from the other end.)
- Points 3 and 4 are official double faults from the manifest override.
- Scoring for tennis9 validates as `0-15`, `0-30`, `0-40`, then game score `0-1`.
- `validate_tennis9_regression.py` checks the known-good serve states, point endings, scoring, audit columns, and point debug images.

Offline bounce detection (2026-08): `bounce_detect.py`
The bottleneck described below is addressed by detecting bounces AFTER tracking,
from the logged JSONL, instead of live inside the tracker.

Why offline helps: the in-tracker EventDetector needs a window of CONSECUTIVE
frames around the bounce, but the bounce is the hardest instant to track - the
ball is fastest, blurriest and lowest-contrast against the court - so the frames
it needs are exactly the ones missing. Working from the whole logged trajectory,
a bounce is found by fitting the ballistic arc either side of a candidate instant
and measuring the discontinuity, which tolerates dropped frames.

    python bounce_detect.py --jsonl <tracking.jsonl> --court-calib-file <calib.json>
    python eval_bounce_detect.py --verbose [--review-csv out.csv]

Measured: tennis11 game 1 goes from 10 bounces to 44, serve bounces from 2/7 to
6/7, and point 5 from zero bounces to one. tennis9 recall is 20/23 against its
reviewed known-good set.

Two things it deliberately does NOT do:
- It does not reject racket contacts. In image space they are not reliably
  separable from bounces; three attempts were measured and all failed (a velocity
  threshold tuned on tennis9 inverted on tennis11 because which end serves decides
  whether a struck ball travels toward or away from camera; ball-size trend
  separated 16/23 from 7/16; striker distance overlaps because players stand where
  balls land). Contacts are FLAGGED via `near_player` and graded, never dropped.
- It does not claim unqualified precision. tennis11 game 1 now has hand labels in
  `labels/tennis11_game1_bounce_labels.csv`: 41/44 detections are real ball
  events, but only 12/44 are live in-play bounces. The useful contract is
  contextual: `rally_scoring_eligible` contains no labelled racket contacts or
  dead balls on that clip, while serve adjudication uses its own contact-anchored
  landing logic.

Two caller contracts, because the consumers need different things:
- `rally_scoring_eligible` - conservative, excludes anything near a player.
- `serve_landing_precondition` - permissive, allows a receiver-side bounce near
  the waiting receiver, since that is where serves land. It is a precondition,
  not a verdict: it filters almost nothing alone and must be combined with
  contact anchoring and receiver-side geometry.

Lesson worth keeping: five separate thresholds in this module were found to be
silently discarding real bounces, and every one of them was expressed in absolute
pixels. Bounce events scale with ball speed and court depth, so thresholds are
normalised to feet or to frames of ball travel. The one absolute pixel gate that
remains (`max_residual_px`) is kept only because an audit showed it costs no
known-good event. Likewise, two bugs came from trusting the GROUND projection for
an airborne ball - it is wrong by construction there, and must not be used to
choose which player to measure reach against, nor to reject a candidate outright.

Automated timeline hypotheses (2026-08): `timeline_hypotheses.py`
The automated path deliberately stops short of scoring. It turns tracked JSONL
logs into point-like hypotheses, then exports a report that keeps uncertainty
visible: confidence is clip-relative, `serve_count` is a hypothesis, point ends
are inferred from activity rather than observed, and the compact JSON is marked
`not_scoring_truth: true`.

One command regenerates the current tennis11 hypothesis audit from the tracked
game-1 and game-2 logs:

    python run_timeline_pipeline.py \
      --config timeline_configs/tennis11_games1_2.json

Add `--render-videos` to also create the configured review MP4:

    .venv/bin/python run_timeline_pipeline.py \
      --config timeline_configs/tennis11_games1_2.json \
      --render-videos \
      --bundle-demo

Outputs:
- `<label>_hypotheses.json` for each clip.
- `timeline_demo.html` as the entry point for show-and-review.
- `timeline_audit.html` for human review.
- `timeline_audit.json` for a compact machine-readable audit.
- Configured hypothesis overlay MP4s when `--render-videos` is passed.
- `tennis11_timeline_demo.zip` when `--bundle-demo` is passed.

The optional manifest and expected-contact inputs are evaluation only. They do
not feed scoring and do not create manifest-shaped `point_frames`. Game 1 has
verified serve contacts, so the report can show contact recall/precision. Game 2
has no ground truth and is labelled that way instead of showing blank metrics.

Validate the timeline path:

    python validate_timeline_pipeline.py
    .venv/bin/python validate_court_geometry.py

This check protects the automation boundary: importing the runner must not load
the scoring/tracker stacks, the compact audit must carry `not_scoring_truth:
true`, and no generated JSON may contain `point_frames`. The geometry check
protects the shared image-to-court projection against drifting from the previous
OpenCV implementation.

Render a hypothesis-only overlay video for visual review:

    .venv/bin/python render_timeline_hypotheses.py \
      --video yoloVids/inputs/tennis11_game1_clean.avi \
      --hypotheses yoloVids/outputs/tennis11/timeline/game_1_hypotheses.json \
      --output yoloVids/outputs/tennis11/timeline/game1_timeline_hypotheses.mp4

The renderer also reads only timeline hypothesis JSON. It draws an explicit
"not scoring truth" panel, active hypothesis details, serve contacts, review
reasons, and a bottom timeline bar. MP4 output is preferred for size; AVI is a
fallback. On this OpenCV build, MJPEG AVI inputs read reliably while the game-2
H.264 MP4 does not; convert MP4 slices to MJPEG AVI before rendering if needed.

`eval_bounce_detect.py` also reports an ANCHORING HAZARD: detections closer to a
serve strike than the minimum flight time. Those are the strike itself, and a
consumer that takes the first bounce after a strike will judge the serve on it.
That check found a live instance in the serve path.

Ball-track recall is the bottleneck for full automation
- Offline bounce detection substantially improves recall on the existing logs:
  tennis11 game 1 now finds 44 bounces instead of the live tracker's 10, and
  serve landings are found at 6/7 loose, 5/7 strict. The remaining failures are
  mostly not downstream logic failures; they happen where the input ball track
  loses the ball at the critical instant.
- The limiting measurement is now the distinct real observation rate, not the
  raw number of frames with a ball. On tennis11 game 1 the tracker reports a
  ball on 62.8% of frames, but after held/coasted repeats only 56.8% are distinct
  real observations. Game 2 drops to 50.3%, and its activity spans fragment more.
- Point timeline automation exists as `timeline_hypotheses.py`, but it is not a
  scoring source. On game 1 it finds all 7 verified serve contacts with 0.538
  contact precision; on game 2 it produces hypotheses without ground truth. A
  hypothesis report is useful for review, but feeding it straight into scoring
  would silently invent or merge points.
Point-classification fixes (2026-08)
These do not add bounces; they stop the classifier from inventing verdicts when
the bounces are missing. With offline bounces and serve motions integrated,
tennis11 game 1 automatic winner inference is 3 correct / 0 wrong / 3 abstained.
- Physical invariant: a serve crosses the net, so it cannot bounce on the
  server's own side. Such a bounce now yields `not_a_serve`, and a point whose
  first bounce is on the server's side reports `serve_unobserved` instead of
  manufacturing faults. This is what produced the phantom double faults on
  tennis11 points 2 and 4.
- Losing the ball track is no longer evidence the ball went out.
  `terminal_ball_state` used to return "out" whenever the ball was missing for 2+
  frames before the point's end frame, which made "out" the default verdict for
  any point whose tracking dropped - and left the net-error branch below it
  unreachable. Only positive evidence (a plausible out projection, or the ball
  moving out of frame) counts now; otherwise the status is `ball_track_lost`.
- New `net` terminal status: a ball hit into the net dies at the net line, so a
  track that stops within `NET_ERROR_MARGIN_FT` of the net now reads as a net
  error rather than "out".
- A wildly out-of-court projection is treated as "still airborne", not "landed
  out" (`OUT_PROJECTION_MAX_FT`). A ball in the air projects through the ground
  homography far past the baselines; tennis11 point 5 was being called out from a
  ball projecting 16ft beyond the far baseline while still in flight.
- New `double_bounce` point end: two in-bounds bounces on the same side with no
  shot from that side between them means the player did not get it back. This is
  the most reliable end signal available because it needs only bounces already
  detected, and it is what correctly resolves a net-cord ball that drops in
  (tennis11 point 1: the return clips the net, lands 3.8ft past it, bounces
  again, K wins). Consecutive bounces must progress away from the net, which
  rejects false pairs.
- `official_double_fault_points` now applies even when fewer than two fault
  bounces were detected. Previously the override could only confirm a double
  fault the detector had already found, which is the case where it is least
  needed; tennis11 point 3 is a real double fault with only one detected bounce.
  The override also keeps its serve attempt slots so the point's shots retain
  their serve labels and server attribution.

`validate_tennis9_regression.py` was re-baselined for tennis9 point 1, which the
serve-side invariant reclassifies from `played_out_after_geometric_fault` to
`serve_unobserved` (point end `unforced_error_out` -> `unknown_end`). That
point's first bounce is at world y=19.1 on the FAR player's own side while far is
the server, so it cannot be that serve - the new labels are the more accurate
diagnosis, and it is the point the notes below already flag as "geometry says
double fault but the players continued". The cost is that shot_001 loses a
`first_serve` label whose `player=far` attribution was right for the wrong
reason. Points 2-4 were unaffected. The regression passes again.

Immediate next step
- Raise bounce recall (see above). That is now the single highest-value change:
  scoring, serve state, point ends, and automatic point segmentation all sit
  downstream of it.
- Wire `court_ai/infer.py` into the tennis10 workflow: generate
  `court_calib_tennis10_model.json`, build a tennis10 manifest with point_frames
  and winners, and run the analysis pipeline end to end as a third known-good.
- Older tennis9 track: the JSONL was already regenerated with the physical jump
  gate; review any remaining visible ball jumps against
  `ball_debug.selector.rejected_candidates` and only promote missed-bounce
  candidates after visual review confirms they are true court bounces.

Speed
- The tracker runs three YOLO passes per frame (ball at `--imgsz`, the far-court
  ball ROI at `--far-ball-imgsz` 1600, and the scene model), and torch does not
  saturate the CPU cores by default. On the dev Mac it sat at ~160% CPU.
- Setting the OpenMP thread count makes it 2.2x faster (2.55 -> 1.17 s/frame,
  measured over 40 frames of 1080p):

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 .venv/bin/python track_ball_yolo.py ...

- Budget roughly 1 minute per 50 frames: a 2-minute 30fps clip is about 70
  minutes, a 28-minute video about 11 hours. Do not lower `--far-ball-imgsz` to
  save time; that pass is what finds the small far-court ball.

Environment issues encountered
- Ultralytics `track()` pulled in a missing `lap` dependency, so the YOLO-only script uses a custom simple tracker instead.
- Running OpenCV/Ultralytics inside the coding sandbox produced OpenMP shared-memory errors.
- Running directly in the user's local terminal worked.

Recommended workflow
1. Edit `track_ball_yolo.py` or the tennis analysis scripts.
2. For tennis9 analysis-only changes, run:

   python run_tennis_pipeline.py --manifest manifests/tennis9_analysis_manifest.json

3. Validate tennis9:

   python validate_tennis9_regression.py

4. For tracker changes that affect the JSONL, rerun `track_ball_yolo.py` on the source clip first, then rerun the manifest pipeline.
5. For older tennis6 tracker debugging, run:

   .venv/bin/python track_ball_yolo.py --video yoloVids/tennis6.MOV --output yoloVids/test_output.avi --court-calib-file=court_calib.json --headless

6. Open the result:

   open yoloVids/test_output.avi

7. Evaluate:
   - ball quality
   - near/far player boxes
   - false bounce markers
   - missed near/far bounces
   - mini court overlay placement
   - impossible ball jumps
   - whether jump rejections appear in `ball_debug.selector.rejected_candidates`

8. Tune one problem at a time.

Current takeaway
- Ball, player, and object tracking now work out of the box on any behind-the-
  baseline clip. The calibration guardrail stops a mismatched calibration from
  silently wrecking recall (tennis10: 41% -> 78% in-rally recall).
- Court calibration is now automatic via `court_ai/` (a CNN trained on synthetic
  courts, transferred to real footage with a top-hat line-map input). It produces
  accurate calibrations on real courts without any hand clicking.
- Known tradeoff: an accurate calibration lowers RAW ball recall because the
  court filter rejects in-flight balls. The fix is to decouple ball-candidate
  filtering from the calibration; use `--no-ball-court-filter` so the calibration
  is used only for bounce/mini-court/scoring.
- Bounce marking and missed-bounce recovery are still experimental and should not
  be treated as scoring truth without review. Hit detection is intentionally
  disabled while bounce quality is improved.
- Far-ball tracking needed special handling because of the wide-angle phone
  footage.
- The next concrete task is the ball-filter / calibration decoupling, then
  running the tennis10 analysis pipeline end to end as a second known-good.
