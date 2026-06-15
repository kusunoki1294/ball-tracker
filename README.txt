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
  - red-mark and clean overlay renders
- The YOLO-only tracker is still experimental for:
  - bounce detection
  - far-side bounce detection
  - separating true bounces from player or racket contact points
- Current known-good analysis target:
  - manifest: `manifests/tennis9_analysis_manifest.json`
  - analysis JSON: `yoloVids/outputs/tennis9/play_segments/ai9.5.analysis.json`
  - red-mark render: `yoloVids/outputs/tennis9/play_segments/ai9.5.avi`
  - clean render: `yoloVids/outputs/tennis9/play_segments/ai9.6.avi`

Files
- `track_ball.py`: original tracker with HSV/motion and optional YOLO support
- `track_ball_yolo.py`: YOLO-only tracker for ball, players, other objects, experimental bounce markers, and a mini court overlay with mapped ball/bounce points
- `court_calib.json`: saved court calibration data
- `vids/`: older sample videos, model files, helper files
- `yoloVids/`: newer YOLO workflow videos and renders
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
- Point 1 is flagged as `played_out_after_geometric_fault` because geometry says double fault but the players continued.
- Points 3 and 4 are official double faults from the manifest override.
- Scoring for tennis9 validates as `0-15`, `0-30`, `0-40`, then game score `0-1`.
- `validate_tennis9_regression.py` checks the known-good serve states, point endings, scoring, audit columns, and point debug images.

Immediate next step
- Rerun `track_ball_yolo.py` to regenerate the tennis9 JSONL with the physical jump gate enabled, then rerun the manifest pipeline.
- Review any remaining visible ball jumps and compare them against `ball_debug.selector.rejected_candidates`.
- Only promote missed-bounce candidates into live bounces after visual review proves they are true court bounces.

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
- YOLO-only ball and player tracking is usable.
- Far-ball tracking needed special handling because of the wide-angle phone footage.
- Bounce marking and missed-bounce recovery are still experimental and should not be treated as scoring truth without review.
- Hit detection is intentionally disabled for now while bounce quality is being improved.
- The next concrete task is regenerating tennis9 tracking logs with the physical jump gate and reviewing any remaining jumps.
