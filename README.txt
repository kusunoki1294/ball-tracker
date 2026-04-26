Ball Tracker

This project started as an OpenCV tennis-ball tracker and now also includes a separate YOLO-only pipeline for tracking the ball, players, and other court objects.

Current status
- `track_ball.py` is the original hybrid tracker. It uses HSV/color, motion, optional court logic, and optional YOLO ball detection.
- `track_ball_yolo.py` is the newer YOLO-only tracker. This is the main experimental path for the tennis project right now.
- The YOLO-only tracker works reasonably well for:
  - ball tracking
  - near/far player labeling
  - other object boxes such as rackets and cars
- The YOLO-only tracker is still experimental for:
  - bounce detection
  - far-side bounce detection
  - separating true bounces from player or racket contact points
- Current real-world result on `yoloVids/tennis6.MOV`:
  - about 5 bounces are detected total
  - near-side bounces can be found
  - far-side bounces are still effectively 0
  - this is still far short of the expected 20+ bounce/contact events in the rally footage

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

8. Experimental bounce markers
   - current event work is bounce-only for now
   - hit markers were removed temporarily because they were too noisy

9. Side-aware bounce filtering
   - uses `court_calib.json` when available
   - tries to treat near-side and far-side bounce candidates differently
   - rejects candidates that project outside the singles court

10. Mini singles-court overlay
   - draws a small singles court in the top-right corner
   - plots the current ball position and recent bounce markers when calibration is available
   - current state is still not correct for `tennis6`
   - mapping improved after replacing the stale calibration file, but the mini-map orientation is still flipped
   - the near side is still appearing at the top of the mini-map, which is backwards for this camera view

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
- Even with the new `tennis6` calibration, the mini-map orientation is still backwards:
  - near side is shown on the top half
  - far side is shown on the bottom half
- Bounce mapping is still not trustworthy enough to use as a source of truth for side classification.
- Current detector behavior on `tennis6` is roughly:
  - around 5 detected bounces total
  - 0 far-side bounces
  - still much lower than the expected number of bounce/contact moments in the clip

Latest calibration/debug notes
- The old `court_calib.json` was invalid for `tennis6` because its coordinates were outside the `1280x720` frame.
- That stale calibration was replaced with a `tennis6`-specific calibration fitted to the actual visible doubles court.
- The projected court overlay from that replacement calibration looked visually plausible on the extracted frame.
- Despite that, the runtime mini-map is still oriented incorrectly, so the world-to-mini-court convention is still wrong somewhere in `track_ball_yolo.py`.
- Because the mini-map and side classification are coupled through projected court coordinates, far-side bounce logic should not be trusted until that orientation issue is fixed.

Immediate next step
- Fix the mini-court/world-coordinate orientation in `track_ball_yolo.py` so:
  - far court is the top half of the mini-map
  - near court is the bottom half of the mini-map
  - mapped bounce points land in the correct service box / lateral lane
- After the map orientation is correct, retune far-side bounce detection.
- Do not spend more time tuning bounce thresholds until the court-world mapping is verified correct during an actual `tennis6` render.

Environment issues encountered
- Ultralytics `track()` pulled in a missing `lap` dependency, so the YOLO-only script uses a custom simple tracker instead.
- Running OpenCV/Ultralytics inside the coding sandbox produced OpenMP shared-memory errors.
- Running directly in the user's local terminal worked.

Recommended workflow
1. Edit `track_ball_yolo.py`.
2. Run on:

   .venv/bin/python track_ball_yolo.py --video yoloVids/tennis6.MOV --output yoloVids/test_output.avi --court-calib-file=court_calib.json --headless

3. Open the result:

   open yoloVids/test_output.avi

4. Evaluate:
   - ball quality
   - near/far player boxes
   - false bounce markers
   - missed near/far bounces
   - mini court overlay placement
   - whether the mini-map is upside down
   - whether near-side bounces map to the lower half
   - whether far-side bounces map to the upper half

5. Tune one problem at a time.

Current takeaway
- YOLO-only ball and player tracking is usable.
- Far-ball tracking needed special handling because of the wide-angle phone footage.
- Bounce marking is still experimental and should not be treated as correct yet.
- Hit detection is intentionally disabled for now while bounce quality is being improved.
- The next concrete task is not generic bounce tuning.
- The next concrete task is fixing the court-world / mini-map orientation bug and then revisiting far-side bounce detection.
