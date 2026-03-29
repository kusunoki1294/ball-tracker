Ball Tracker

This project tracks a tennis ball in video using OpenCV-based color/motion detection, with optional YOLO-based detection.

Files
- track_ball.py: main tracking script
- track_ball_yolo.py: YOLO-only tracker for the ball, players, and other scene objects
- court_calib.json: saved court calibration data
- vids/: sample videos, model files, and helper files
- yolo.txt: reference links about YOLO

Requirements
- Python 3
- OpenCV (`opencv-python`)
- Optional: `ultralytics` for YOLO detection
- Optional: `ffmpeg` and `ffprobe` for video decoding/encoding fallback

Install
1. Create and activate a virtual environment if you want one.
2. Install dependencies:

   pip install -r vids/requirements.txt

3. Optional YOLO support:

   pip install ultralytics

Usage
Basic run:

python track_ball.py --video vids/tennis.MOV

Save annotated output:

python track_ball.py --video vids/tennis.MOV --output output.mp4

Use the included preset:

python track_ball.py --video vids/annotated2/annotated2.MOV --preset annotated2

Run with YOLO:

python track_ball.py --video vids/tennis.MOV --yolo-model vids/models/tennisball.pt

YOLO-only tracker:

python track_ball_yolo.py --video vids/tennis.MOV --output yolo_tracking.mp4

YOLO-only tracker with a custom scene model:

python track_ball_yolo.py --video vids/tennis.MOV --scene-model /path/to/yolo_scene_model.pt --output yolo_tracking.mp4

Useful options
- `--tune`: open trackbars for HSV tuning
- `--headless`: disable OpenCV windows
- `--manual-court`: click court points manually on the first frame
- `--auto-court-lines`: try automatic court line detection
- `--yolo-only`: skip HSV/motion tracking and use YOLO only
- `--force-ffmpeg`: always decode input with ffmpeg

Notes
- The script expects a video path through `--video`.
- YOLO features only work if `ultralytics` is installed and a `.pt` model is provided.
- Sample videos and a sample tennis ball model are already included under `vids/`.
- `track_ball_yolo.py` uses the local tennis ball model at `vids/models/tennisball.pt` and a second YOLO model for people/scene objects. Its default scene model is `yolov8n.pt`, which may download automatically if it is not already available.
