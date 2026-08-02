# court_ai — automatic court calibration via a keypoint model

Predicts the four doubles-court corners from a match frame, so any
behind-the-baseline video can be calibrated automatically (no hand-clicking,
no per-camera setup). This is the automatic path classical CV could not reach,
because the far baseline is faint and occluded by the net.

## How it works

1. **clean_court.py** — median-average ~40 sampled frames of a match video.
   Moving players/ball vanish; the static court remains.
2. **preprocess.py `to_linemap`** — reduce any court image to a single-channel
   **top-hat line map**. This isolates thin bright lines regardless of court or
   background brightness, giving a *domain-invariant* representation shared by
   synthetic and real images. (A fixed white threshold floods light courts.)
3. **synth.py / gen_dataset.py** — render the real court geometry
   (`court_geometry.py`) through random behind-baseline homographies at 16:9,
   with net bands, player occlusions, and clutter. Exact corner labels for free.
4. **model.py `CourtNet`** — a compact CNN that regresses the 4 corners from the
   line map. Trained only on synthetic data.
5. **train.py** — trains on cached synthetic line maps with clutter augmentation.
6. **infer.py** — video/frame → clean court → line map → predict corners →
   write a `track_ball_yolo` calibration JSON (+ homography-derived net_points)
   and an overlay for visual verification.

## Reproduce

```bash
python -m court_ai.gen_dataset --n 12000 --seed 1 --out court_ai/_data/train.npz
python -m court_ai.gen_dataset --n 1500 --seed 999 --out court_ai/_data/val.npz
# convert to line maps (see commit history) then:
python -m court_ai.train --steps 4000 --batch 48
python -m court_ai.infer --video yoloVids/inputs/tennis10_input.mp4 \
    --out yoloVids/calibration/court_calib_tennis10_model.json
```

## Results (2026-08)

Trained on 100% synthetic data, evaluated on **real** courts:

- Synthetic→real transfer with RGB input: **failed** (417px corner error).
- With the top-hat line-map input: **works**. On tennis9 the predicted court
  boundary visually traces the real court (far corners land exactly); balls
  project to world coordinates centered on the court (median x=17.4, court
  center 18). tennis10 prediction is likewise accurate (see overlays).

### Important interaction with the ball-candidate court filter

An *accurate* calibration actually lowers raw ball recall in track_ball_yolo
(64.7% vs 81% with no court filter), because a ball **in the air** projects via
the ground homography to a world point outside the court (world y down to
~-103), so the court filter rejects it. The fix is architectural: **do not gate
in-flight ball candidates on the court bounds** (keep recall high, as the
calibration guardrail already does when no calib is present), and use this
accurate calibration for its real purpose — **bounce localization, the mini
court overlay, side classification, and scoring**. Decoupling those two uses is
the recommended follow-up.
