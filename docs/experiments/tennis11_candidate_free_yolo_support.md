# Tennis11 Candidate-Free YOLO Support

## Question

Can the existing tennis-ball YOLO model independently support vertical-reversal
events that the bounce detector never proposed?

## Experiment

The input was the 30 candidate-free events in
`yoloVids/outputs/tennis11/track_reversal_review.csv`. For each event, the
existing `vids/models/tennisball.pt` model was run at confidence `0.08` and
image size `1280` over an eleven-frame window centered on the event. A moving
YOLO track means only that a detected object moved through the window; it is not
a bounce label.

Command:

```text
.venv/bin/python export_yolo_reversal_support.py --video yoloVids/inputs/tennis11_game1_clean.avi --reviews yoloVids/outputs/tennis11/track_reversal_review.csv --model vids/models/tennisball.pt --confidence 0.08 --image-size 1280 --radius 5 --output-json yoloVids/outputs/tennis11/reversal_yolo_support_latest.json
```

## Result

- `28/30` candidate-free reversals had at least one moving YOLO track.
- `2/30` had no moving YOLO track.
- The support is not discriminative: it also appears on known racket contacts,
  tracker association switches, and false or stationary-object events.
- Therefore no candidate-free reversal is promoted into the production bounce
  list from this experiment.

### High-resolution follow-up

The same windows were rerun with confidence `0.03` and image size `1920`.
Moving-track support changed from `28/30` to `30/30`, adding support at f2158
and f2298 only. It did not provide an independent bounce discriminator, so the
higher-resolution result is also review-only and does not justify a production
retrack or threshold change by itself.

## Conclusion

The existing YOLO model can provide review evidence, but moving-object support
does not establish a ground bounce. The next recall experiment needs either
independent source-video labels or a validated physical cue such as a rebound
trajectory, with explicit abstention when the track is ambiguous. This output
remains review-only and must not be used as bounce ground truth.

## Detector parameter check

An offline sweep over `81` combinations of `max_gap`, `min_samples`, fitting
`window`, and `max_join_error_frames` was also run against the existing reviewed
detector candidates. The best result stayed at `11/11` known `live_bounce`
labels with zero additional labeled mismatches. No parameter-only variant
recovered another verified live bounce, so the production defaults were left
unchanged.
