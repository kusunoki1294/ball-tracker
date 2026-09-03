# Tracker Association Audit

`export_association_audit.py` lists tracker steps where the accepted detection
landed far from the tracker's own prediction while tracking continuously. Those
steps are where the track can jump to a different object — a player's body, a
ball in someone's hand — after which the poisoned prediction rejects real
detections behind it. That is the cause of the two tennis9 bounce recall misses
driven by association, `f1147` and `f1401`. The third remaining tennis9
miss, `f1446`, is a different failure class: the bounce detector finds the
bounce, but suppression keeps a nearby racket contact instead. See
`docs/experiments/tennis9_f1446_suppression.md`.

```bash
.venv/bin/python export_association_audit.py \
  --clip "tennis9=yoloVids/outputs/tennis9/play_segments/ai9.3.jsonl" \
  --clip "tennis11=yoloVids/outputs/tennis11/ai11.1.jsonl" \
  --output-csv yoloVids/outputs/association_audit.csv \
  --output-html yoloVids/outputs/association_audit.html
```

## What it is not

**It does not identify mis-associations.** An error and a recovery are the same
jump in opposite directions: at tennis9 `f1839` the flagged step is the tracker
climbing back off the near player's body onto the real ball. Labelling these
automatically would be wrong roughly half the time, which is why `size_class` and
`track_outcome` are descriptions and `review_verdict` ships empty for a person to
fill in.

Of the 82 steps it currently reports, 73 continue tracking normally and 9 sit on
a track that dies. See `docs/experiments/tennis9_association_labelled_set.md` for
the reasoning and `docs/experiments/tennis9_association_signals.md` for why a
rejection rule was measured and rejected.

## Columns

| column | meaning |
| --- | --- |
| `prediction_error_px` | how far the accepted detection was from the predicted position |
| `jump_px` | distance travelled since the previous accepted frame |
| `size_class` | `onto_larger_blob`, `off_larger_blob`, `similar_size` — observation only |
| `track_outcome` | whether the track survived the next 6 frames |
| `selector_reason` | the tracker's own reason string |
| `review_verdict` | **empty by design**, for a human |

Only steps with `missed_frames_before == 0` are considered: after a gap the
prediction is stale and routinely hundreds of pixels out, which would bury the
real candidates.

## Guarantees

`validate_association_audit.py` (in the default `validate_project.py` suite)
checks four hand-verified steps keep their observed classification, that no row
ships a verdict, and that the HTML keeps its caveat. It reads tracking JSONL
only and never imports or influences `track_ball_yolo`.
