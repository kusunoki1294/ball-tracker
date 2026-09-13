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

## Depth-relative size (`size_vs_expected_depth`)

`size_class` compares the landed object to the **previous frame**, which provably
cannot separate an error from a recovery — they are the same jump in opposite
directions. `size_vs_expected_depth` compares it to the expected ball size **at
that image row**, fitted per clip, which can: an error lands on something too big
for where it is, a recovery lands on something correctly sized.

| tennis9 step | truth | `size_class` | `size_vs_expected_depth` |
| --- | --- | --- | ---: |
| f1405 | error | onto_larger_blob | **2.05** |
| f1591 | recovery | off_larger_blob | 0.87 |
| f1839 | recovery | off_larger_blob | 0.83 |
| f1150 | error | similar_size | 0.89 |

`f1150` stays invisible because it genuinely landed on a ball-sized object. Size
cannot see that error, and no depth normalisation changes it.

### Why this is a column and not a rule

Combined with a jump — prediction error > 60 px **and** ≥ 1.8× expected size —
this fires on **1 of 1451** continuous tennis9 steps, and that one is `f1405`, a
true error. Precise, and it breaks neither known recovery.

It is still not a production rejection rule: it catches **1 of 2** labelled
errors, and a rule fitted to a single caught example is fitted to n=1.

### `depth_model_corr` is reported for a reason

The model is only as good as the clip's ball-height distribution. A near-player
toss is high in frame *and* large, which flattens the relationship:

| clip | correlation | combined rule fires |
| --- | ---: | ---: |
| tennis9 | 0.82 | 1 / 1451 |
| tennis11 | **0.57** | 0 / 1713 |

At 0.57 the column should not be trusted. It is printed beside every row so a
reader can see that rather than reading an unreliable number as fact — the same
reason the player-box audit prints its population `n`.
