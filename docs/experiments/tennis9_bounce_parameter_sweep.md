# tennis9 Bounce Parameter Sweep

Date: 2026-09-13

This is an experiment only. It uses the 23 reviewed reference frames from
`yoloVids/outputs/tennis9/play_segments/ai9.5.analysis.json` and does not change
production defaults.

## Results

| `max_gap` | `max_residual_px` | detections | reviewed recall | misses |
| ---: | ---: | ---: | ---: | --- |
| 6 | 10 | 38 | 19/23 | f1147, f1401, f1446, f2059 |
| 6 | 18 | 43 | 20/23 | f1147, f1401, f1446 |
| 6 | 30 | 44 | 20/23 | f1147, f1401, f1446 |
| 12 | 10 | 39 | 19/23 | f1147, f1401, f1446, f2059 |
| 12 | 18 | 44 | 20/23 | f1147, f1401, f1446 |
| 12 | 30 | 45 | 20/23 | f1147, f1401, f1446 |
| 18 | 10 | 39 | 19/23 | f1147, f1401, f1446, f2059 |
| 18 | 18 | 44 | 20/23 | f1147, f1401, f1446 |
| 18 | 30 | 45 | 20/23 | f1147, f1401, f1446 |

The current `max_gap=12`, `max_residual_px=18` setting is on the best recall
frontier in this sweep. Tightening the residual limit loses a reviewed bounce;
loosening it adds a candidate but recovers none of the three known misses.

The remaining misses are therefore not exposed by these downstream thresholds:
f1147 and f1401 are poisoned association cases, while f1446 is an NMS ranking
case. Changing these parameters would trade reviewed recall or add unvalidated
candidates without addressing the failure causes.

