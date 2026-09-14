# tennis9 Association Shadow Experiment

Date: 2026-09-13

This is an experiment only. It changes a temporary copy of the tracking log and
does not change tracker or bounce-detector defaults.

## Question

Can removing the two diagnosed mis-associated tracker rows recover the known
misses at `f1147` and `f1401` without creating a new false candidate?

## Results

| temporary edit | detections | f1147 | f1401 | f1446 | observation |
| --- | ---: | --- | --- | --- | --- |
| none | 44 | miss | miss | miss | baseline |
| remove f1150 | 44 | miss | miss | miss | no change |
| remove f1405 | 45 | miss | found | miss | new f1420 candidate appears |
| remove f1150 and f1405 | 45 | miss | found | miss | same extra candidate |

Removing the diagnosed f1405 association recovers f1401, but it does not
recover f1147 or f1446 and introduces f1420. The one-row recovery is therefore
not a safe production fix: it trades one known miss for an unvalidated
candidate. Removing f1150 alone has no effect because the remaining after-arc
still lacks enough clean samples for f1147.

The experiment supports the existing diagnosis: the remedy belongs upstream in
association, where the tracker can reject a poisoned jump while preserving the
real ball track. It does not justify changing `max_residual_px`, `min_samples`,
or NMS in the bounce detector.

## Reproduction

Input: `yoloVids/outputs/tennis9/play_segments/ai9.3.jsonl` with
`yoloVids/calibration/court_calib_tennis7.json`. The shadow logs were created in
`/private/tmp` by removing the `ball` object at the named frame, then passed to
`bounce_detect.py`. No repository input was modified.

