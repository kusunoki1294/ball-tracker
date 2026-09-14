# tennis9 Ball Candidate Model Smoke Test

Date: 2026-09-13

The existing `ball_candidate_logistic_v1` model was evaluated as a shadow
input to the tracker output. This did not change tracker defaults or production
selection.

## Result

The model artifact reports rank-1 candidate selection of 97.1% on 140 training
groups and 100% on a 35-group validation tail. Those groups are drawn from the
same tennis9 clip and describe candidate ranking where the tracker already has
multiple detections; they do not measure missing detections or bounce recall.

Running `bounce_detect.py` on the model-assisted JSONL still produced 44
detections and missed all three known bounce frames: `f1147`, `f1401`, and
`f1446`. The baseline and model-assisted output therefore have the same known
recall result. The model's strong candidate-ranking score is not evidence that
it repairs the poisoned associations or recovers a candidate-free bounce.

## Conclusion

Do not enable the model as a bounce-recall fix based on this smoke test. A
useful next evaluation needs independent clips and labels for association
errors, recoveries, and candidate-free bounces. Until then, the model remains a
shadow candidate-ranking experiment.

