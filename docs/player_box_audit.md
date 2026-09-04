# Player Box Reliability Audit

`export_player_box_audit.py` reports where the player boxes cannot be trusted.

A **swap** is a player-box centre move of more than one box height between
consecutive frames. Nobody travels a body length in 1/30 s, so a swap means the
detector changed what it was looking at.

```bash
.venv/bin/python export_player_box_audit.py \
  --clip "game1=yoloVids/outputs/tennis11/ai11.1.jsonl" \
  --clip "game2=yoloVids/outputs/tennis11/ai11.g2.jsonl" \
  --control "game1=1485=verified serve, receiver window corrupted" \
  --control "game2=4183=verified serve, clean" \
  --output-csv yoloVids/outputs/player_box_audit.csv \
  --output-html yoloVids/outputs/player_box_audit.html
```

## What a swap is not

**Not a player error, not proof a rally is live, not a point boundary.** A
swap-heavy window means the boxes there are unreliable, which is grounds to
*abstain* from any judgement that reads them.

That distinction has already cost a result. A metric built on these swaps was
reported as a "receiver stillness" point-ended signal and retracted once it
turned out to be ~99% a swap counter rather than a movement measure. See
`docs/experiments/tennis11_player_box_stability.md`.

## Current picture

| clip | player | frames | swaps | rate |
| --- | --- | ---: | ---: | ---: |
| game1 | near | 3190 | 14 | 0.4% |
| game1 | **far** | 3330 | **231** | **6.9%** |
| game2 | near | 3940 | 4 | 0.1% |
| game2 | far | 4244 | 43 | 1.0% |

## The population baseline is not optional

Every report carries a random-population figure with its **n**, because two
signals in this project separated hand-picked controls and then failed against a
random population.

The baseline immediately earns its place here: **37% of random game-1 windows
(n=300) already contain a swap**, against 6% in game 2. So a control frame with
"a swap in the prior second" is unremarkable in game 1 and notable in game 2, and
you cannot tell which without the baseline printed beside it.

A control-only number is provisional by definition. `validate_player_box_audit.py`
fails if the sample size is removed from the report.

## Guarantees

`validate_player_box_audit.py` (in the default `validate_project.py` suite)
checks that the far player's swap count exceeds the near player's in game 1,
that no row asserts a verdict, that the caveat survives, and that the population
sample size is stated. All three contract checks are mutation-tested. The
exporter reads tracking JSONL only and influences no detector.
