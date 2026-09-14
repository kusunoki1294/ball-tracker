# Bounce Detection And Tracker Holds

Date: 2026-09-13

This is a diagnostic experiment, not a production threshold change. It tests
whether `coast`/`interpolated` tracker rows should be removed before fitting
bounce arcs.

## Result

| clip | current input | remove every hold | remove only held runs >= 3 frames |
| --- | ---: | ---: | ---: |
| tennis9 | 44 detections, 20/23 known recall | 41 detections, 21/23 | 44 detections, 20/23 |
| tennis11 game 1 | 44 detections, 44/44 verified-contact recall | 41 detections, 33/44 | 44 detections, 44/44 |

There are no held runs of three or more consecutive frames in either input, so
the adaptive run-length variant is equivalent to the current input. Removing
all holds loses eleven verified tennis11 events, while gaining only one known
tennis9 event. The clips therefore do not support filtering holds as a general
rule.

## Policy

The current detector policy is the supported compromise:

- short held positions may bridge an arc fit when the surrounding observations
  need continuity;
- a held position cannot anchor a bounce at its own frame;
- optional missed-bounce recovery uses observed positions only, so it cannot
  manufacture a recovery candidate from a tracker hold.

The last rule is enforced in `find_missed_bounce_candidates`. It is deliberately
separate from the main detector because the two paths have different evidence
contracts.

## Reproduction

The experiment used `bounce_detect.detect_bounces` on the existing logs, with
rows filtered by contiguous hold-run length. Known recall is measured against
the existing tennis9 analysis bounces and tennis11's verified serve-contact
frames. Those references are not exhaustive precision labels.
