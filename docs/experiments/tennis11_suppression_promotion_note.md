# Suppressed Serve Promotion Note

Date: 2026-08-31

## Context

Federer's game-2 verification found 4 real serves among 8 suppressed
serve-motion contacts: `f623`, `f2517`, `f3506`, and `f4183`. These are real
detection wins that the timeline grouping layer currently hides as rally
continuations.

The tempting fix is to stop suppressing high-confidence `ball_toss` motions when
they either have a long gap from the previous contact or land in the service box.

## Tested Rule

Surface a suppressed rally motion as its own serve-motion hypothesis when:

- `confidence == high`
- `source == ball_toss`
- and either the gap from the previous contact is at least 10 seconds or the
  hypothesised landing is in the receiver service box

Apply this only to clips that are not declared `single_server`.

## Result

Game 1 stayed stable only after constraining the rule away from `single_server`
clips. Without that guard, a far-side false positive entered the sequence before
the single-server filter and let two known false motions surface later.

On game 2, the rule surfaced several confirmed missed serves, but it also caused
the later confirmed serve at `f819` to disappear from accepted hypotheses after
`f623` became the active previous motion. It also surfaced `f1931`, a verified
non-serve.

So the rule trades one class of error for another. It recovers some missed
serves, but it can hide an already accepted true serve and promote a known
non-serve.

## Decision

Do not promote suppressed motions automatically yet.

The current product behavior is better:

- keep stable accepted hypotheses unchanged
- surface `f786`, `f623`, `f3506`, and `f4183` as front-page review priorities
- provide `timeline_preroll_review.html` so a human can inspect the pre-contact
  context
- keep all suppressed motions in the contact review sheet

Future work should model these as contested branches or alternate hypotheses,
not mutate the single accepted sequence in place.

