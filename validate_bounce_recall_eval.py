"""Validate the review-only bounce recall evaluator's contract."""

from eval_bounce_recall import evaluate


def main():
    detector = [
        {"label": "live_bounce"},
        {"label": "racket"},
        {"label": "dead_bounce"},
    ]
    candidate_free = [
        {"label": "live_bounce", "review_scope": "candidate_free_reversal"},
        {"label": "ambiguous", "review_scope": "candidate_free_reversal"},
    ]
    result = evaluate(detector, candidate_free)
    errors = []
    if result["reviewed_live_bounces"] != 2:
        errors.append("reviewed live-bounce count must include candidate-free labels")
    if result["detected_live_bounces"] != 1:
        errors.append("detected live-bounce count must come only from detector labels")
    if result["recall_before_retracking"] != 0.5:
        errors.append("recall must be detected live divided by reviewed live")
    if result["not_scoring_truth"] is not True:
        errors.append("evaluation must be marked not_scoring_truth")
    try:
        evaluate(detector, [{"label": "", "review_scope": "candidate_free_reversal"}])
    except ValueError:
        pass
    else:
        errors.append("blank candidate-free labels must fail before metrics are emitted")
    try:
        evaluate(detector, [{
            "label": "live_bounce", "review_scope": "wrong_population"
        }])
    except ValueError:
        pass
    else:
        errors.append("wrong candidate-free population must fail before metrics are emitted")

    if errors:
        for error in errors:
            print(error)
        return 1
    print("bounce recall evaluator validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
