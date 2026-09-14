"""Validate the review-only bounce recall evaluator's contract."""

from eval_bounce_recall import evaluate


def main():
    detector = [
        {"frame": "1", "label": "live_bounce"},
        {"frame": "2", "label": "racket"},
        {"frame": "3", "label": "dead_bounce"},
    ]
    candidate_free = [
        {"frame": "4", "label": "live_bounce", "review_scope": "candidate_free_reversal", "reviewer_confidence": "high"},
        {"frame": "5", "label": "ambiguous", "review_scope": "candidate_free_reversal", "reviewer_confidence": "medium"},
    ]
    result = evaluate(detector, candidate_free)
    errors = []
    if result["reviewed_live_bounces"] != 2:
        errors.append("reviewed live-bounce count must include candidate-free labels")
    if result["detected_live_bounces"] != 1:
        errors.append("detected live-bounce count must come only from detector labels")
    if result["recall_before_retracking"] != 0.5:
        errors.append("recall must be detected live divided by reviewed live")
    if result["candidate_free_live_bounces_by_confidence"] != {"high": 1, "low": 0, "medium": 0}:
        errors.append("candidate-free live counts must retain reviewer confidence")
    if result["not_scoring_truth"] is not True:
        errors.append("evaluation must be marked not_scoring_truth")
    try:
        evaluate(detector, [{"frame": "6", "label": "", "review_scope": "candidate_free_reversal"}])
    except ValueError:
        pass
    else:
        errors.append("blank candidate-free labels must fail before metrics are emitted")
    try:
        evaluate(detector, [{
            "frame": "6", "label": "live_bounce", "review_scope": "candidate_free_reversal",
            "reviewer_confidence": "",
        }])
    except ValueError:
        pass
    else:
        errors.append("candidate-free labels without confidence must fail before metrics are emitted")
    try:
        evaluate(detector, [{
            "frame": "6", "label": "live_bounce", "review_scope": "wrong_population",
        }])
    except ValueError:
        pass
    else:
        errors.append("wrong candidate-free population must fail before metrics are emitted")
    try:
        evaluate(detector, [{
            "frame": "2", "label": "live_bounce",
            "review_scope": "candidate_free_reversal", "reviewer_confidence": "high",
        }])
    except ValueError:
        pass
    else:
        errors.append("overlapping frame populations must fail before metrics are emitted")
    try:
        evaluate([
            {"frame": "1", "label": "live_bounce"},
            {"frame": "1", "label": "racket"},
        ], candidate_free)
    except ValueError:
        pass
    else:
        errors.append("duplicate detector frames must fail before metrics are emitted")
    try:
        evaluate(detector, [
            {"frame": "4", "label": "live_bounce", "review_scope": "candidate_free_reversal", "reviewer_confidence": "high"},
            {"frame": "4", "label": "ambiguous", "review_scope": "candidate_free_reversal", "reviewer_confidence": "medium"},
        ])
    except ValueError:
        pass
    else:
        errors.append("duplicate candidate-free frames must fail before metrics are emitted")

    if errors:
        for error in errors:
            print(error)
        return 1
    print("bounce recall evaluator validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
