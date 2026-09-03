"""Shared validation for timeline hypothesis artifacts.

Timeline hypotheses are review artifacts, not scoring truth. This module keeps
that boundary check dependency-free so product smoke tests and pipeline tests
cannot drift apart.
"""

SCORING_KEYS = {
    "winner",
    "winner_player",
    "winner_source",
    "score_after",
    "point_score_before",
    "game_score_after",
    "set_score_after",
    "games",
    "sets",
    "point_frames",
}


def scoring_key_paths(value, path="$"):
    hits = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in SCORING_KEYS:
                hits.append(child_path)
            hits.extend(scoring_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(scoring_key_paths(child, f"{path}[{index}]"))
    return hits


def validate_hypothesis_contract(label, data):
    errors = []
    forbidden = scoring_key_paths(data)
    if forbidden:
        errors.append(f"{label}: timeline hypotheses contain scoring keys: {', '.join(forbidden)}")

    summary = data.get("summary") or {}
    serve_motion_count = summary.get("serve_motion_hypotheses")
    compatibility_count = summary.get("point_hypotheses")
    if serve_motion_count is None:
        errors.append(f"{label}: summary missing serve_motion_hypotheses")
    elif serve_motion_count != compatibility_count:
        errors.append(
            f"{label}: serve_motion_hypotheses={serve_motion_count} does not match "
            f"point_hypotheses={compatibility_count}"
        )

    for hypothesis in data.get("hypotheses") or []:
        item_label = hypothesis.get("display_id") or hypothesis.get("id")
        for key in ("start_frame", "end_frame", "start_source", "end_source"):
            if hypothesis.get(key) is None:
                errors.append(f"{label} {item_label}: missing {key}")
        if hypothesis.get("starts_have_no_truth") is not True:
            errors.append(f"{label} {item_label}: missing starts_have_no_truth=true")
        if hypothesis.get("ends_have_no_truth") is not True:
            errors.append(f"{label} {item_label}: missing ends_have_no_truth=true")
    return errors
