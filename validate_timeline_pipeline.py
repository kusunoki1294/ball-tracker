"""Validate the experimental timeline-hypothesis pipeline.

This protects the most important product boundary in the automated path:
timeline hypotheses are review artifacts, not scoring truth. The runner must not
import the scoring/tracker stacks, and its compact JSON must not emit
`point_frames` or any shape that can be pasted into a manifest as fact.
"""

import json
import os
import subprocess
import sys
import tempfile

from timeline_hypotheses import resolve_single_server_vote


TIMELINE_CONFIG = "timeline_configs/tennis11_games1_2.json"


def pipeline_python():
    venv_python = os.path.join(".venv", "bin", "python")
    return venv_python if os.path.exists(venv_python) else sys.executable


def run(command):
    return subprocess.run(command, check=True, text=True, capture_output=True)


def validate_import_boundary():
    code = """
import sys
import run_timeline_pipeline
for name in ('analyze_tennis_events', 'track_ball_yolo', 'torch', 'ultralytics'):
    print(f'{name}={name in sys.modules}')
"""
    result = run([sys.executable, "-c", code])
    loaded = dict(line.split("=", 1) for line in result.stdout.strip().splitlines())
    errors = []
    for name, value in loaded.items():
        if value != "False":
            errors.append(f"importing run_timeline_pipeline unexpectedly loaded {name}")
    return errors


def validate_outputs(output_dir):
    run(
        [
            pipeline_python(),
            "run_timeline_pipeline.py",
            "--config",
            TIMELINE_CONFIG,
            "--out-dir",
            output_dir,
        ]
    )
    audit_path = os.path.join(output_dir, "timeline_audit.json")
    demo_path = os.path.join(output_dir, "timeline_demo.html")
    with open(audit_path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)

    errors = []
    payload = json.dumps(audit)
    if audit.get("not_scoring_truth") is not True:
        errors.append("timeline audit JSON must carry not_scoring_truth=true")
    if "point_frames" in payload:
        errors.append("timeline audit JSON must not contain point_frames")
    if not os.path.exists(demo_path):
        errors.append("timeline demo HTML was not generated")
    else:
        with open(demo_path, "r", encoding="utf-8") as handle:
            demo = handle.read()
        if "point_frames" in demo:
            errors.append("timeline demo HTML must not contain point_frames")
        if "contact sheet (7 accepted)" not in demo:
            errors.append("timeline demo must label game 1 contact review contents")
        if "contact sheet (9 accepted + 8 suppressed)" not in demo:
            errors.append("timeline demo must label game 2 contact review contents")
        for frame in ("f623", "f3506", "f4183"):
            if frame not in demo:
                errors.append(f"timeline demo must surface review priority {frame}")

    expected_contact_assets = {
        "game1_serve_contact_review": 28,
        "game2_serve_contact_review": 68,
    }
    for stem, expected_count in expected_contact_assets.items():
        review_path = os.path.join(output_dir, f"{stem}.html")
        assets_dir = os.path.join(output_dir, f"{stem}_assets")
        if not os.path.exists(review_path):
            errors.append(f"missing generated contact review {review_path}")
            continue
        if not os.path.isdir(assets_dir):
            errors.append(f"missing generated contact review assets {assets_dir}")
            continue
        count = len(
            [
                name
                for name in os.listdir(assets_dir)
                if name.startswith("contact_") and name.lower().endswith(".jpg")
            ]
        )
        if count != expected_count:
            errors.append(
                f"{stem}: expected {expected_count} strip assets, got {count}"
            )
    for name in ("serve_racket_cue_eval.html", "serve_racket_cue_eval.csv"):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            errors.append(f"missing racket cue audit artifact {path}")

    clips = {clip["label"]: clip for clip in audit.get("clips", [])}
    expected = {
        "game 1": {
            "point_hypotheses": 6,
            "isolated_point_start_candidates": 0,
            "high_confidence_hypotheses": 1,
            "serve_motions": 13,
            "suppressed_rally_motions": 6,
            "contact_recall": 1.0,
            "contact_precision": 0.538,
            "single_server": True,
        },
        "game 2": {
            "point_hypotheses": 9,
            "isolated_point_start_candidates": 0,
            "high_confidence_hypotheses": 5,
            "serve_motions": 17,
            "suppressed_rally_motions": 8,
            "contact_evaluation": None,
            "single_server": False,
        },
    }
    for label, expected_values in expected.items():
        clip = clips.get(label)
        if not clip:
            errors.append(f"missing clip {label!r} in timeline audit JSON")
            continue
        summary = clip.get("summary") or {}
        hypotheses = clip.get("hypotheses") or []
        for key in (
            "point_hypotheses",
            "isolated_point_start_candidates",
            "high_confidence_hypotheses",
            "serve_motions",
            "suppressed_rally_motions",
        ):
            if summary.get(key) != expected_values[key]:
                errors.append(
                    f"{label}: expected {key}={expected_values[key]}, got {summary.get(key)}"
                )
        expected_single_server = expected_values.get("single_server", False)
        if summary.get("single_server") is not expected_single_server:
            errors.append(f"{label}: expected single_server={expected_single_server}")
        servers = {
            (hypothesis.get("attempts") or [{}])[0].get("server")
            for hypothesis in hypotheses
            if hypothesis.get("attempts")
        }
        if expected_single_server and len(servers) > 1:
            errors.append(f"{label}: single-game hypotheses have mixed servers {sorted(servers)}")
        if expected_single_server and servers and summary.get("resolved_single_server") not in servers:
            errors.append(
                f"{label}: resolved_single_server={summary.get('resolved_single_server')} "
                f"does not match hypothesis servers {sorted(servers)}"
            )
        vote = summary.get("single_server_vote")
        if expected_single_server:
            if not vote:
                errors.append(f"{label}: expected single_server_vote")
            elif vote.get("contested"):
                errors.append(f"{label}: expected uncontested single_server_vote")
            elif vote.get("margin", 0) < vote.get("min_margin", 0):
                errors.append(f"{label}: single_server_vote margin is below threshold")
        elif vote is not None:
            errors.append(f"{label}: expected no single_server_vote")
        evaluation = clip.get("contact_evaluation")
        if "contact_evaluation" in expected_values and expected_values["contact_evaluation"] is None:
            if evaluation is not None:
                errors.append(f"{label}: expected no contact_evaluation")
        else:
            if not evaluation:
                errors.append(f"{label}: missing contact_evaluation")
            else:
                for key in ("contact_recall", "contact_precision"):
                    if evaluation.get(key) != expected_values[key]:
                        errors.append(
                            f"{label}: expected {key}={expected_values[key]}, got {evaluation.get(key)}"
                        )
    return errors


def fake_point(server, confidence="high", source="ball_toss", landing_result="in"):
    return {
        "attempts": [
            {
                "server": server,
                "confidence": confidence,
                "source": source,
                "landing": {
                    "bounce_id": "bounce_fake" if landing_result else None,
                    "result": landing_result,
                },
            }
        ]
    }


def validate_contested_single_server_vote():
    errors = []
    both_gates = [
        fake_point("near", "high", "ball_toss", "in"),
        fake_point("near", "medium", "peak_reach", None),
        fake_point("far", "high", "ball_toss", "in"),
    ]
    resolved, vote = resolve_single_server_vote(both_gates)
    if resolved is not None:
        errors.append(f"contested single-server vote should abstain, got {resolved}")
    if vote.get("contested") is not True:
        errors.append("contested single-server vote did not set contested=true")

    margin_only = [
        fake_point("near", "medium", "peak_reach", None),
        fake_point("near", "medium", "peak_reach", None),
        fake_point("near", "medium", "peak_reach", None),
        fake_point("far", "high", "ball_toss", "in"),
    ]
    resolved, vote = resolve_single_server_vote(margin_only)
    if resolved is not None or vote.get("contested") is not True:
        errors.append("single-server vote should abstain when only the vote margin is thin")

    count_only = [
        fake_point("near", "high", "peak_reach", None),
        fake_point("near", "high", "peak_reach", None),
        fake_point("near", "high", "peak_reach", None),
        fake_point("near", "high", "peak_reach", None),
        fake_point("far", "medium", "ball_toss", "fault"),
        fake_point("far", "medium", "ball_toss", "fault"),
        fake_point("far", "medium", "ball_toss", "fault"),
        fake_point("far", "medium", "ball_toss", "fault"),
        fake_point("far", "medium", "ball_toss", "fault"),
    ]
    resolved, vote = resolve_single_server_vote(count_only)
    if resolved is not None or vote.get("contested") is not True:
        errors.append("single-server vote should abstain when only the count margin is thin")

    at_threshold = [
        fake_point("near", "high", "ball_toss", "fault"),
        fake_point("near", "medium", "ball_toss", "fault"),
        fake_point("near", "unknown", "unknown", "in"),
        fake_point("far", "high", "ball_toss", "in"),
    ]
    resolved, vote = resolve_single_server_vote(at_threshold)
    if resolved != "near" or vote.get("contested"):
        errors.append("single-server vote should resolve at the exact margin/count thresholds")
    return errors


def main():
    errors = []
    errors.extend(validate_import_boundary())
    errors.extend(validate_contested_single_server_vote())
    with tempfile.TemporaryDirectory() as tmpdir:
        errors.extend(validate_outputs(tmpdir))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("timeline pipeline validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
