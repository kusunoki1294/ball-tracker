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


GAME1_JSONL = "yoloVids/outputs/tennis11/ai11.1.jsonl"
GAME2_JSONL = "yoloVids/outputs/tennis11/ai11.g2.jsonl"
COURT_CALIB = "yoloVids/calibration/court_calib_tennis11.json"
GAME1_MANIFEST = "manifests/tennis11_game1_manifest.json"
GAME1_CONTACTS = "159,635,1485,1659,2091,2432,2952"


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
            "--clip",
            f"game 1={GAME1_JSONL}",
            "--clip",
            f"game 2={GAME2_JSONL}",
            "--court-calib-file",
            COURT_CALIB,
            "--out-dir",
            output_dir,
            "--expected-contact-frames",
            f"game 1={GAME1_CONTACTS}",
            "--manifest",
            f"game 1={GAME1_MANIFEST}",
        ]
    )
    audit_path = os.path.join(output_dir, "timeline_audit.json")
    with open(audit_path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)

    errors = []
    payload = json.dumps(audit)
    if audit.get("not_scoring_truth") is not True:
        errors.append("timeline audit JSON must carry not_scoring_truth=true")
    if "point_frames" in payload:
        errors.append("timeline audit JSON must not contain point_frames")

    clips = {clip["label"]: clip for clip in audit.get("clips", [])}
    expected = {
        "game 1": {
            "point_hypotheses": 12,
            "high_confidence_hypotheses": 1,
            "serve_motions": 13,
            "contact_recall": 1.0,
            "contact_precision": 0.538,
        },
        "game 2": {
            "point_hypotheses": 17,
            "high_confidence_hypotheses": 6,
            "serve_motions": 17,
            "contact_evaluation": None,
        },
    }
    for label, expected_values in expected.items():
        clip = clips.get(label)
        if not clip:
            errors.append(f"missing clip {label!r} in timeline audit JSON")
            continue
        summary = clip.get("summary") or {}
        for key in ("point_hypotheses", "high_confidence_hypotheses", "serve_motions"):
            if summary.get(key) != expected_values[key]:
                errors.append(
                    f"{label}: expected {key}={expected_values[key]}, got {summary.get(key)}"
                )
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


def main():
    errors = []
    errors.extend(validate_import_boundary())
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
