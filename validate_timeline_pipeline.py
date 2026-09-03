"""Validate the experimental timeline-hypothesis pipeline.

This protects the most important product boundary in the automated path:
timeline hypotheses are review artifacts, not scoring truth. The runner must not
import the scoring/tracker stacks, and its compact JSON must not emit
`point_frames` or any shape that can be pasted into a manifest as fact.
"""

import json
import os
import subprocess
import time
import sys
import tempfile

from timeline_contract import validate_hypothesis_contract
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


def jsonl_max_frame(path):
    max_frame = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line).get("frame")
            if frame is not None:
                max_frame = max(max_frame, int(frame))
    return max_frame


def video_frame_count(path):
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nw=1:nk=1",
            path,
        ]
    )
    text = result.stdout.strip()
    if not text or text == "N/A":
        return 0
    return int(text)


def validate_config_video_alignment():
    with open(TIMELINE_CONFIG, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    errors = []
    for clip in config.get("clips") or []:
        label = clip.get("label") or "unknown"
        jsonl = clip.get("jsonl")
        video = clip.get("video")
        if not jsonl or not video:
            continue
        log_frames = jsonl_max_frame(jsonl)
        source_frames = video_frame_count(video)
        if source_frames != log_frames:
            errors.append(
                f"{label}: configured video has {source_frames} frames but JSONL max frame is {log_frames}"
            )
    return errors


def validate_stale_video_guard(output_dir):
    """The bundler must refuse when a review MP4 predates the JSON it depicts.

    Video rendering is opt-in behind --render-videos while pages regenerate every
    run, so without this guard a plain --bundle-demo ships fresh HTML beside a
    video burned with older hypothesis text.
    """
    import run_timeline_pipeline

    errors = []

    fresh_json = os.path.join(output_dir, "_guard_probe.json")
    fresh_video = os.path.join(output_dir, "_guard_probe.mp4")
    with open(fresh_json, "w", encoding="utf-8") as handle:
        handle.write("{}")
    time.sleep(0.02)
    with open(fresh_video, "w", encoding="utf-8") as handle:
        handle.write("")
    if run_timeline_pipeline.stale_render_reason(fresh_video, fresh_json) is not None:
        errors.append("a video newer than its hypotheses JSON must not be called stale")
    os.utime(fresh_video, (0, 0))
    if run_timeline_pipeline.stale_render_reason(fresh_video, fresh_json) != "stale":
        errors.append("a video older than its hypotheses JSON must be reported stale")
    os.unlink(fresh_video)
    if run_timeline_pipeline.stale_render_reason(fresh_video, fresh_json) != "missing":
        errors.append("an absent video must be reported missing")
    os.unlink(fresh_json)

    stale_probe_dir = os.path.join(output_dir, "_stale_video_guard")
    os.makedirs(stale_probe_dir, exist_ok=True)
    for name in ("game1_timeline_hypotheses.mp4", "game2_timeline_hypotheses.mp4"):
        path = os.path.join(stale_probe_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        os.utime(path, (0, 0))
    command = [
        pipeline_python(),
        "run_timeline_pipeline.py",
        "--config",
        TIMELINE_CONFIG,
        "--out-dir",
        stale_probe_dir,
        "--bundle-demo",
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode == 0:
        errors.append("bundling must fail when a configured review MP4 is stale")
    elif "refusing to bundle" not in completed.stderr:
        errors.append(
            "stale-video refusal must say why it refused "
            f"(exit {completed.returncode}, stderr={completed.stderr[-500:]!r})"
        )
    return errors


def validate_outputs(output_dir):
    for legacy_name in ("game1_hypotheses.json", "game2_hypotheses.json"):
        with open(os.path.join(output_dir, legacy_name), "w", encoding="utf-8") as handle:
            json.dump({"stale": True}, handle)
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
    for path in (
        os.path.join(output_dir, "game_1_hypotheses.json"),
        os.path.join(output_dir, "game_2_hypotheses.json"),
    ):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                errors.extend(validate_hypothesis_contract(path, json.load(handle)))
    if not os.path.exists(demo_path):
        errors.append("timeline demo HTML was not generated")
    else:
        with open(demo_path, "r", encoding="utf-8") as handle:
            demo = handle.read()
        if "point_frames" in demo:
            errors.append("timeline demo HTML must not contain point_frames")
        if "contact sheet (7 accepted + 6 suppressed)" not in demo:
            errors.append("timeline demo must label game 1 contact review contents")
        if "contact sheet (5 accepted + 12 suppressed)" not in demo:
            errors.append("timeline demo must label game 2 contact review contents")
        if "labelled contacts: 5/5 accepted serves" not in demo:
            errors.append("timeline demo must summarize game 2 contact labels")
        if "tennis11_demo_guide.md" not in demo:
            errors.append("timeline demo must link the demo guide")
        if "timeline_preroll_review.html" not in demo:
            errors.append("timeline demo must link the pre-roll review")
        for frame in (
            "f786",
            "f623",
            "f2517",
            "f3506",
            "f4183",
            "f257",
            "f408",
            "f2116",
            "f2629",
            "f3154",
        ):
            if frame not in demo:
                errors.append(f"timeline demo must surface review priority {frame}")
        if "f623 · 206.8s" not in demo:
            errors.append("timeline demo must derive f623 source timestamp from frame and clip start")
    for legacy_name in ("game1_hypotheses.json", "game2_hypotheses.json"):
        if os.path.exists(os.path.join(output_dir, legacy_name)):
            errors.append(f"timeline runner must remove stale {legacy_name}")

    expected_contact_assets = {
        "game1_serve_contact_review": 52,
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
    guide_path = os.path.join(output_dir, "tennis11_demo_guide.md")
    if not os.path.exists(guide_path):
        errors.append(f"missing copied demo guide {guide_path}")
    preroll_path = os.path.join(output_dir, "timeline_preroll_review.html")
    preroll_assets = os.path.join(output_dir, "timeline_preroll_review_assets")
    if not os.path.exists(preroll_path):
        errors.append(f"missing pre-roll review {preroll_path}")
    else:
        with open(preroll_path, "r", encoding="utf-8") as handle:
            preroll = handle.read()
        if "image-space tracked-ball trail" not in preroll or "tracked coverage" not in preroll:
            errors.append("pre-roll review must explain image-space trail and coverage semantics")
        if "trail interval: f635-f786" not in preroll:
            errors.append("pre-roll f786 control must use the previous-contact interval")
    if os.path.exists(preroll_path) and not os.path.isdir(preroll_assets):
        errors.append(f"missing pre-roll review assets {preroll_assets}")
    elif os.path.isdir(preroll_assets):
        count = len(
            [
                name
                for name in os.listdir(preroll_assets)
                if name.startswith("preroll_") and name.lower().endswith(".jpg")
            ]
        )
        if count != 10:
            errors.append(f"expected 10 pre-roll assets, got {count}")

    clips = {clip["label"]: clip for clip in audit.get("clips", [])}
    game1_hypotheses_path = os.path.join(output_dir, "game_1_hypotheses.json")
    if os.path.exists(game1_hypotheses_path):
        with open(game1_hypotheses_path, "r", encoding="utf-8") as handle:
            game1_hypotheses = json.load(handle)
        f786 = [
            motion
            for hypothesis in game1_hypotheses.get("hypotheses", [])
            for motion in hypothesis.get("suppressed_rally_motions", [])
            if motion.get("contact_frame") == 786
        ]
        if not f786:
            errors.append("game 1 f786 suppression must remain visible")
        else:
            reasons = f786[0].get("review_reasons") or []
            if "ball_return_evidence_dominated_by_stuck_track" not in reasons:
                errors.append("game 1 f786 must flag stuck-track-dominated return evidence")
    expected = {
        "game 1": {
            "serve_motion_hypotheses": 6,
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
            "serve_motion_hypotheses": 5,
            "point_hypotheses": 5,
            "isolated_point_start_candidates": 0,
            "high_confidence_hypotheses": 2,
            "serve_motions": 17,
            "suppressed_rally_motions": 12,
            "accepted_serve_fraction": 1.0,
            "accepted_by_side": {"near": {"serve": 5, "total": 5}},
            "suppressed_verified_serves": 4,
            "contact_evaluation": None,
            "single_server": True,
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
            "serve_motion_hypotheses",
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
        label_evaluation = clip.get("contact_label_evaluation")
        if "accepted_serve_fraction" in expected_values:
            if not label_evaluation:
                errors.append(f"{label}: expected contact label evaluation")
            elif label_evaluation.get("accepted_serve_fraction") != expected_values["accepted_serve_fraction"]:
                errors.append(
                    f"{label}: expected accepted_serve_fraction="
                    f"{expected_values['accepted_serve_fraction']}, got "
                    f"{label_evaluation.get('accepted_serve_fraction')}"
                )
            elif (
                label_evaluation.get("suppressed_verified_serves")
                != expected_values["suppressed_verified_serves"]
            ):
                errors.append(
                    f"{label}: expected suppressed_verified_serves="
                    f"{expected_values['suppressed_verified_serves']}, got "
                    f"{label_evaluation.get('suppressed_verified_serves')}"
                )
            else:
                for side, side_expected in expected_values.get("accepted_by_side", {}).items():
                    side_values = (label_evaluation.get("accepted_by_side") or {}).get(side) or {}
                    for key, value in side_expected.items():
                        if side_values.get(key) != value:
                            errors.append(
                                f"{label}: expected accepted_by_side[{side}][{key}]="
                                f"{value}, got {side_values.get(key)}"
                            )
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
    errors.extend(validate_config_video_alignment())
    errors.extend(validate_contested_single_server_vote())
    with tempfile.TemporaryDirectory() as tmpdir:
        errors.extend(validate_outputs(tmpdir))
        errors.extend(validate_stale_video_guard(tmpdir))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("timeline pipeline validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
