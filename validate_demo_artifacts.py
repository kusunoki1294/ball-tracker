"""Validate the showable tennis11 demo artifacts currently on disk.

This is a product smoke check, not a regeneration test. The pipeline validators
prove code paths in temp directories; this script catches the easier-to-ship
mistake where the committed code is green but the demo files beside it are stale
or missing their caveats.
"""

import argparse
import json
import os
import sys
import zipfile


DEFAULT_ANALYSIS_MANIFEST = "manifests/tennis11_game1_manifest.json"
DEFAULT_TIMELINE_CONFIG = "timeline_configs/tennis11_games1_2.json"
LEGACY_BOUNCE_EVIDENCE_SENTINEL = "not_available_jsonl_bounce_source"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate current tennis11 demo artifacts.")
    parser.add_argument(
        "--analysis-manifest",
        default=DEFAULT_ANALYSIS_MANIFEST,
        help="Manifest for the main annotated analysis video.",
    )
    parser.add_argument(
        "--timeline-config",
        default=DEFAULT_TIMELINE_CONFIG,
        help="Timeline hypothesis config for the portable demo bundle.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_exists(errors, path, label):
    if not path:
        errors.append(f"{label}: path is not configured")
        return False
    if not os.path.exists(path):
        errors.append(f"{label}: missing {path}")
        return False
    if os.path.isfile(path) and os.path.getsize(path) <= 0:
        errors.append(f"{label}: empty {path}")
        return False
    return True


def check_newer(errors, product, source, label):
    if not check_exists(errors, product, label):
        return
    if not check_exists(errors, source, f"{label} source"):
        return
    if os.path.getmtime(product) < os.path.getmtime(source):
        errors.append(f"{label}: {product} is older than {source}")


def require_text(errors, path, snippets, label):
    if not check_exists(errors, path, label):
        return
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    for snippet in snippets:
        if snippet not in content:
            errors.append(f"{label}: {path} missing {snippet!r}")


def validate_main_analysis(manifest_path):
    errors = []
    manifest = load_json(manifest_path)
    analysis_path = manifest.get("output")
    check_exists(errors, analysis_path, "analysis JSON")
    audit = manifest.get("audit") or {}
    report = manifest.get("report") or {}
    for label, path in (
        ("audit CSV", audit.get("csv")),
        ("audit summary JSON", audit.get("summary_json")),
        ("court map", audit.get("court_map")),
        ("match report HTML", report.get("output")),
        ("match report data JSON", report.get("data_json")),
    ):
        check_exists(errors, path, label)
    for job in manifest.get("renders") or []:
        check_newer(errors, job.get("output"), job.get("analysis") or analysis_path, f"render {job.get('name')}")

    if check_exists(errors, report.get("data_json"), "match report data JSON"):
        data = load_json(report["data_json"])
        summary = data.get("summary") or {}
        expected = {
            "bounce_source": "detector",
            "points": 6,
            "final_game_score": "1-0",
            "raw_bounces": 44,
            "excluded_bounces": 2,
        }
        for key, value in expected.items():
            actual = data.get(key) if key == "bounce_source" else summary.get(key)
            if actual != value:
                errors.append(f"match report data {key}: expected {value!r}, got {actual!r}")
        if LEGACY_BOUNCE_EVIDENCE_SENTINEL in json.dumps(data):
            errors.append("detector-source tennis11 report data contains legacy-source sentinel")
        p4 = next((point for point in data.get("points") or [] if point.get("index") == 4), {})
        if "landing_evidence_low_confidence" not in (p4.get("serve_reasons") or []):
            errors.append("point 4 report data must expose low-confidence landing evidence")
    require_text(
        errors,
        report.get("output"),
        [
            "Point By Point",
            "Serve States",
            "Bounces",
            "landing_evidence_low_confidence",
        ],
        "match report HTML",
    )
    return errors


def validate_timeline(config_path):
    errors = []
    config = load_json(config_path)
    out_dir = config.get("out_dir")
    check_exists(errors, out_dir, "timeline output directory")
    if not out_dir:
        return errors

    audit_json = os.path.join(out_dir, "timeline_audit.json")
    audit_html = os.path.join(out_dir, "timeline_audit.html")
    demo_html = os.path.join(out_dir, "timeline_demo.html")
    bundle = os.path.join(out_dir, config.get("bundle_output") or "timeline_demo_bundle.zip")
    for label, path in (
        ("timeline audit JSON", audit_json),
        ("timeline audit HTML", audit_html),
        ("timeline demo HTML", demo_html),
        ("timeline bundle", bundle),
    ):
        check_exists(errors, path, label)

    for clip in config.get("clips") or []:
        label = clip.get("label") or "unknown"
        stem = label.replace(" ", "_")
        hypothesis = os.path.join(out_dir, f"{stem}_hypotheses.json")
        render = os.path.join(out_dir, clip.get("render_output", ""))
        check_newer(errors, render, hypothesis, f"{label} timeline MP4")
        contact_review = os.path.join(out_dir, clip.get("contact_review_output", ""))
        check_newer(errors, contact_review, hypothesis, f"{label} contact review")

    if check_exists(errors, audit_json, "timeline audit JSON"):
        audit = load_json(audit_json)
        payload = json.dumps(audit)
        if audit.get("not_scoring_truth") is not True:
            errors.append("timeline audit JSON must carry not_scoring_truth=true")
        if "point_frames" in payload:
            errors.append("timeline audit JSON must not contain point_frames")
        clips = {clip.get("label"): clip for clip in audit.get("clips") or []}
        game1 = clips.get("game 1") or {}
        game2 = clips.get("game 2") or {}
        game1_summary = game1.get("summary") or {}
        game2_summary = game2.get("summary") or {}
        if game1_summary.get("point_hypotheses") != 6:
            errors.append("game 1 must show 6 serve-motion hypotheses")
        if game2_summary.get("point_hypotheses") != 5:
            errors.append("game 2 must show 5 accepted serve-motion hypotheses")
        labels = game2.get("contact_label_evaluation") or {}
        if labels.get("accepted_serve_fraction") != 1.0:
            errors.append("game 2 accepted contact labels must remain 5/5 serves")
        if labels.get("suppressed_verified_serves") != 4:
            errors.append("game 2 must surface 4 suppressed verified serves")
    require_text(
        errors,
        demo_html,
        [
            "Timeline hypotheses are not scoring truth",
            "contact sheet (7 accepted + 6 suppressed)",
            "contact sheet (5 accepted + 12 suppressed)",
            "labelled contacts: 5/5 accepted serves",
        ],
        "timeline demo HTML",
    )
    require_text(
        errors,
        audit_html,
        [
            "serve-motion hypotheses",
            "They do not establish point boundaries, point winners, or scoring truth",
            "This measures SERVE CONTACTS",
        ],
        "timeline audit HTML",
    )

    if check_exists(errors, bundle, "timeline bundle"):
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
        required = {
            "timeline_demo.html",
            "timeline_audit.html",
            "timeline_audit.json",
            "game1_timeline_hypotheses.mp4",
            "game2_timeline_hypotheses.mp4",
            "game1_serve_contact_review.html",
            "game2_serve_contact_review.html",
            "timeline_preroll_review.html",
            "tennis11_demo_guide.md",
        }
        missing = sorted(required - names)
        if missing:
            errors.append("timeline bundle missing entries: " + ", ".join(missing))
        for source in (demo_html, audit_html, audit_json):
            if os.path.exists(source) and os.path.getmtime(bundle) < os.path.getmtime(source):
                errors.append(f"timeline bundle is older than {source}")
    return errors


def main():
    args = parse_args()
    errors = []
    errors.extend(validate_main_analysis(args.analysis_manifest))
    errors.extend(validate_timeline(args.timeline_config))
    if errors:
        print("demo artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("demo artifact validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
