"""Run the experimental automated timeline-hypothesis pipeline.

This is intentionally separate from `run_tennis_pipeline.py`. It does not read
or produce scoring inputs, and it does not emit `point_frames`. Its job is to
turn one or more tracked JSONL logs into timeline hypothesis JSON files and the
hypothesis-only audit report.
"""

import argparse
import json
import os
import sys

from export_timeline_audit import build_html, compact_data
from timeline_hypotheses import (
    FPS_DEFAULT,
    build_hypotheses,
    evaluate_against_manifest,
    evaluate_contacts,
    parse_contact_frames,
    read_tracking_log,
    print_summary,
)


def parse_label_path(raw, flag_name):
    if "=" not in raw:
        raise ValueError(f"{flag_name} expects LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"{flag_name} expects non-empty LABEL=PATH, got {raw!r}")
    return label, path


def parse_label_value(raw, flag_name):
    if "=" not in raw:
        raise ValueError(f"{flag_name} expects LABEL=VALUE, got {raw!r}")
    label, value = raw.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise ValueError(f"{flag_name} expects non-empty LABEL=VALUE, got {raw!r}")
    return label, value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="LABEL=JSONL",
        help="Tracked JSONL log to process. Repeat for comparison reports.",
    )
    parser.add_argument("--court-calib-file", required=True, help="Court calibration JSON.")
    parser.add_argument("--out-dir", required=True, help="Directory for generated timeline outputs.")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--title", default="Timeline Hypothesis Audit")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional evaluation manifest for the matching clip label.",
    )
    parser.add_argument(
        "--expected-contact-frames",
        action="append",
        default=[],
        metavar="LABEL=F1,F2,...",
        help="Optional verified serve-contact frames for the matching clip label.",
    )
    parser.add_argument("--contact-tolerance-frames", type=int, default=3)
    parser.add_argument("--activity-gap-seconds", type=float, default=2.0)
    parser.add_argument("--span-pad-seconds", type=float, default=1.5)
    parser.add_argument("--scan-window-seconds", type=float, default=15.0)
    parser.add_argument("--scan-step-seconds", type=float, default=5.0)
    return parser.parse_args()


def output_stem(label):
    safe = []
    for char in label.lower().replace(" ", "_"):
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    stem = "".join(safe).strip("_")
    return stem or "clip"


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(path, data):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def build_lookup(entries, parser, flag_name):
    result = {}
    for raw in entries:
        label, value = parse_label_value(raw, flag_name)
        if label in result:
            raise ValueError(f"duplicate {flag_name} label {label!r}")
        result[label] = parser(value)
    return result


def run_clip(label, jsonl_path, args, manifests, expected_contacts):
    rows, by_frame = read_tracking_log(jsonl_path)
    result = build_hypotheses(
        rows,
        by_frame,
        args.court_calib_file,
        args.fps,
        args.activity_gap_seconds,
        args.span_pad_seconds,
        args.scan_window_seconds,
        args.scan_step_seconds,
    )
    result["source_jsonl"] = jsonl_path
    if label in manifests:
        result["evaluation"] = evaluate_against_manifest(result["hypotheses"], manifests[label])
    if label in expected_contacts:
        result["contact_evaluation"] = evaluate_contacts(
            result["serve_motions"],
            expected_contacts[label],
            args.contact_tolerance_frames,
        )
    return result


def main():
    args = parse_args()
    clips = [parse_label_path(raw, "--clip") for raw in args.clip]
    labels = [label for label, _ in clips]
    if len(labels) != len(set(labels)):
        raise ValueError("clip labels must be unique")

    manifests = build_lookup(args.manifest, lambda value: value, "--manifest")
    expected_contacts = build_lookup(
        args.expected_contact_frames,
        parse_contact_frames,
        "--expected-contact-frames",
    )
    unknown_labels = (set(manifests) | set(expected_contacts)) - set(labels)
    if unknown_labels:
        raise ValueError(
            "evaluation labels not present in --clip: " + ", ".join(sorted(unknown_labels))
        )

    os.makedirs(args.out_dir, exist_ok=True)
    report_clips = []
    for label, jsonl_path in clips:
        result = run_clip(label, jsonl_path, args, manifests, expected_contacts)
        hypothesis_path = os.path.join(args.out_dir, f"{output_stem(label)}_hypotheses.json")
        write_json(hypothesis_path, result)
        print(f"wrote {hypothesis_path}")
        print_summary(result)
        report_clips.append({"label": label, "path": hypothesis_path, "data": result})

    audit_html = os.path.join(args.out_dir, "timeline_audit.html")
    audit_json = os.path.join(args.out_dir, "timeline_audit.json")
    with open(audit_html, "w", encoding="utf-8") as handle:
        handle.write(build_html(args.title, report_clips))
    write_json(audit_json, compact_data(report_clips))
    print(f"wrote {audit_html}")
    print(f"wrote {audit_json}")
    print("not_scoring_truth: true")
    print("No scoring analysis was run, and no point_frames output was produced.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
