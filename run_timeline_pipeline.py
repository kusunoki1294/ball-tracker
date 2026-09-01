"""Run the experimental automated timeline-hypothesis pipeline.

This is intentionally separate from `run_tennis_pipeline.py`. It does not read
or produce scoring inputs, and it does not emit `point_frames`. Its job is to
turn one or more tracked JSONL logs into timeline hypothesis JSON files and the
hypothesis-only audit report.
"""

import argparse
import csv
import html
import json
import os
import shutil
import sys
import zipfile

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
        "--config",
        default="",
        help="Optional non-scoring timeline pipeline config JSON.",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        metavar="LABEL=JSONL",
        help="Tracked JSONL log to process. Repeat for comparison reports.",
    )
    parser.add_argument("--court-calib-file", default="", help="Court calibration JSON.")
    parser.add_argument("--out-dir", default="", help="Directory for generated timeline outputs.")
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
    parser.add_argument(
        "--single-server",
        action="store_true",
        help="Treat each clip as one game and suppress opposite-server hypotheses.",
    )
    parser.add_argument(
        "--render-videos",
        action="store_true",
        help="Render configured hypothesis overlay videos after writing JSON/HTML outputs.",
    )
    parser.add_argument(
        "--render-max-frames",
        type=int,
        default=0,
        help="Optional frame limit for rendered videos, useful for smoke checks.",
    )
    parser.add_argument(
        "--bundle-demo",
        action="store_true",
        help="Write a flat zip bundle containing the demo page, audit files, JSON, and videos.",
    )
    parser.add_argument(
        "--bundle-output",
        default="",
        help="Optional bundle path or filename. Defaults to timeline_demo_bundle.zip in --out-dir.",
    )
    return parser.parse_args()


def load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("--config must point to a JSON object")
    return config


def configured_args(args, config):
    for field in (
        "court_calib_file",
        "out_dir",
        "title",
        "fps",
        "contact_tolerance_frames",
        "activity_gap_seconds",
        "span_pad_seconds",
        "scan_window_seconds",
        "scan_step_seconds",
        "bundle_output",
    ):
        value = config.get(field)
        if value is not None and getattr(args, field) in ("", None):
            setattr(args, field, value)
        elif value is not None and field == "title" and args.title == "Timeline Hypothesis Audit":
            setattr(args, field, value)
        elif value is not None and field in {"fps", "contact_tolerance_frames"}:
            # Numeric argparse defaults are indistinguishable from explicit CLI
            # values; the config is the intended source for config-driven runs.
            setattr(args, field, value)
        elif value is not None and field.endswith("_seconds"):
            setattr(args, field, value)
    if config.get("single_server") is not None:
        args.single_server = bool(config["single_server"])
    if not args.court_calib_file:
        raise ValueError("court calibration is required via --court-calib-file or config")
    if not args.out_dir:
        raise ValueError("output directory is required via --out-dir or config")
    return args


def copy_demo_guide(config, out_dir):
    guide = config.get("demo_guide")
    if not guide:
        return ""
    output = os.path.join(out_dir, os.path.basename(guide))
    shutil.copyfile(guide, output)
    return output


def config_entries(config, out_dir):
    clips = []
    manifests = {}
    contacts = {}
    renders = {}
    contact_reviews = {}
    contact_review_options = {}
    clip_options = {}
    review_priorities = {}
    contact_labels = {}
    for item in config.get("clips") or []:
        label = item.get("label")
        jsonl = item.get("jsonl")
        if not label or not jsonl:
            raise ValueError("each config clip must define label and jsonl")
        clips.append((label, jsonl))
        if item.get("evaluation_manifest"):
            manifests[label] = item["evaluation_manifest"]
        if item.get("expected_contact_frames"):
            raw_contacts = item["expected_contact_frames"]
            if isinstance(raw_contacts, list):
                contacts[label] = [int(frame) for frame in raw_contacts]
            else:
                contacts[label] = parse_contact_frames(str(raw_contacts))
        if item.get("video"):
            renders[label] = {
                "video": item["video"],
                "output": item.get("render_output") or f"{output_stem(label)}_timeline_hypotheses.mp4",
            }
        if item.get("contact_review_output"):
            output = item["contact_review_output"]
            if not os.path.isabs(output):
                output = os.path.join(out_dir, output)
            contact_reviews[label] = output
            contact_review_options[label] = {
                "include_suppressed": bool(item.get("contact_review_include_suppressed")),
            }
        clip_options[label] = {
            "single_server": item.get("single_server") if "single_server" in item else None,
        }
        if item.get("review_priorities"):
            review_priorities[label] = {
                "clip_start_seconds": item.get("clip_start_seconds"),
                "items": item["review_priorities"],
            }
        if item.get("contact_labels"):
            contact_labels[label] = item["contact_labels"]
    return (
        clips,
        manifests,
        contacts,
        renders,
        contact_reviews,
        contact_review_options,
        clip_options,
        review_priorities,
        contact_labels,
    )


def output_stem(label):
    safe = []
    for char in label.lower().replace(" ", "_"):
        safe.append(char if char.isalnum() or char in ("-", "_") else "_")
    stem = "".join(safe).strip("_")
    return stem or "clip"


def legacy_output_stem(label):
    safe = []
    for char in label.lower():
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
    stem = "".join(safe).strip("_")
    return stem or "clip"


def remove_stale_hypothesis_outputs(out_dir, labels):
    for label in labels:
        current = output_stem(label)
        legacy = legacy_output_stem(label)
        if legacy == current:
            continue
        path = os.path.join(out_dir, f"{legacy}_hypotheses.json")
        if os.path.exists(path):
            os.unlink(path)
            print(f"removed stale {path}")


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(path, data):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def rel_link(path, base_dir):
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return path


def priority_section(priority_items):
    if not priority_items:
        return ""
    return (
        '<section class="priorities">'
        "<h2>Review priorities</h2>"
        "<p>These are not ground truth. They are the frames most worth checking by eye.</p>"
        f"<ul>{''.join(priority_items)}</ul>"
        "</section>"
    )


def write_demo_index(
    path,
    title,
    report_clips,
    audit_html,
    audit_json,
    rendered_videos,
    contact_reviews,
    contact_review_options,
    review_priorities,
    fps,
    demo_guide,
    preroll_review,
):
    base_dir = os.path.dirname(path) or "."
    racket_html = os.path.join(base_dir, "serve_racket_cue_eval.html")
    racket_link = ""
    if os.path.exists(racket_html):
        racket_link = (
            f'<a class="secondary" href="{html.escape(rel_link(racket_html, base_dir))}">'
            "Racket cue audit</a>"
        )
    guide_link = ""
    if demo_guide and os.path.exists(demo_guide):
        guide_link = (
            f'<a class="secondary" href="{html.escape(rel_link(demo_guide, base_dir))}">'
            "Demo guide</a>"
        )
    preroll_link = ""
    if preroll_review and os.path.exists(preroll_review):
        preroll_link = (
            f'<a class="secondary" href="{html.escape(rel_link(preroll_review, base_dir))}">'
            "Pre-roll review</a>"
        )
    rows = []
    highlights = []
    video_sections = []
    priority_items = []
    for clip in report_clips:
        label = clip["label"]
        data = clip["data"]
        summary = data.get("summary") or {}
        manifest_eval = data.get("evaluation") or {}
        evaluation = data.get("contact_evaluation")
        label_evaluation = data.get("contact_label_evaluation")
        contact = "no contact ground truth"
        status = "unverified"
        if evaluation:
            contact = (
                f"contact recall {evaluation.get('contact_recall')} "
                f"precision {evaluation.get('contact_precision')}"
            )
            if (
                manifest_eval.get("truth_points")
                and summary.get("point_hypotheses") == manifest_eval.get("truth_points")
                and evaluation.get("contact_recall") == 1.0
            ):
                status = "verified contacts"
        elif label_evaluation:
            accepted = label_evaluation.get("accepted") or {}
            contact = (
                f"labelled contacts: {accepted.get('serve', 0)}/"
                f"{accepted.get('total', 0)} accepted serves"
            )
        server_mode = "boundary unknown"
        vote = summary.get("single_server_vote") or {}
        if summary.get("single_server"):
            server_mode = f"single server: {summary.get('resolved_single_server') or 'contested'}"
            if vote:
                server_mode += f" (margin {vote.get('margin')})"
        elif summary.get("resolved_single_server"):
            server_mode = f"server: {summary.get('resolved_single_server')}"
        video = rendered_videos.get(label)
        contact_review = contact_reviews.get(label)
        accepted_contact_count = sum(
            len(hypothesis.get("attempts") or [])
            for hypothesis in data.get("hypotheses") or []
        )
        suppressed_contact_count = sum(
            len(hypothesis.get("suppressed_rally_motions") or [])
            for hypothesis in data.get("hypotheses") or []
        )
        contact_review_label = f"contact sheet ({accepted_contact_count} accepted)"
        if (
            suppressed_contact_count
            and contact_review_options.get(label, {}).get("include_suppressed")
        ):
            contact_review_label = (
                f"contact sheet ({accepted_contact_count} accepted + "
                f"{suppressed_contact_count} suppressed)"
            )
        video_link = (
            f'<a href="{html.escape(rel_link(video, base_dir))}">review MP4</a>'
            if video
            else "not rendered"
        )
        contact_review_link = (
            f'<a href="{html.escape(rel_link(contact_review, base_dir))}">'
            f"{html.escape(contact_review_label)}</a>"
            if contact_review and os.path.exists(contact_review)
            else "not generated"
        )
        highlights.append(
            "<article>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{summary.get('point_hypotheses', '')} serve-motion hypotheses</strong>"
            f"<em>{html.escape(status)} · {html.escape(server_mode)}</em>"
            "</article>"
        )
        if video:
            video_href = html.escape(rel_link(video, base_dir))
            video_sections.append(
                "<section class=\"clip-video\">"
                f"<h2>{html.escape(label)}</h2>"
                f"<video src=\"{video_href}\" controls preload=\"metadata\"></video>"
                f"<p><a href=\"{video_href}\">Open {html.escape(label)} MP4</a></p>"
                "</section>"
            )
        priority_config = review_priorities.get(label) or {}
        clip_start_seconds = priority_config.get("clip_start_seconds")
        for item in priority_config.get("items") or []:
            frame = item.get("frame")
            seconds = item.get("seconds")
            if clip_start_seconds is not None and frame is not None:
                seconds = round(float(clip_start_seconds) + int(frame) / float(fps), 1)
            note = item.get("note") or item.get("reason") or ""
            kind = item.get("kind") or "review"
            time_text = f" · {html.escape(str(seconds))}s" if seconds is not None else ""
            priority_items.append(
                "<li>"
                f"<strong>{html.escape(label)} f{html.escape(str(frame))}{time_text}</strong>"
                f" <span>{html.escape(kind)}</span>"
                f" — {html.escape(str(note))}"
                "</li>"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{summary.get('point_hypotheses', '')}</td>"
            f"<td>{summary.get('distinct_real_observations_pct', '')}%</td>"
            f"<td>{summary.get('isolated_point_start_candidates', '')}</td>"
            f"<td>{summary.get('high_confidence_hypotheses', '')}</td>"
            f"<td>{summary.get('suppressed_rally_motions', 0)}</td>"
            f"<td>{summary.get('activity_spans', '')}</td>"
            f"<td>{summary.get('serve_motions', '')}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(server_mode)}</td>"
            f"<td>{html.escape(contact)}</td>"
            f"<td><a href=\"{html.escape(rel_link(clip['path'], base_dir))}\">hypotheses JSON</a></td>"
            f"<td>{video_link}</td>"
            f"<td>{contact_review_link}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #15191f; background: #f4f6f8; }}
    main {{ padding: 28px; max-width: 1320px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    .subhead {{ margin: 0 0 18px; color: #52606d; max-width: 900px; }}
    .warning {{ border-left: 5px solid #d97706; background: #fff7ed; padding: 14px 16px; margin-bottom: 20px; max-width: 1040px; }}
    .actions {{ margin: 14px 0 24px; display: flex; gap: 14px; flex-wrap: wrap; }}
    .actions a {{ background: #0f65b7; color: #fff; text-decoration: none; padding: 9px 13px; border-radius: 6px; font-weight: 650; }}
    .actions a.secondary {{ background: #dfe6ee; color: #14212e; }}
    .highlights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 18px 0 26px; }}
    .highlights article {{ background: #fff; border: 1px solid #d7dde5; border-radius: 8px; padding: 14px 16px; }}
    .highlights span {{ display: block; color: #52606d; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
    .highlights strong {{ display: block; font-size: 24px; margin-top: 3px; }}
    .highlights em {{ display: block; color: #52606d; font-style: normal; margin-top: 4px; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #d7dde5; border-radius: 8px; }}
    table {{ border-collapse: collapse; min-width: 1120px; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d7dde5; padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6fa; font-weight: 650; font-size: 13px; }}
    .videos {{ margin-top: 28px; display: grid; gap: 24px; }}
    .clip-video {{ background: #fff; border: 1px solid #d7dde5; border-radius: 8px; padding: 14px; }}
    .clip-video h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .priorities {{ background: #fff; border: 1px solid #d7dde5; border-radius: 8px; padding: 16px; margin: 0 0 22px; }}
    .priorities h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .priorities p {{ margin: 0 0 10px; color: #52606d; }}
    .priorities ul {{ margin: 0; padding-left: 20px; }}
    .priorities li {{ margin: 6px 0; }}
    .priorities span {{ color: #7a4b00; font-weight: 650; }}
    video {{ width: 100%; background: #111; display: block; border-radius: 4px; }}
    a {{ color: #0f65b7; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="subhead">Automated tennis timeline review for the tracked tennis11 clips.</p>
  <div class="warning"><strong>Timeline hypotheses are not scoring truth.</strong>
    Confidence is clip-relative, serve counts are hypotheses, and point ends are inferred.
    Game 1 has verified serve-contact checks; game 2 has contact labels but remains
    boundary-unknown and unverified for scoring.</div>
  <div class="actions">
    <a href="{html.escape(rel_link(audit_html, base_dir))}">Detailed audit</a>
    <a class="secondary" href="{html.escape(rel_link(audit_json, base_dir))}">Audit JSON</a>
    {racket_link}
    {guide_link}
    {preroll_link}
  </div>
  <section class="highlights">
    {''.join(highlights)}
  </section>
  {priority_section(priority_items)}
  <div class="table-wrap"><table>
    <thead>
      <tr>
        <th>Clip</th>
        <th>Serve-Motion Hypotheses</th>
        <th>Observed</th>
        <th>Dead-Time Isolated</th>
        <th>High, Clip-Relative</th>
        <th>Suppressed Rally Motions</th>
        <th>Activity Spans</th>
        <th>Serve Motions</th>
        <th>Status</th>
        <th>Server Mode</th>
        <th>Contact Evaluation</th>
        <th>Data</th>
        <th>Video</th>
        <th>Contact Review</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table></div>
  <p>
    <strong>Read observed track quality beside confidence.</strong>
    A clip can show more high-confidence hypotheses while being worse tracked;
    confidence is useful within a clip, not as a cross-clip quality score.
  </p>
  <div class="videos">
    {''.join(video_sections)}
  </div>
</main>
</body>
</html>
"""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


def write_demo_bundle(
    path,
    demo_index,
    audit_html,
    audit_json,
    report_clips,
    rendered_videos,
    contact_reviews,
    demo_guide,
    preroll_review,
):
    ensure_parent(path)
    files = [demo_index, audit_html, audit_json]
    if demo_guide:
        files.append(demo_guide)
    if preroll_review:
        files.append(preroll_review)
    out_dir = os.path.dirname(demo_index) or "."
    files.extend(
        os.path.join(out_dir, name)
        for name in ("serve_racket_cue_eval.html", "serve_racket_cue_eval.csv")
    )
    files.extend(clip["path"] for clip in report_clips)
    files.extend(rendered_videos.values())
    files.extend(contact_reviews.values())
    seen = set()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for file_path in files:
            if not file_path or file_path in seen or not os.path.exists(file_path):
                continue
            seen.add(file_path)
            archive.write(file_path, arcname=os.path.basename(file_path))
        for review_path in contact_reviews.values():
            assets_dir = os.path.splitext(review_path)[0] + "_assets"
            if not os.path.isdir(assets_dir):
                continue
            for name in sorted(os.listdir(assets_dir)):
                file_path = os.path.join(assets_dir, name)
                if not os.path.isfile(file_path):
                    continue
                arcname = os.path.join(os.path.basename(assets_dir), name)
                archive.write(file_path, arcname=arcname)
        if preroll_review:
            assets_dir = os.path.splitext(preroll_review)[0] + "_assets"
            if os.path.isdir(assets_dir):
                for name in sorted(os.listdir(assets_dir)):
                    file_path = os.path.join(assets_dir, name)
                    if not os.path.isfile(file_path):
                        continue
                    arcname = os.path.join(os.path.basename(assets_dir), name)
                    archive.write(file_path, arcname=arcname)


def bundle_output_path(args):
    output = args.bundle_output or "timeline_demo_bundle.zip"
    if not os.path.isabs(output):
        output = os.path.join(args.out_dir, output)
    return output


def build_lookup(entries, parser, flag_name):
    result = {}
    for raw in entries:
        label, value = parse_label_value(raw, flag_name)
        if label in result:
            raise ValueError(f"duplicate {flag_name} label {label!r}")
        result[label] = parser(value)
    return result


def load_contact_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame = row.get("frame")
            if frame:
                labels[int(frame)] = row
    return labels


def increment_label_count(bucket, label):
    key = label or "unlabelled"
    bucket[key] = bucket.get(key, 0) + 1


def evaluate_contact_labels(result, labels_path):
    labels = load_contact_labels(labels_path)
    accepted = {"total": 0}
    suppressed = {"total": 0}
    accepted_by_side = {}
    unmatched = []
    for hypothesis in result.get("hypotheses") or []:
        for attempt in hypothesis.get("attempts") or []:
            accepted["total"] += 1
            side = attempt.get("server") or "unknown"
            accepted_by_side.setdefault(side, {"total": 0})
            accepted_by_side[side]["total"] += 1
            label = labels.get(int(attempt.get("contact_frame")))
            if label:
                increment_label_count(accepted, label.get("label"))
                increment_label_count(accepted_by_side[side], label.get("label"))
            else:
                unmatched.append(int(attempt.get("contact_frame")))
        for attempt in hypothesis.get("suppressed_rally_motions") or []:
            suppressed["total"] += 1
            label = labels.get(int(attempt.get("contact_frame")))
            if label:
                increment_label_count(suppressed, label.get("label"))
            else:
                unmatched.append(int(attempt.get("contact_frame")))
    accepted_labelled = sum(
        count for label, count in accepted.items() if label != "total"
    )
    accepted_serves = accepted.get("serve", 0)
    return {
        "labels": labels_path,
        "labelled_contacts": len(labels),
        "accepted": accepted,
        "accepted_by_side": accepted_by_side,
        "suppressed": suppressed,
        "accepted_serve_fraction": round(
            accepted_serves / float(max(1, accepted_labelled)), 3
        ),
        "suppressed_verified_serves": suppressed.get("serve", 0),
        "unmatched_contact_frames": unmatched,
        "caveat": (
            "These labels evaluate serve-motion contacts only. They do not "
            "establish point boundaries, point winners, or scoring truth."
        ),
    }


def run_clip(label, jsonl_path, args, manifests, expected_contacts, clip_options, contact_labels):
    rows, by_frame = read_tracking_log(jsonl_path)
    options = clip_options.get(label, {})
    single_server = options.get("single_server")
    if single_server is None:
        single_server = args.single_server
    result = build_hypotheses(
        rows,
        by_frame,
        args.court_calib_file,
        args.fps,
        args.activity_gap_seconds,
        args.span_pad_seconds,
        args.scan_window_seconds,
        args.scan_step_seconds,
        single_server=bool(single_server),
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
    if label in contact_labels:
        result["contact_label_evaluation"] = evaluate_contact_labels(
            result,
            contact_labels[label],
        )
    return result


def render_clip_video(label, hypothesis_path, render_config, args):
    from render_timeline_hypotheses import render_video

    output = render_output_path(render_config, args)
    render_video(
        render_config["video"],
        hypothesis_path,
        output,
        max_frames=args.render_max_frames,
    )
    print(f"wrote {output}")
    return output


def render_contact_review(label, hypothesis_path, review_output, render_config, jsonl_path, options):
    from export_serve_contact_review import export_review

    export_review(
        render_config["video"],
        hypothesis_path,
        review_output,
        jsonl=jsonl_path,
        include_suppressed=bool(options.get("include_suppressed")),
    )
    return review_output


def render_output_path(render_config, args):
    output = render_config["output"]
    if not os.path.isabs(output):
        output = os.path.join(args.out_dir, output)
    return output


def optional_output_path(raw_output, out_dir):
    if not raw_output:
        return ""
    if os.path.isabs(raw_output):
        return raw_output
    return os.path.join(out_dir, raw_output)


def write_racket_cue_eval(config, report_clips, clip_jsonls, expected_contacts, out_dir):
    eval_config = config.get("racket_cue_eval") or {}
    if not eval_config:
        return None
    output_csv = optional_output_path(eval_config.get("output_csv"), out_dir)
    output_html = optional_output_path(eval_config.get("output_html"), out_dir)
    if not output_csv or not output_html:
        raise ValueError("racket_cue_eval requires output_csv and output_html")
    from serve_racket_cue_eval import build_rows, write_csv, write_html

    clip_inputs = [
        (clip["label"], clip_jsonls[clip["label"]], clip["path"])
        for clip in report_clips
    ]
    rows = build_rows(clip_inputs, expected_contacts)
    write_csv(output_csv, rows)
    write_html(output_html, rows)
    print(f"wrote {output_csv}")
    print(f"wrote {output_html}")
    return {"csv": output_csv, "html": output_html}


def write_preroll_review(config_renders, review_priorities, clip_jsonls, out_dir):
    clips = []
    for label, priority_config in review_priorities.items():
        render_config = config_renders.get(label)
        if not render_config:
            continue
        items = priority_config.get("items") or []
        if items:
            clips.append(
                {
                    "label": label,
                    "video": render_config["video"],
                    "jsonl": clip_jsonls.get(label, ""),
                    "items": items,
                }
            )
    if not clips:
        return ""
    output = os.path.join(out_dir, "timeline_preroll_review.html")
    from export_timeline_preroll_review import export_review

    export_review(clips, output)
    return output


def main():
    args = parse_args()
    config = load_config(args.config)
    args = configured_args(args, config)
    (
        config_clips,
        config_manifests,
        config_contacts,
        config_renders,
        config_contact_reviews,
        config_contact_review_options,
        config_clip_options,
        config_review_priorities,
        config_contact_labels,
    ) = config_entries(config, args.out_dir)

    clips = config_clips + [parse_label_path(raw, "--clip") for raw in args.clip]
    if not clips:
        raise ValueError("at least one clip is required via --clip or config")
    labels = [label for label, _ in clips]
    if len(labels) != len(set(labels)):
        raise ValueError("clip labels must be unique")

    manifests = {
        **config_manifests,
        **build_lookup(args.manifest, lambda value: value, "--manifest"),
    }
    expected_contacts = build_lookup(
        args.expected_contact_frames,
        parse_contact_frames,
        "--expected-contact-frames",
    )
    expected_contacts = {**config_contacts, **expected_contacts}
    unknown_labels = (set(manifests) | set(expected_contacts)) - set(labels)
    if unknown_labels:
        raise ValueError(
            "evaluation labels not present in --clip: " + ", ".join(sorted(unknown_labels))
        )

    os.makedirs(args.out_dir, exist_ok=True)
    remove_stale_hypothesis_outputs(args.out_dir, labels)
    report_clips = []
    rendered_videos = {}
    clip_jsonls = {}
    for label, jsonl_path in clips:
        clip_jsonls[label] = jsonl_path
        result = run_clip(
            label,
            jsonl_path,
            args,
            manifests,
            expected_contacts,
            config_clip_options,
            config_contact_labels,
        )
        hypothesis_path = os.path.join(args.out_dir, f"{output_stem(label)}_hypotheses.json")
        write_json(hypothesis_path, result)
        print(f"wrote {hypothesis_path}")
        print_summary(result)
        report_clips.append({"label": label, "path": hypothesis_path, "data": result})
        if args.render_videos and label in config_renders:
            rendered_videos[label] = render_clip_video(
                label, hypothesis_path, config_renders[label], args
            )
        elif label in config_renders:
            existing_render = render_output_path(config_renders[label], args)
            if os.path.exists(existing_render):
                rendered_videos[label] = existing_render
        if label in config_contact_reviews and label in config_renders:
            render_contact_review(
                label,
                hypothesis_path,
                config_contact_reviews[label],
                config_renders[label],
                jsonl_path,
                config_contact_review_options.get(label, {}),
            )

    write_racket_cue_eval(config, report_clips, clip_jsonls, expected_contacts, args.out_dir)
    demo_guide = copy_demo_guide(config, args.out_dir)
    preroll_review = write_preroll_review(
        config_renders,
        config_review_priorities,
        clip_jsonls,
        args.out_dir,
    )

    audit_html = os.path.join(args.out_dir, "timeline_audit.html")
    audit_json = os.path.join(args.out_dir, "timeline_audit.json")
    with open(audit_html, "w", encoding="utf-8") as handle:
        handle.write(build_html(args.title, report_clips))
    write_json(audit_json, compact_data(report_clips))
    demo_index = os.path.join(args.out_dir, "timeline_demo.html")
    write_demo_index(
        demo_index,
        args.title,
        report_clips,
        audit_html,
        audit_json,
        rendered_videos,
        config_contact_reviews,
        config_contact_review_options,
        config_review_priorities,
        args.fps,
        demo_guide,
        preroll_review,
    )
    print(f"wrote {audit_html}")
    print(f"wrote {audit_json}")
    print(f"wrote {demo_index}")
    if args.bundle_demo:
        bundle_path = bundle_output_path(args)
        write_demo_bundle(
            bundle_path,
            demo_index,
            audit_html,
            audit_json,
            report_clips,
            rendered_videos,
            config_contact_reviews,
            demo_guide,
            preroll_review,
        )
        print(f"wrote {bundle_path}")
    print("not_scoring_truth: true")
    print("No scoring analysis was run, and no point_frames output was produced.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
