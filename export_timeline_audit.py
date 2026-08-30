"""Export a human-readable audit of timeline point HYPOTHESES.

Deliberately reads only `timeline_hypotheses.py` output. It does NOT read
analysis JSON from `analyze_tennis_events.py`, and it never emits a
`point_frames`-shaped structure, so a hypothesis cannot be laundered into
scoring truth by copy-paste or by a careless reader. Every count here is a count
of hypotheses, not of points.

Where ground truth exists (verified serve contact frames) the contact metrics
are shown. Where it does not, that is stated rather than left blank.
"""
import argparse
import html
import json
import os

CONFIDENCE_ORDER = {"high": 0, "uncertain": 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypotheses",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Timeline hypotheses JSON, as label=path. Repeat to compare clips.",
    )
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--data-json", default="", help="Optional compact JSON summary path.")
    parser.add_argument("--title", default="Timeline Hypothesis Audit", help="Report title.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def esc(value):
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def format_list(values):
    return ", ".join(esc(item) for item in values) if values else "—"


def parse_sources(entries):
    clips = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--hypotheses expects label=path, got {entry!r}")
        label, path = entry.split("=", 1)
        clips.append({"label": label.strip(), "path": path.strip(), "data": load_json(path.strip())})
    return clips


def landing_summary(hypothesis):
    first = (hypothesis.get("attempts") or [{}])[0]
    landing = first.get("landing") or {}
    if landing.get("bounce_frame") is not None:
        return f"{landing.get('result', 'unknown')} @f{landing['bounce_frame']}"
    return landing.get("reason") or "no landing detected"


def boundary_summary(hypothesis):
    status = hypothesis.get("boundary_status")
    if status == "point_start_hypothesis_deadtime_isolated":
        return "point-start hypothesis; dead-time isolated"
    if status == "point_start_hypothesis_no_deadtime_evidence":
        return "point-start hypothesis; no dead-time evidence"
    return status or "unknown"


def hypothesis_rows(data):
    rows = []
    for hypothesis in data.get("hypotheses", []):
        first = (hypothesis.get("attempts") or [{}])[0]
        isolation = hypothesis.get("isolation") or {}
        fragmentation = hypothesis.get("local_fragmentation") or {}
        end_note = " (inferred)" if hypothesis.get("ends_have_no_truth") else ""
        rows.append(
            "<tr class='conf-{cls}'>"
            "<td>{id}</td>"
            "<td class='num'>f{start}–f{end}{end_note}</td>"
            "<td>{conf} <span class='score'>{score}</span></td>"
            "<td>{boundary}</td>"
            "<td>{server}</td>"
            "<td class='num'>{serves}</td>"
            "<td class='num'>f{contact}</td>"
            "<td>{landing}</td>"
            "<td class='num'>{suppressed}</td>"
            "<td class='num'>{dead}</td>"
            "<td class='num'>{spans}</td>"
            "<td class='reasons'>{reasons}</td>"
            "<td class='reasons'>{review}</td>"
            "</tr>".format(
                cls=esc(hypothesis.get("confidence", "uncertain")),
                id=esc(hypothesis.get("display_id") or hypothesis.get("id")),
                start=esc(hypothesis.get("start_frame")),
                end=esc(hypothesis.get("end_frame")),
                end_note=end_note,
                conf=esc(hypothesis.get("confidence")),
                score=esc(hypothesis.get("confidence_score")),
                boundary=esc(boundary_summary(hypothesis)),
                server=esc(first.get("server")),
                serves=esc(hypothesis.get("serve_count")),
                contact=esc(first.get("contact_frame")),
                landing=esc(landing_summary(hypothesis)),
                suppressed=esc(hypothesis.get("suppressed_rally_motion_count", 0)),
                dead=esc(isolation.get("dead_frames_before")),
                spans=esc(fragmentation.get("nearby_activity_spans")),
                reasons=format_list(hypothesis.get("reasons")),
                review=format_list(hypothesis.get("review_reasons")),
            )
        )
    return "\n".join(rows)


def contact_block(data):
    evaluation = data.get("contact_evaluation")
    if not evaluation:
        return (
            "<p class='nogt'><strong>No ground truth for this clip.</strong> No verified serve "
            "contact frames were supplied, so no recall or precision can be reported. The "
            "hypothesis counts below are unvalidated.</p>"
        )
    return (
        "<p class='gt'><strong>Against verified serve contacts:</strong> "
        f"recall {esc(evaluation.get('contact_recall'))} "
        f"({esc(evaluation.get('matched_expected_contacts'))}/{esc(evaluation.get('expected_contacts'))} "
        "verified contacts found), "
        f"precision {esc(evaluation.get('contact_precision'))} "
        f"({esc(evaluation.get('matched_detected_contacts'))}/{esc(evaluation.get('detected_contacts'))} "
        "detections real), "
        f"tolerance ±{esc(evaluation.get('tolerance_frames'))}f. "
        "This measures SERVE CONTACTS, which are verified; it does not measure point boundaries, "
        "which are not.</p>"
    )


def clip_section(clip):
    data = clip["data"]
    summary = data.get("summary") or {}
    return f"""
<section>
  <h2>{esc(clip['label'])}</h2>
  <p class='src'>{esc(data.get('source_jsonl') or clip['path'])}</p>
  <div class='stats'>
    <div><span class='k'>serve-motion hypotheses</span><span class='v'>{esc(summary.get('point_hypotheses'))}</span></div>
    <div><span class='k'>high confidence</span><span class='v'>{esc(summary.get('high_confidence_hypotheses'))}</span></div>
    <div><span class='k'>uncertain</span><span class='v'>{esc(summary.get('uncertain_hypotheses'))}</span></div>
    <div><span class='k'>serve motions</span><span class='v'>{esc(summary.get('serve_motions'))}</span></div>
    <div><span class='k'>suppressed rally motions</span><span class='v'>{esc(summary.get('suppressed_rally_motions', 0))}</span></div>
    <div><span class='k'>activity spans</span><span class='v'>{esc(summary.get('activity_spans'))}</span></div>
    <div><span class='k'>distinct real observations</span><span class='v'>{esc(summary.get('distinct_real_observations_pct'))}%</span></div>
  </div>
  {contact_block(data)}
  <table>
    <thead><tr>
      <th>hypothesis</th><th>frames</th><th>confidence</th><th>boundary status</th>
      <th>server</th><th>serves<sup>†</sup></th><th>first contact</th><th>serve landing</th>
      <th>suppressed rally motions</th><th>dead frames before</th><th>nearby spans</th><th>reasons</th><th>review reasons</th>
    </tr></thead>
    <tbody>
{hypothesis_rows(data)}
    </tbody>
  </table>
  <p class='foot'><sup>†</sup> serve count is itself a hypothesis, not an observation:
  a second serve is grouped from evidence and can be over- or under-grouped.</p>
</section>"""


def comparison_section(clips):
    if len(clips) < 2:
        return ""
    header = "".join(f"<th>{esc(clip['label'])}</th>" for clip in clips)
    metrics = [
        ("serve-motion hypotheses", "point_hypotheses"),
        ("isolated point-start candidates", "isolated_point_start_candidates"),
        ("high confidence", "high_confidence_hypotheses"),
        ("uncertain", "uncertain_hypotheses"),
        ("serve motions", "serve_motions"),
        ("suppressed rally motions", "suppressed_rally_motions"),
        ("activity spans", "activity_spans"),
        ("frames with a ball %", "frames_with_ball_pct"),
        ("distinct real observations %", "distinct_real_observations_pct"),
    ]
    rows = []
    for label, key in metrics:
        cells = "".join(
            f"<td class='num'>{esc((clip['data'].get('summary') or {}).get(key))}</td>" for clip in clips
        )
        rows.append(f"<tr><td>{esc(label)}</td>{cells}</tr>")
    return f"""
<section>
  <h2>Clip comparison</h2>
  <table>
    <thead><tr><th>metric</th>{header}</tr></thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
  <p class='warn'><strong>Confidence is clip-relative and must not be compared across clips.</strong>
  A clip can produce more high-confidence hypotheses while having a worse ball track, because
  "high" largely reflects whether a serve landing happened to be detected and classified in-box —
  which is a property of the bounce detector on that clip, not evidence that the point boundaries
  are more correct. Compare confidence within a clip, never between them.</p>
</section>"""


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:2rem;
background:#f6f7f9;color:#1b1f24;line-height:1.5}
h1{margin:0 0 .25rem}h2{margin:2rem 0 .5rem}
.caveat{background:#fff4e5;border:1px solid #f0b37e;border-left:6px solid #e8871a;
padding:1rem 1.25rem;border-radius:6px;margin:1rem 0 2rem}
.caveat h3{margin:0 0 .5rem;font-size:1rem}
.src{color:#5b6570;font-size:.85rem;margin:.1rem 0 .75rem}
.stats{display:flex;flex-wrap:wrap;gap:.75rem;margin:.5rem 0 1rem}
.stats div{background:#fff;border:1px solid #dfe3e8;border-radius:6px;padding:.5rem .75rem;min-width:9rem}
.stats .k{display:block;font-size:.72rem;text-transform:uppercase;color:#5b6570;letter-spacing:.04em}
.stats .v{font-size:1.25rem;font-weight:600}
table{border-collapse:collapse;width:100%;background:#fff;font-size:.85rem;
border:1px solid #dfe3e8;border-radius:6px;overflow:hidden}
th,td{padding:.45rem .6rem;text-align:left;border-bottom:1px solid #eceff2;vertical-align:top}
th{background:#f0f2f5;font-size:.75rem;text-transform:uppercase;letter-spacing:.03em;color:#3b444d}
td.num{white-space:nowrap;font-variant-numeric:tabular-nums}
.reasons{font-size:.78rem;color:#48515a;max-width:22rem}
.score{color:#5b6570;font-size:.8rem}
tr.conf-high td{background:#f2fbf4}
.gt{background:#eef6ff;border-left:4px solid #4a90d9;padding:.6rem .9rem;border-radius:4px}
.nogt{background:#fdeeee;border-left:4px solid #d9534f;padding:.6rem .9rem;border-radius:4px}
.warn{background:#fff4e5;border-left:4px solid #e8871a;padding:.6rem .9rem;border-radius:4px}
.foot{color:#5b6570;font-size:.8rem}
@media(prefers-color-scheme:dark){
body{background:#14171a;color:#e6e9ec}
.stats div,table{background:#1d2125;border-color:#2c3238}
th{background:#232830;color:#c3cad2}td{border-color:#262c33}
tr.conf-high td{background:#17251b}
.src,.stats .k,.reasons,.score,.foot{color:#9aa4ae}
.caveat{background:#2a2113;border-color:#7a5a22}
.gt{background:#152234}.nogt{background:#2c1718}.warn{background:#2a2113}}
"""


def build_html(title, clips):
    caveat = ""
    for clip in clips:
        text = (clip["data"].get("summary") or {}).get("confidence_caveat")
        if text:
            caveat = text
            break
    sections = "".join(clip_section(clip) for clip in clips)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<h1>{esc(title)}</h1>
<div class="caveat">
  <h3>These are hypotheses, not points.</h3>
  <p>{esc(caveat) or "Hypothesis confidence is clip-relative and experimental."}</p>
  <p>Nothing here has been scored. This report reads only timeline hypothesis output and
  deliberately does not read analysis JSON, so it cannot show hypotheses as if they were
  adjudicated points. Point <em>ends</em> in particular are inferred from ball activity and have
  no ground truth at all. "No dead-time evidence" means the boundary lacks that
  specific corroboration; it does not mean the serve-contact detection is wrong.</p>
</div>
{sections}
{comparison_section(clips)}
</body></html>"""


def compact_data(clips):
    return {
        "kind": "timeline_hypothesis_audit",
        "not_scoring_truth": True,
        "clips": [
            {
                "label": clip["label"],
                "source": clip["data"].get("source_jsonl") or clip["path"],
                "summary": clip["data"].get("summary"),
                "contact_evaluation": clip["data"].get("contact_evaluation"),
                "hypotheses": [
                    {
                        "id": item.get("id"),
                        "display_id": item.get("display_id"),
                        "start_frame": item.get("start_frame"),
                        "end_frame": item.get("end_frame"),
                        "ends_have_no_truth": item.get("ends_have_no_truth"),
                        "confidence": item.get("confidence"),
                        "confidence_score": item.get("confidence_score"),
                        "boundary_status": item.get("boundary_status"),
                        "serve_count_hypothesis": item.get("serve_count"),
                        "review_reasons": item.get("review_reasons"),
                    }
                    for item in clip["data"].get("hypotheses", [])
                ],
            }
            for clip in clips
        ],
    }


def main():
    args = parse_args()
    clips = parse_sources(args.hypotheses)
    ensure_parent(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(build_html(args.title, clips))
    print(f"wrote {args.output}")
    for clip in clips:
        summary = clip["data"].get("summary") or {}
        ground_truth = "verified contacts" if clip["data"].get("contact_evaluation") else "NO ground truth"
        print(f"  {clip['label']}: {summary.get('point_hypotheses')} hypotheses "
              f"({summary.get('high_confidence_hypotheses')} high), {ground_truth}")
    if args.data_json:
        ensure_parent(args.data_json)
        with open(args.data_json, "w", encoding="utf-8") as handle:
            json.dump(compact_data(clips), handle, indent=2)
        print(f"wrote {args.data_json}")


if __name__ == "__main__":
    main()
