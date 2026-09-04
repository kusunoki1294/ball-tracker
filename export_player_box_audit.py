"""Export player-box detection swaps as reliability evidence. Audit only.

A "swap" is a frame-to-frame player-box centre move of more than one box height.
No person travels a body length in 1/30 s, so a swap is a change of what the
detector is looking at, not human motion.

This reports where the player boxes are unreliable. It deliberately does not say
a player made an error, that a rally is live, or that a point started. A
swap-heavy window means the boxes there cannot be trusted, which is grounds to
ABSTAIN from any judgement that reads them - nothing more. That distinction is
not pedantic: a metric built on these swaps was reported as a "receiver
stillness" point-ended signal and retracted, because it was counting swaps
rather than movement. See docs/experiments/tennis11_player_box_stability.md.

Every population figure here states its n, and control-only figures are marked
provisional, because two signals in this project separated hand-picked controls
and then failed against a random population.

Reads tracking JSONL only. It does not import or influence any detector.
"""

import argparse
import csv
import html
import json
import math
import os
import random


SWAP_BOX_HEIGHTS = 1.0
PLAYER_KEYS = ("player_near", "player_far")
WINDOW_FRAMES = 30
POPULATION_SAMPLE = 300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", required=True,
                        help="label=path/to/tracking.jsonl (repeatable)")
    parser.add_argument("--control", action="append", default=[],
                        help="label=frame=note, a hand-picked frame to report alongside "
                             "the population (repeatable)")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-html")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sample", type=int, default=POPULATION_SAMPLE,
                        help="random frames per clip for the population baseline "
                             "(default %(default)s)")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def read_rows(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[int(record["frame"])] = record
    return rows


def player_boxes(rows, key):
    out = {}
    for frame, record in rows.items():
        player = record.get(key) or {}
        bbox = player.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            out[frame] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                          max(1.0, float(y2 - y1)), float(player.get("conf") or 0.0))
    return out


def swaps_for(boxes):
    """Frames where the box moved further than a body length since the previous frame."""
    found = {}
    frames = sorted(boxes)
    for previous, frame in zip(frames, frames[1:]):
        if frame - previous != 1:
            continue
        (x0, y0, h0, c0), (x1, y1, h1, c1) = boxes[previous], boxes[frame]
        height = max(h0, h1)
        distance = math.hypot(x1 - x0, y1 - y0) / height
        if distance > SWAP_BOX_HEIGHTS:
            found[frame] = {
                "move_box_heights": round(distance, 2),
                "size_ratio": round(max(h0, h1) / max(1.0, min(h0, h1)), 2),
                "conf_before": round(c0, 2),
                "conf_after": round(c1, 2),
                "weaker_conf": round(min(c0, c1), 2),
            }
    return found


def window_density(swap_frames, frame, window=WINDOW_FRAMES):
    return sum(1 for k in range(frame - window, frame + 1) if k in swap_frames)


FIELDS = ["clip", "player", "frame", "seconds", "move_box_heights", "size_ratio",
          "conf_before", "conf_after", "weaker_conf", "swaps_in_prior_1s",
          "reliability_note"]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    clips = {}
    for spec in args.clip:
        if "=" not in spec:
            raise SystemExit(f"--clip needs label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        clips[label] = read_rows(path)

    rows_out = []
    summary = []
    for label, rows in clips.items():
        for key in PLAYER_KEYS:
            boxes = player_boxes(rows, key)
            found = swaps_for(boxes)
            summary.append({
                "clip": label, "player": key,
                "frames_with_box": len(boxes),
                "swaps": len(found),
                "swap_rate_pct": round(100.0 * len(found) / max(1, len(boxes)), 1),
            })
            for frame, detail in sorted(found.items()):
                rows_out.append({
                    "clip": label, "player": key, "frame": frame,
                    "seconds": round(frame / args.fps, 2),
                    **detail,
                    "swaps_in_prior_1s": window_density(found, frame),
                    # Deliberately a description of the evidence, not a judgement.
                    "reliability_note": "boxes unreliable in this window; abstain from "
                                        "judgements that read them",
                })

    # Population baseline. Control-only numbers have twice been misleading in this
    # project, so the report always carries a random population with its n.
    population = []
    for label, rows in clips.items():
        boxes = player_boxes(rows, "player_far")
        found = swaps_for(boxes)
        frames = sorted(boxes)
        pick = rng.sample(frames, min(args.sample, len(frames)))
        densities = [window_density(found, f) for f in pick]
        population.append({
            "clip": label, "n": len(pick),
            "windows_with_a_swap": sum(1 for d in densities if d > 0),
            "pct_with_a_swap": round(100.0 * sum(1 for d in densities if d > 0) / max(1, len(pick)), 1),
        })

    controls = []
    for spec in args.control:
        parts = spec.split("=", 2)
        if len(parts) < 2:
            raise SystemExit(f"--control needs label=frame[=note], got {spec!r}")
        label, frame = parts[0], int(parts[1])
        note = parts[2] if len(parts) > 2 else ""
        rows = clips.get(label)
        if rows is None:
            raise SystemExit(f"--control refers to unknown clip {label!r}")
        found = swaps_for(player_boxes(rows, "player_far"))
        controls.append({"clip": label, "frame": frame, "note": note,
                         "swaps_in_prior_1s": window_density(found, frame)})

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {args.output_csv} ({len(rows_out)} swaps)")

    if args.output_html:
        write_html(args.output_html, summary, population, controls, rows_out, args.sample)
        print(f"wrote {args.output_html}")
    print("audit only: no swap is called a player error, a live rally, or a point boundary")
    return 0


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def write_html(path, summary, population, controls, rows_out, sample):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    srows = "".join(
        "<tr>" + "".join(f"<td>{esc(s[k])}</td>" for k in
                         ("clip", "player", "frames_with_box", "swaps", "swap_rate_pct")) + "</tr>"
        for s in summary)
    prows = "".join(
        f"<tr><td>{esc(p['clip'])}</td><td>{esc(p['n'])}</td>"
        f"<td>{esc(p['windows_with_a_swap'])}</td><td>{esc(p['pct_with_a_swap'])}</td></tr>"
        for p in population)
    crows = "".join(
        f"<tr><td>{esc(c['clip'])}</td><td>f{esc(c['frame'])}</td>"
        f"<td>{esc(c['swaps_in_prior_1s'])}</td><td>{esc(c['note'])}</td></tr>"
        for c in controls) or "<tr><td colspan='4'>none supplied</td></tr>"
    document = f"""<title>Player Box Reliability Audit</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 24px; background: #f5f7f9; color: #16202b; }}
 table {{ border-collapse: collapse; margin-bottom: 22px; background: #fff; font-size: 13px; }}
 th, td {{ border: 1px solid #d5dce4; padding: 5px 9px; text-align: left; }}
 th {{ background: #eef2f6; }}
 .caveat {{ background: #fff5e6; border: 1px solid #f0c987; padding: 12px 14px;
            border-radius: 6px; margin-bottom: 18px; }}
</style>
<h1>Player Box Reliability Audit</h1>
<div class="caveat">
  <strong>Reliability evidence, not findings.</strong> A swap is a player-box centre
  move of more than one box height between consecutive frames. Nobody travels a body
  length in 1/30 s, so a swap means the detector changed what it was looking at.
  <strong>A swap is not a player error, not proof a rally is live, and not a point
  boundary.</strong> A swap-heavy window means the boxes there cannot be trusted, which
  is grounds to abstain from any judgement that reads them.
  A metric built on these swaps was reported as a &ldquo;receiver stillness&rdquo;
  point-ended signal and retracted, because it was counting swaps rather than movement.
</div>
<h2>Swap rate per clip and player</h2>
<table><thead><tr><th>clip</th><th>player</th><th>frames with a box</th><th>swaps</th><th>rate %</th></tr></thead>
<tbody>{srows}</tbody></table>
<h2>Random population baseline (far player)</h2>
<p>Control-only numbers have twice been misleading here, so every figure states its n.
A control frame is only informative against this baseline.</p>
<table><thead><tr><th>clip</th><th>n sampled</th><th>windows containing a swap</th><th>%</th></tr></thead>
<tbody>{prows}</tbody></table>
<h2>Supplied control frames</h2>
<table><thead><tr><th>clip</th><th>frame</th><th>swaps in prior 1s</th><th>note</th></tr></thead>
<tbody>{crows}</tbody></table>
<p>{len(rows_out)} swap step(s) listed in the CSV. Population sample size per clip: {sample}.</p>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)


if __name__ == "__main__":
    raise SystemExit(main())
