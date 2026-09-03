import argparse
import html
import json
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Export a static HTML match report from tennis analysis JSON.")
    parser.add_argument("--analysis", required=True, help="Analysis JSON produced by analyze_tennis_events.py.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="Tennis Tracking Report", help="Report title.")
    parser.add_argument("--point-debug-dir", default="", help="Optional per-point debug PNG directory.")
    parser.add_argument("--data-json", default="", help="Optional compact report data JSON path.")
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


def rel_link(target, base_path):
    if not target:
        return ""
    return os.path.relpath(target, os.path.dirname(base_path) or ".")


def format_list(values):
    values = values or []
    if not values:
        return '<span class="muted">none</span>'
    return "".join(f'<span class="pill warn">{esc(value)}</span>' for value in values)


def confidence_class(value):
    if value == "high":
        return "good"
    if value == "medium":
        return "mid"
    if value == "low":
        return "bad"
    return "muted"


def player_label(value):
    return esc(value or "unknown")


def display_shot_type(value):
    return {
        "first_serve": "1st serve",
        "second_serve": "2nd serve",
        "opening_shot": "opening",
    }.get(value, value or "")


def by_id(items):
    return {item.get("id"): item for item in items if item.get("id")}


def detector_value(analysis, bounce, key):
    value = bounce.get(key)
    if value is None and analysis.get("bounce_source") == "jsonl":
        return "not_available_jsonl_bounce_source"
    return value


def point_debug_link(point, debug_dir, output_path):
    if not debug_dir:
        return ""
    path = os.path.join(debug_dir, f"point_{int(point['index']):02d}.png")
    href = rel_link(path, output_path)
    return f'<a href="{esc(href)}">debug PNG</a>'


def point_debug_card(point, debug_dir, output_path):
    if not debug_dir:
        return ""
    path = os.path.join(debug_dir, f"point_{int(point['index']):02d}.png")
    href = rel_link(path, output_path)
    return f"""
    <a class="thumb" href="{esc(href)}">
      <img src="{esc(href)}" alt="Point {point.get('index')} debug map">
      <span>Point {point.get('index')}</span>
    </a>
    """


def stat_counts(analysis):
    points = analysis.get("points") or []
    shots = analysis.get("shots") or []
    bounces = analysis.get("bounces") or []
    high_speed_shots = [
        shot for shot in shots
        if (shot.get("speed") or {}).get("quality") == "high" and isinstance((shot.get("speed") or {}).get("mph"), (int, float))
    ]
    avg_speed = None
    if high_speed_shots:
        avg_speed = sum(shot["speed"]["mph"] for shot in high_speed_shots) / len(high_speed_shots)
    serve_states = Counter(point.get("serve_state") or "unknown" for point in points)
    end_types = Counter(point.get("point_end_type") or "unknown" for point in points)
    winners = Counter(point.get("winner_player") or point.get("winner") or "unknown" for point in points)
    stroke_sides = Counter(
        shot.get("stroke_side") or "unknown"
        for shot in shots
        if shot.get("stroke_confidence") in {"high", "medium"}
    )
    review_points = sum(1 for point in points if point.get("point_review_flags"))
    return {
        "serve_states": serve_states,
        "end_types": end_types,
        "winners": winners,
        "stroke_sides": stroke_sides,
        "avg_speed": avg_speed,
        "high_speed_count": len(high_speed_shots),
        "live_bounces": sum(1 for bounce in bounces if bounce.get("live")),
        "excluded_bounces": sum(1 for bounce in bounces if not bounce.get("live")),
        "review_points": review_points,
    }


def compact_report_data(analysis):
    stats = stat_counts(analysis)
    shot_lookup = by_id(analysis.get("shots") or [])
    bounce_lookup = by_id(analysis.get("bounces") or [])
    points = []
    for point in analysis.get("points") or []:
        points.append(
            {
                "index": point.get("index"),
                "frames": [point.get("start_frame"), point.get("end_frame")],
                "score_before": point.get("point_score_before"),
                "score_after": point.get("score_after"),
                "game_score_after": point.get("game_score_after"),
                "set_score_after": point.get("set_score_after"),
                "server_player": point.get("server_player"),
                "receiver_player": point.get("receiver_player"),
                "winner_player": point.get("winner_player"),
                "winner_source": point.get("winner_source"),
                "serve_state": point.get("serve_state"),
                "serve_confidence": point.get("serve_confidence"),
                "serve_reasons": point.get("serve_reasons") or [],
                "point_end_type": point.get("point_end_type"),
                "point_end_confidence": point.get("point_end_confidence"),
                "point_end_reasons": point.get("point_end_reasons") or [],
                "review_flags": point.get("point_review_flags") or [],
                "shot_ids": point.get("shot_ids") or [],
                "bounce_ids": point.get("bounce_ids") or [],
            }
        )
    return {
        "summary": analysis.get("summary") or {},
        "bounce_source": analysis.get("bounce_source"),
        "stats": {
            "serve_states": dict(stats["serve_states"]),
            "point_endings": dict(stats["end_types"]),
            "points_won": dict(stats["winners"]),
            "trusted_stroke_sides": dict(stats["stroke_sides"]),
            "avg_trusted_speed_mph": round(stats["avg_speed"], 1) if stats["avg_speed"] is not None else None,
            "high_speed_shot_count": stats["high_speed_count"],
            "live_bounces": stats["live_bounces"],
            "excluded_bounces": stats["excluded_bounces"],
            "review_points": stats["review_points"],
        },
        "points": points,
        "shots": [
            {
                "id": shot.get("id"),
                "frame": shot.get("frame"),
                "player": shot.get("player"),
                "type": shot.get("type"),
                "serve_attempt": shot.get("serve_attempt"),
                "stroke_side": shot.get("stroke_side"),
                "stroke_confidence": shot.get("stroke_confidence"),
                "speed_mph": (shot.get("speed") or {}).get("mph"),
                "speed_quality": (shot.get("speed") or {}).get("quality"),
            }
            for shot in shot_lookup.values()
        ],
        "bounces": [
            {
                "id": bounce.get("id"),
                "frame": bounce.get("frame"),
                "point_index": bounce.get("point_index"),
                "side": bounce.get("side"),
                "pattern": bounce.get("pattern"),
                "world_point": bounce.get("world_point"),
                "live": bounce.get("live"),
                "exclude_reason": bounce.get("exclude_reason"),
                "detector_confidence": detector_value(
                    analysis, bounce, "detector_confidence"
                ),
                "detector_shape_confidence": detector_value(
                    analysis, bounce, "detector_shape_confidence"
                ),
                "detector_provenance": detector_value(analysis, bounce, "provenance"),
                "detector_near_player": detector_value(analysis, bounce, "near_player"),
                "detector_rally_scoring_eligible": detector_value(
                    analysis, bounce, "rally_scoring_eligible"
                ),
                "detector_serve_landing_precondition": detector_value(
                    analysis, bounce, "serve_landing_precondition"
                ),
                "review_reasons": bounce.get("review_reasons") or [],
            }
            for bounce in bounce_lookup.values()
        ],
        "review": {
            "missed_bounce_candidates": analysis.get("missed_bounce_candidates") or [],
            "excluded_bounces": analysis.get("excluded_bounces") or [],
        },
    }


def render_count_table(title, counts):
    rows = "\n".join(
        f"<tr><td>{esc(label)}</td><td class=\"num\">{count}</td></tr>"
        for label, count in sorted(counts.items())
    )
    if not rows:
        rows = '<tr><td colspan="2" class="muted">none</td></tr>'
    return f"""
    <section>
      <h2>{esc(title)}</h2>
      <table>
        <thead><tr><th>Label</th><th>Count</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def render_point_rows(points, debug_dir, output_path):
    rows = []
    for point in points:
        conf = point.get("point_end_confidence")
        serve_conf = point.get("serve_confidence")
        rows.append(
            f"""
            <tr>
              <td class="num">{point.get('index')}</td>
              <td>{esc(point.get('point_score_before'))}</td>
              <td>{esc(point.get('score_after'))}</td>
              <td>{player_label(point.get('server_player'))}</td>
              <td>{player_label(point.get('winner_player'))}</td>
              <td>{esc(point.get('winner_source'))}</td>
              <td><span class="pill {confidence_class(serve_conf)}">{esc(point.get('serve_state'))}</span></td>
              <td><span class="pill {confidence_class(conf)}">{esc(point.get('point_end_type'))}</span></td>
              <td>{format_list(point.get('point_review_flags'))}</td>
              <td>{point_debug_link(point, debug_dir, output_path)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_shot_rows(analysis):
    rows = []
    links_by_shot = {link.get("shot_id"): link for link in analysis.get("shot_bounce_links", [])}
    for shot in analysis.get("shots") or []:
        speed = shot.get("speed") or {}
        link = links_by_shot.get(shot.get("id")) or {}
        rows.append(
            f"""
            <tr>
              <td>{esc(shot.get('id'))}</td>
              <td class="num">{shot.get('frame')}</td>
              <td class="num">{link.get('point_index', '')}</td>
              <td>{esc(shot.get('player'))}</td>
              <td>{esc(display_shot_type(shot.get('type')))}</td>
              <td>{esc(shot.get('stroke_side'))}</td>
              <td><span class="pill {confidence_class(shot.get('stroke_confidence'))}">{esc(shot.get('stroke_confidence'))}</span></td>
              <td class="num">{esc(speed.get('mph'))}</td>
              <td><span class="pill {confidence_class(speed.get('quality'))}">{esc(speed.get('quality'))}</span></td>
              <td>{esc(link.get('bounce_id'))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_bounce_rows(analysis):
    rows = []
    for bounce in analysis.get("bounces") or []:
        world = bounce.get("world_point") or ["", ""]
        rows.append(
            f"""
            <tr>
              <td>{esc(bounce.get('id'))}</td>
              <td class="num">{bounce.get('frame')}</td>
              <td class="num">{bounce.get('point_index') or ''}</td>
              <td>{esc(bounce.get('side'))}</td>
              <td>{esc(bounce.get('pattern'))}</td>
              <td class="num">{esc(world[0])}</td>
              <td class="num">{esc(world[1])}</td>
              <td>{'yes' if bounce.get('live') else 'no'}</td>
              <td>{esc(bounce.get('exclude_reason'))}</td>
              <td>{format_list((bounce.get('dead_ball_reasons') or []) + (bounce.get('review_reasons') or []))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def render_review_rows(points):
    rows = []
    for point in points:
        flags = point.get("point_review_flags") or []
        if not flags:
            continue
        rows.append(
            f"""
            <tr>
              <td class="num">{point.get('index')}</td>
              <td>{format_list(flags)}</td>
              <td>{format_list(point.get('point_end_reasons'))}</td>
              <td>{format_list(point.get('serve_reasons'))}</td>
            </tr>
            """
        )
    if not rows:
        return '<tr><td colspan="4" class="muted">No point-level review flags.</td></tr>'
    return "\n".join(rows)


def render_report(analysis, output_path, title, debug_dir, data_json_path=""):
    summary = analysis.get("summary") or {}
    points = analysis.get("points") or []
    stats = stat_counts(analysis)
    court_map_path = output_path.replace("match_report.html", "ai9.5.court_map.png")
    court_map = rel_link(court_map_path, output_path) if os.path.exists(court_map_path) else ""
    data_json = rel_link(data_json_path, output_path) if data_json_path else ""
    avg_speed = "n/a"
    if stats["avg_speed"] is not None:
        avg_speed = f"{stats['avg_speed']:.1f} mph"
    point_rows = render_point_rows(points, debug_dir, output_path)
    shot_rows = render_shot_rows(analysis)
    bounce_rows = render_bounce_rows(analysis)
    review_rows = render_review_rows(points)
    debug_cards = "\n".join(point_debug_card(point, debug_dir, output_path) for point in points)
    data_link = f'<a href="{esc(data_json)}">match_report_data.json</a>' if data_json else '<span class="muted">not exported</span>'
    court_map_section = ""
    if court_map:
        court_map_section = f"""
        <section>
          <h2>Court Map</h2>
          <a href="{esc(court_map)}"><img class="court-map" src="{esc(court_map)}" alt="Court map with bounces"></a>
        </section>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f4;
      --ink: #1c241f;
      --muted: #66706a;
      --line: #d7ddd6;
      --panel: #ffffff;
      --good: #16784a;
      --mid: #9b6b00;
      --bad: #ad2d2d;
      --accent: #245b8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #eef2ed;
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    main {{ padding: 24px 32px 40px; }}
    section {{ margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
    }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .tables {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #eef2ed; font-size: 12px; color: #39433d; }}
    tr:last-child td {{ border-bottom: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #edf0ed;
      color: #39433d;
      white-space: nowrap;
      margin: 0 4px 4px 0;
      font-size: 12px;
    }}
    .pill.good {{ background: #dff2e8; color: var(--good); }}
    .pill.mid {{ background: #fff2cf; color: var(--mid); }}
    .pill.bad, .pill.warn {{ background: #f9dddd; color: var(--bad); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wide {{ overflow-x: auto; }}
    .report-links {{ margin-top: 10px; display: flex; gap: 14px; flex-wrap: wrap; }}
    .court-map {{ max-width: 100%; width: 720px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }}
    .thumbs {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .thumb {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
      color: var(--ink);
    }}
    .thumb img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; }}
    .thumb span {{ display: block; padding: 8px 10px; font-weight: 600; }}
    @media (max-width: 1100px) {{
      .grid, .tables, .thumbs {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 720px) {{
      main, header {{ padding-left: 16px; padding-right: 16px; }}
      .grid, .tables, .thumbs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="muted">Source: {esc(analysis.get('source_jsonl'))}</div>
    <div class="report-links">
      <span>Structured data: {data_link}</span>
    </div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Final point score</div><div class="metric">{esc(summary.get('final_point_score'))}</div></div>
      <div class="card"><div class="label">Final game score</div><div class="metric">{esc(summary.get('final_game_score'))}</div></div>
      <div class="card"><div class="label">Final set score</div><div class="metric">{esc(summary.get('final_set_score'))}</div></div>
      <div class="card"><div class="label">Avg trusted speed</div><div class="metric">{esc(avg_speed)}</div></div>
      <div class="card"><div class="label">Points</div><div class="metric">{esc(summary.get('points'))}</div></div>
      <div class="card"><div class="label">Live bounces</div><div class="metric">{stats['live_bounces']}</div></div>
      <div class="card"><div class="label">Shots</div><div class="metric">{esc(summary.get('shots'))}</div></div>
      <div class="card"><div class="label">Review points</div><div class="metric">{stats['review_points']}</div></div>
    </section>

    {court_map_section}

    <section>
      <h2>Point Debug Maps</h2>
      <div class="thumbs">{debug_cards}</div>
    </section>

    <section>
      <h2>Point By Point</h2>
      <div class="wide">
        <table>
          <thead><tr><th>#</th><th>Before</th><th>After</th><th>Server</th><th>Winner</th><th>Winner source</th><th>Serve</th><th>End</th><th>Review flags</th><th>Debug</th></tr></thead>
          <tbody>{point_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="tables">
      {render_count_table('Serve States', stats['serve_states'])}
      {render_count_table('Point Endings', stats['end_types'])}
      {render_count_table('Points Won', stats['winners'])}
      {render_count_table('Trusted Stroke Sides', stats['stroke_sides'])}
    </section>

    <section>
      <h2>Review Flags</h2>
      <table>
        <thead><tr><th>Point</th><th>Flags</th><th>Point end reasons</th><th>Serve reasons</th></tr></thead>
        <tbody>{review_rows}</tbody>
      </table>
    </section>

    <section>
      <h2>Shots</h2>
      <div class="wide">
        <table>
          <thead><tr><th>ID</th><th>Frame</th><th>Point</th><th>Player</th><th>Type</th><th>Stroke</th><th>Stroke conf</th><th>MPH</th><th>Speed conf</th><th>Bounce</th></tr></thead>
          <tbody>{shot_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Bounces</h2>
      <div class="wide">
        <table>
          <thead><tr><th>ID</th><th>Frame</th><th>Point</th><th>Side</th><th>Pattern</th><th>World X</th><th>World Y</th><th>Live</th><th>Exclude</th><th>Review</th></tr></thead>
          <tbody>{bounce_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main():
    args = parse_args()
    analysis = load_json(args.analysis)
    ensure_parent(args.output)
    if args.data_json:
        ensure_parent(args.data_json)
        with open(args.data_json, "w", encoding="utf-8") as handle:
            json.dump(compact_report_data(analysis), handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.data_json}")
    html_text = render_report(analysis, args.output, args.title, args.point_debug_dir, args.data_json)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(html_text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
