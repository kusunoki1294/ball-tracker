import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run tennis analysis and all configured render outputs.")
    parser.add_argument("--manifest", required=True, help="Analysis/render manifest JSON.")
    parser.add_argument("--skip-analysis", action="store_true", help="Render from the existing analysis JSON.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip audit CSV/summary/court-map exports.")
    parser.add_argument("--skip-report", action="store_true", help="Skip static HTML report export.")
    parser.add_argument("--skip-render", action="store_true", help="Only regenerate the analysis JSON.")
    return parser.parse_args()


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stale_render_reason(output, analysis):
    if not os.path.exists(output):
        return None
    if not os.path.exists(analysis):
        return f"analysis missing: {analysis}"
    output_mtime = os.path.getmtime(output)
    analysis_mtime = os.path.getmtime(analysis)
    if output_mtime < analysis_mtime:
        return (
            f"{output} is older than {analysis}; rerun without --skip-render "
            "before using the video as a product artifact"
        )
    return None


def warn_stale_renders(jobs):
    stale = []
    for job in jobs:
        output = job.get("output")
        analysis = job.get("analysis")
        if not output or not analysis:
            continue
        reason = stale_render_reason(output, analysis)
        if reason:
            stale.append(reason)
    if stale:
        print("WARNING: render outputs are stale:", file=sys.stderr)
        for reason in stale:
            print(f"  - {reason}", file=sys.stderr)


def render_jobs(manifest):
    jobs = manifest.get("renders") or []
    if not isinstance(jobs, list):
        raise ValueError("manifest field 'renders' must be a list")
    return jobs


def audit_config(manifest):
    audit = manifest.get("audit")
    if audit is None:
        return None
    if not isinstance(audit, dict):
        raise ValueError("manifest field 'audit' must be an object")
    return audit


def report_config(manifest):
    report = manifest.get("report")
    if report is None:
        return None
    if not isinstance(report, dict):
        raise ValueError("manifest field 'report' must be an object")
    return report


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    analysis_path = manifest.get("output")
    if not analysis_path:
        raise ValueError("manifest must define an analysis 'output' path")

    if not args.skip_analysis:
        run_command([sys.executable, "analyze_tennis_events.py", "--manifest", args.manifest])

    audit = audit_config(manifest)
    if audit and not args.skip_audit:
        csv_path = audit.get("csv")
        summary_json = audit.get("summary_json")
        court_map = audit.get("court_map")
        point_debug_dir = audit.get("point_debug_dir")
        if not csv_path or not summary_json or not court_map:
            raise ValueError("audit config must define 'csv', 'summary_json', and 'court_map'")
        command = [
            sys.executable,
            "export_tennis_audit.py",
            "--analysis",
            analysis_path,
            "--csv",
            csv_path,
            "--summary-json",
            summary_json,
            "--court-map",
            court_map,
        ]
        if point_debug_dir:
            command.extend(["--point-debug-dir", point_debug_dir])
        run_command(command)

    report = report_config(manifest)
    if report and not args.skip_report:
        output = report.get("output")
        if not output:
            raise ValueError("report config must define 'output'")
        command = [
            sys.executable,
            "export_match_report.py",
            "--analysis",
            analysis_path,
            "--output",
            output,
        ]
        title = report.get("title")
        if title:
            command.extend(["--title", title])
        data_json = report.get("data_json")
        if data_json:
            command.extend(["--data-json", data_json])
        point_debug_dir = report.get("point_debug_dir") or (audit or {}).get("point_debug_dir")
        if point_debug_dir:
            command.extend(["--point-debug-dir", point_debug_dir])
        run_command(command)

    if args.skip_render:
        warn_stale_renders(
            [
                {**job, "analysis": job.get("analysis") or analysis_path}
                for job in render_jobs(manifest)
            ]
        )
        return

    jobs = render_jobs(manifest)
    if not jobs:
        raise ValueError("manifest does not define any render jobs")

    for job in jobs:
        video = job.get("video")
        output = job.get("output")
        analysis = job.get("analysis") or analysis_path
        if not video or not output:
            raise ValueError("each render job must define 'video' and 'output'")
        run_command(
            [
                sys.executable,
                "render_tennis_analysis.py",
                "--video",
                video,
                "--analysis",
                analysis,
                "--output",
                output,
            ]
        )
        reason = stale_render_reason(output, analysis)
        if reason:
            raise RuntimeError(f"render freshness check failed: {reason}")


if __name__ == "__main__":
    main()
