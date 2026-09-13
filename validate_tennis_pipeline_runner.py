"""Validate the manifest-driven tennis pipeline runner's render freshness guard."""

import contextlib
import io
import json
import os
import sys
import tempfile

import run_tennis_pipeline


def touch(path, mtime):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(path)
    os.utime(path, (mtime, mtime))


def main():
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        analysis = os.path.join(tmp, "analysis.json")
        render = os.path.join(tmp, "render.mp4")
        touch(analysis, 200)
        touch(render, 300)

        if run_tennis_pipeline.render_staleness_message(render, analysis) is not None:
            errors.append("fresh render must not be reported stale")
        touch(render, 100)
        stale = run_tennis_pipeline.render_staleness_message(render, analysis)
        if not stale or "older than" not in stale or "without --skip-render" not in stale:
            errors.append(f"stale render message is not actionable: {stale!r}")
        os.remove(render)
        if run_tennis_pipeline.render_staleness_message(render, analysis) is not None:
            errors.append("missing render must be quiet when render was skipped")
        missing = run_tennis_pipeline.render_staleness_message(
            render, analysis, missing_is_stale=True
        )
        if not missing or "was not written" not in missing:
            errors.append(f"missing render after normal render must be fatal: {missing!r}")

        manifest_path = os.path.join(tmp, "manifest.json")
        touch(render, 100)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "output": analysis,
                "renders": [{"name": "review", "output": render, "analysis": analysis}],
            }, handle)
        original_argv = sys.argv[:]
        stderr = io.StringIO()
        try:
            sys.argv = [
                "run_tennis_pipeline.py",
                "--manifest",
                manifest_path,
                "--skip-analysis",
                "--skip-audit",
                "--skip-report",
                "--skip-render",
            ]
            with contextlib.redirect_stderr(stderr):
                run_tennis_pipeline.main()
        finally:
            sys.argv = original_argv
        warning = stderr.getvalue()
        if "WARNING: render outputs are stale" not in warning or render not in warning:
            errors.append(f"--skip-render did not warn about stale render: {warning!r}")

    if errors:
        print("tennis pipeline runner validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("tennis pipeline runner validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
