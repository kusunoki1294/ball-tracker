"""Validate the manifest-driven tennis pipeline runner's render freshness guard."""

import contextlib
import io
import json
import os
import sys
import tempfile
from unittest import mock

import run_tennis_pipeline


def touch(path, mtime):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(path)
    os.utime(path, (mtime, mtime))


def main():
    errors = []
    command = run_tennis_pipeline.render_command({
        "video": "source.mp4", "output": "review.mp4",
        "review_all_candidates": True,
    }, "analysis.json")
    if "--review-all-candidates" not in command:
        errors.append("manifest review_all_candidates must reach renderer command")
    default_command = run_tennis_pipeline.render_command(
        {"video": "source.mp4", "output": "review.mp4"}, "analysis.json"
    )
    if "--review-all-candidates" in default_command:
        errors.append("review-all mode must remain opt-in")
    with mock.patch.object(
        run_tennis_pipeline,
        "video_frame_count",
        side_effect=lambda path: {"input.mp4": 10, "output.mp4": 10}.get(path),
    ):
        if run_tennis_pipeline.render_media_message("output.mp4", "input.mp4") is not None:
            errors.append("equal render/input frame counts must pass")
        run_tennis_pipeline.video_frame_count.side_effect = lambda path: {
            "input.mp4": 10, "output.mp4": 9
        }.get(path)
        mismatch = run_tennis_pipeline.render_media_message("output.mp4", "input.mp4")
        if not mismatch or "9 decoded frames" not in mismatch or "10" not in mismatch:
            errors.append(f"frame-count mismatch is not actionable: {mismatch!r}")
        run_tennis_pipeline.video_frame_count.side_effect = lambda path: None
        unavailable = run_tennis_pipeline.render_media_message("output.mp4", "input.mp4")
        if not unavailable or "could not verify decoded frame counts" not in unavailable:
            errors.append(f"unprobeable media must be reported: {unavailable!r}")

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
                "renders": [{
                    "name": "review", "output": render, "analysis": analysis,
                    "review_all_candidates": True,
                }],
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
