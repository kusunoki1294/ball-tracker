"""Run the project validation suite with sensible defaults.

The individual validators remain the source of truth. This runner gives humans
and agents one command for the checks that matter before showing the current
tennis demo, with an optional full mode for the slower timeline pipeline check.
"""

import argparse
import os
import subprocess
import sys
import time


CORE_CHECKS = (
    ("demo artifacts", ("validate_demo_artifacts.py",)),
    ("tennis9 regression", ("validate_tennis9_regression.py",)),
    ("serve detection", ("validate_serve_detection.py",)),
    ("court geometry", ("validate_court_geometry.py",)),
    ("bounce labels", ("eval_bounce_detect.py", "--check-labels")),
    ("association audit", ("validate_association_audit.py",)),
    ("player box audit", ("validate_player_box_audit.py",)),
)

FULL_CHECKS = (
    ("timeline pipeline", ("validate_timeline_pipeline.py",)),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ball-tracker validation checks.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run slower regeneration-based checks such as validate_timeline_pipeline.py.",
    )
    return parser.parse_args()


def python_executable():
    venv_python = os.path.join(".venv", "bin", "python")
    return venv_python if os.path.exists(venv_python) else sys.executable


def run_check(python, name, args):
    command = [python, *args]
    started = time.time()
    print(f"\n== {name} ==", flush=True)
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, text=True)
    elapsed = time.time() - started
    if completed.returncode != 0:
        print(f"{name} failed after {elapsed:.1f}s", file=sys.stderr)
        return completed.returncode
    print(f"{name} passed in {elapsed:.1f}s", flush=True)
    return 0


def main():
    args = parse_args()
    python = python_executable()
    checks = list(CORE_CHECKS)
    if args.full:
        checks.extend(FULL_CHECKS)
    failures = []
    started = time.time()
    for name, command in checks:
        status = run_check(python, name, command)
        if status:
            failures.append((name, status))
            break
    elapsed = time.time() - started
    if failures:
        name, status = failures[0]
        print(f"\nvalidation stopped at {name} (exit {status}) after {elapsed:.1f}s", file=sys.stderr)
        return status
    mode = "full" if args.full else "core"
    print(f"\n{mode} validation passed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
