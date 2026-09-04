"""Check the player-box audit reports reliability evidence, not verdicts.

Two signals in this project separated hand-picked controls and then failed
against a random population. This audit therefore has to carry a population
baseline with its n, and must never call a swap a player error, a live rally, or
a point boundary.
"""

import csv
import os
import subprocess
import sys
import tempfile


CLIPS = [
    "game1=yoloVids/outputs/tennis11/ai11.1.jsonl",
    "game2=yoloVids/outputs/tennis11/ai11.g2.jsonl",
]
CONTROLS = [
    "game1=1485=verified serve, receiver window corrupted",
    "game1=786=live-rally control",
    "game2=4183=verified serve, clean",
]
# Hand-checked; see docs/experiments/tennis11_player_box_stability.md.
EXPECTED_RATES = {("game1", "player_far"), ("game1", "player_near"),
                  ("game2", "player_far"), ("game2", "player_near")}
FORBIDDEN_WORDS = ("player_error", "rally_live", "point_start", "point_boundary",
                   "mis-association", "misassociation")
REQUIRED_PHRASES = ("Reliability evidence, not findings",
                    "not a player error",
                    "Random population baseline",
                    "every figure states its n")


def python_bin():
    venv = os.path.join(".venv", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def main():
    errors = []
    warnings = []
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "player_box_audit.csv")
        html_path = os.path.join(tmp, "player_box_audit.html")
        command = [python_bin(), "export_player_box_audit.py",
                   "--output-csv", csv_path, "--output-html", html_path]
        for clip in CLIPS:
            command += ["--clip", clip]
        for control in CONTROLS:
            command += ["--control", control]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            print("player box audit failed to run:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 1
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            warnings.append("player box audit produced no swaps")
        far = sum(1 for row in rows if row["clip"] == "game1" and row["player"] == "player_far")
        near = sum(1 for row in rows if row["clip"] == "game1" and row["player"] == "player_near")
        if far <= near:
            warnings.append(
                "game1 far-player swaps no longer exceed near "
                f"({far} vs {near}); the current tracker observation changed"
            )
        for row in rows:
            blob = " ".join(str(v).lower() for v in row.values())
            hit = [w for w in FORBIDDEN_WORDS if w in blob]
            if hit:
                errors.append(f"{row['clip']} f{row['frame']}: row asserts a verdict ({hit[0]})")
                break
        with open(html_path, encoding="utf-8") as handle:
            page = handle.read()
        for phrase in REQUIRED_PHRASES:
            if phrase not in page:
                errors.append(f"audit HTML missing required text: {phrase!r}")
        # A control table without a population baseline beside it is exactly the
        # mistake that produced two retractions.
        if "n sampled" not in page:
            errors.append("audit HTML must state the population sample size")
    if errors:
        print("player box audit validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"player box audit validation passed ({len(rows)} swaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
