"""A solid red bounce marker must mean the bounce itself is well evidenced.

Solid red is the strongest claim the analysis video makes about a bounce. It
should not be reachable by a candidate the detector graded below high, nor by
one with no observation at the bounce instant - where the drawn position is the
arc-join midpoint rather than a sighting.

bounce_022/f1145 was both (medium, interpolated) and rendered solid red, which
is what this check exists to stop recurring.
"""

import json
import sys

import render_tennis_analysis as R

ANALYSIS = "yoloVids/outputs/tennis11/ai11.2.analysis.json"


def main():
    errors = []
    with open(ANALYSIS, encoding="utf-8") as handle:
        analysis = json.load(handle)
    serve_ids = {
        attempt["bounce_id"]
        for point in analysis.get("points", [])
        for attempt in (point.get("serve_analysis") or {}).get("attempts") or []
        if attempt.get("bounce_id")
    }
    drawn = [b for b in analysis.get("bounces", [])
             if b.get("point") and R.analysis_bounce_visible(b, serve_ids)]
    if not drawn:
        return ["no bounce markers are drawn at all; the render contract cannot be checked"]

    solid = [b for b in drawn if not R.bounce_is_provisional(b, serve_ids)]
    for bounce in solid:
        if bounce.get("detector_confidence") != "high":
            errors.append(
                f"{bounce['id']} f{bounce['frame']} draws SOLID at detector_confidence "
                f"{bounce.get('detector_confidence')!r}; solid must mean high"
            )
        if bounce.get("provenance") == "interpolated":
            errors.append(
                f"{bounce['id']} f{bounce['frame']} draws SOLID on an interpolated "
                f"position; there was no observation at the bounce instant"
            )
    if not solid:
        errors.append("every marker is provisional; the qualification is too broad to be useful")

    if errors:
        print("render marker contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"render marker contract validation passed "
          f"({len(solid)} solid, {len(drawn) - len(solid)} provisional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
