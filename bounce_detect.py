"""Detect ball bounces from a track_ball_yolo JSONL, offline.

The in-tracker EventDetector decides bounces live, from a short window of
consecutive frames. That works when the ball is tracked continuously, but the
ball track breaks up exactly at the bounce -- the ball is at its fastest, most
motion-blurred and lowest-contrast against the court there -- so on harder
footage most bounces never get a clean enough window and are simply lost
(tennis11 game 1: 10 bounces across 6 points, one point with none at all).

Working offline from the logged track removes that constraint: the whole
trajectory is available at once, so a bounce can be found by fitting the arc on
either side of a candidate instant rather than requiring consecutive samples.

What a bounce looks like, measured on the 23 validated tennis9 bounces:

* Vertical image velocity is positive before (the ball is descending in frame)
  and <= 0 after, without exception.
* The drop in vertical velocity (`dvy`) ranges from 4.4 to 37.1 px/frame.
* Far-court bounces are much weaker than near-court ones (dvy 5-15 vs 19-35)
  purely because of perspective, so thresholds are normalised to feet using the
  court homography instead of being applied in pixels.

Racket hits share the "descending then rising" shape and can have a LARGER dvy
than a bounce (measured up to 66), so dvy alone cannot separate them.

Nothing here rejects racket contacts, because in image space they are not
reliably separable and every attempt measured worse than useless:

* "the ball leaves harder off a racket" holds on tennis9 but inverts on
  tennis11. Which end serves decides whether a struck ball travels toward or
  away from the camera, and image-y velocity mixes depth with height, so a real
  far-court serve bounce reads as -10.6 ft/frame where tennis9 bounces never
  passed -4.1. A threshold tuned on one clip rejected the other clip's serve.
* ball-size trend (growing/shrinking) separated 16/23 bounces from 7/16 hits.
* distance to the striker overlaps: bounces sit at 0ft from a player about as
  often as contacts do, because players stand where balls land.

So contacts are FLAGGED, never dropped. Each bounce carries `near_player`, a
`shape_confidence` (trajectory evidence alone) and a `confidence` (that graded
down near a player), plus two caller contracts: `rally_scoring_eligible` for
rally scoring, which excludes anything near a player, and
`serve_landing_precondition` for the serve path, which allows a receiver-side
bounce near the waiting receiver because that is where serves land. The latter
is a permissive precondition rather than an eligibility verdict -- it filters
almost nothing by itself and must be combined with contact anchoring and
receiver-side geometry. Getting
this right needs the ball's height, which one calibrated camera cannot give
without a full ballistic reconstruction.
"""
import numpy as np
import cv2

CORNERS_WORLD = np.array([[0.0, 78.0], [36.0, 78.0], [36.0, 0.0], [0.0, 0.0]], dtype=np.float32)
COURT_NET_Y_FT = 39.0
COURT_LENGTH_FT = 78.0
COURT_WIDTH_FT = 36.0

# Tuned on tennis9's validated bounces; see module docstring for the measurements.
DEFAULTS = {
    "window": 10,            # frames each side used for the local arc fit
    "min_samples": 4,        # samples required on each side
    # Frames the samples must span on each side. Was 4, which silently dropped
    # tennis11's P3 first serve: the track there is clumpy (4 samples crammed
    # into 3 frames), and 4 samples already determine a quadratic regardless of
    # how tightly they sit. Audited at 2/3/4 -- tennis9: detected 44/44/43,
    # recall 20/23 at all three; tennis11: detected 44/44/41, loose serve
    # bounces 6/6/5, strict 5 at all three. Spans 2 and 3 are EQUIVALENT on both
    # clips, so 3 is a degeneracy guard (two samples must not set a velocity),
    # not a measured improvement over 2.
    "min_span": 3,
    "max_gap": 12,           # a longer hole than this splits the trajectory
    "min_dvy_ft": 0.8,       # minimum drop in vertical velocity, feet/frame
    "max_vy_after_ft": 0.35, # ball must not still be descending after
    # No racket-rejection constant exists: contacts are graded and flagged, never
    # dropped (see the module docstring for the three separation attempts that
    # measured worse than useless). near_player_ft below only sets the flag.
    # Residual is judged RELATIVE to how fast the ball is moving: a fast ball
    # covers more pixels between samples and so fits less tightly, and an
    # absolute pixel limit silently discards exactly the fastest events (serve
    # bounces). Absolute floor keeps it sane when the ball is nearly still.
    "max_residual_ratio": 0.85,
    # The one absolute-pixel gate that survives, and it is kept on evidence
    # rather than habit: audited at 18 / 30 / 60 / disabled, tennis9 recall stays
    # 20/23 and tennis11 strict serve bounces stay 5/7 at every setting, while
    # detections grow 43 -> 51. It costs no known-good event and only suppresses
    # noise, which is the opposite of the join-error and velocity caps that were
    # removed for silently discarding the fastest real bounces. Re-audit it
    # before trusting it on a third clip.
    "max_residual_px": 18.0,
    # When the bounce frame itself was not tracked, the two arcs must still meet
    # there. A wide disagreement means these are not two halves of one bounce.
    # Interpolated candidates must still have both arcs meeting at the bounce,
    # measured in frames of ball travel so a fast serve bounce is not discarded
    # for covering more pixels between samples.
    "max_join_error_frames": 3.5,
    # Arc agreement is measured in FRAMES OF BALL TRAVEL, not pixels, for the
    # same reason as the residual: a serve bounce moves ~195px/frame, so a 259px
    # gap there is 1.3 frames -- tight -- while the same 259px on a drifting ball
    # is a different event entirely. An absolute pixel cap graded exactly the
    # fastest, most valuable bounces as noise.
    "join_error_high_frames": 1.5,
    "join_error_low_frames": 3.5,
    "suppress": 18,          # frames; keeps genuine double bounces (~20 apart) distinct
    # Sanity cap only. This was 12.0 and silently cost a real serve bounce:
    # tennis11's P4 serve lands at 11.95 ft/frame measured at the contact point,
    # which tips over 12 once scaled at the ball centre. Serve bounces are the
    # fastest events in the clip, so a tight cap removes exactly what matters.
    "max_vy_before_ft": 20.0,
    # Allow a little outside the lines so genuinely-out balls are still found,
    # but not the metres-outside projections an airborne ball produces.
    "court_margin_ft": 3.0,
    "near_player_ft": 3.0,   # within this of a striker, a bounce is not separable from a contact
}


class PerspectiveScale:
    """Pixels-per-foot as a function of height in the image.

    Near-court motion covers several times more pixels per foot than far-court
    motion, so raw pixel velocities are not comparable between the two ends.
    Sampling the court centreline gives a scale curve to divide them by.
    """

    def __init__(self, calib_points):
        image = np.array(calib_points, dtype=np.float32)
        self.homography = cv2.getPerspectiveTransform(CORNERS_WORLD, image)
        ys = np.linspace(0.5, COURT_LENGTH_FT - 0.5, 160)
        centre = np.array([[[COURT_WIDTH_FT / 2.0, y] for y in ys]], dtype=np.float32)
        step = np.array([[[COURT_WIDTH_FT / 2.0, y + 1.0] for y in ys]], dtype=np.float32)
        centre_img = cv2.perspectiveTransform(centre, self.homography)[0]
        step_img = cv2.perspectiveTransform(step, self.homography)[0]
        # px covered by one foot of court length at each sampled depth
        self._image_y = centre_img[:, 1]
        self._px_per_ft = np.linalg.norm(step_img - centre_img, axis=1)
        order = np.argsort(self._image_y)
        self._image_y = self._image_y[order]
        self._px_per_ft = np.maximum(self._px_per_ft[order], 1e-3)

    def at(self, image_y):
        return float(np.interp(image_y, self._image_y, self._px_per_ft))


def _arc_fit(samples, at_frame):
    """Fit the ball's arc and report its velocity at `at_frame`.

    Between contacts the ball follows a ballistic arc, so a quadratic is the
    right model; fitting a straight line instead pushes the curvature into the
    residual and makes a clean arc look like noise. Velocity is evaluated at the
    candidate instant rather than averaged over the window, which sharpens the
    discontinuity a bounce creates.

    Returns (vx, vy, residual_px, count, span, x_at, y_at) or None, where
    x_at/y_at are where this arc puts the ball at `at_frame` -- needed when the
    ball was not tracked on the bounce frame itself.
    """
    if len(samples) < 3:
        return None
    frames = np.array([s[0] for s in samples], dtype=float)
    span = frames.max() - frames.min()
    if span <= 0:
        return None
    t = frames - at_frame
    xs = np.array([s[1] for s in samples], dtype=float)
    ys = np.array([s[2] for s in samples], dtype=float)
    degree = 2 if len(samples) >= 4 else 1
    cx = np.polyfit(t, xs, degree)
    cy = np.polyfit(t, ys, degree)
    # derivative at t=0 (i.e. at at_frame) is the last-but-one coefficient
    vx = cx[-2]
    vy = cy[-2]
    residual = float(np.sqrt(np.mean((ys - np.polyval(cy, t)) ** 2)
                             + np.mean((xs - np.polyval(cx, t)) ** 2)))
    return (float(vx), float(vy), residual, len(samples), float(span),
            float(cx[-1]), float(cy[-1]))


def ball_samples(rows):
    """(frame, centre_x, centre_y, contact_x, contact_y) for every tracked ball."""
    out = []
    for row in rows:
        ball = row.get("ball")
        if not ball:
            continue
        centre = ball.get("center")
        if not centre:
            continue
        bbox = ball.get("bbox")
        contact = (centre[0], bbox[3]) if bbox else tuple(centre)
        out.append((int(row["frame"]), float(centre[0]), float(centre[1]),
                    float(contact[0]), float(contact[1])))
    out.sort(key=lambda item: item[0])
    return out


def _strike_distance(row, point):
    """Pixel distance from `point` to the NEAREST player's box, or None.

    Both players are checked deliberately. Selecting one by the ball's projected
    court side looks natural and is wrong: a ball at the near server's racket is
    airborne, so the ground homography places it in the far court, and comparing
    it against the far player reports it as nowhere near anybody. That is how a
    serve strike came through flagged near_player=False and would have been
    anchored as its own serve's landing. Reach is about the image, so ask the
    image, not the projection.
    """
    distances = [d for d in (_box_distance(row.get("player_near"), point),
                             _box_distance(row.get("player_far"), point))
                 if d is not None]
    return min(distances) if distances else None


def _box_distance(player, point):
    if not player or not player.get("bbox"):
        return None
    x0, y0, x1, y1 = player["bbox"]
    # Widen the box to a reach: a racket extends well beyond the body.
    pad_x = (x1 - x0) * 0.6
    pad_y = (y1 - y0) * 0.25
    dx = max(x0 - pad_x - point[0], 0.0, point[0] - (x1 + pad_x))
    dy = max(y0 - pad_y - point[1], 0.0, point[1] - (y1 + pad_y))
    return float(np.hypot(dx, dy))


def detect_bounces(rows, calib_points, params=None):
    """Find ball bounces in a tracking log. Returns a list of bounce dicts."""
    cfg = dict(DEFAULTS)
    cfg.update(params or {})

    scale = PerspectiveScale(calib_points)
    inv_homography = np.linalg.inv(scale.homography)
    samples = ball_samples(rows)
    if len(samples) < 2 * cfg["min_samples"]:
        return []
    rows_by_frame = {int(r["frame"]): r for r in rows}
    by_frame = {s[0]: s for s in samples}
    frames = [s[0] for s in samples]

    # Split where the track has a hole too long to reason across.
    segments, current = [], [frames[0]]
    for previous, frame in zip(frames, frames[1:]):
        if frame - previous > cfg["max_gap"]:
            segments.append(current)
            current = []
        current.append(frame)
    segments.append(current)

    candidates = []
    for segment in segments:
        if len(segment) < 2 * cfg["min_samples"]:
            continue
        lo, hi = segment[0], segment[-1]
        for frame in range(lo + cfg["min_span"], hi - cfg["min_span"] + 1):
            before = [by_frame[f][:3] for f in segment if frame - cfg["window"] <= f <= frame - 1]
            after = [by_frame[f][:3] for f in segment if frame + 1 <= f <= frame + cfg["window"]]
            if len(before) < cfg["min_samples"] or len(after) < cfg["min_samples"]:
                continue
            fit_before = _arc_fit(before, frame)
            fit_after = _arc_fit(after, frame)
            if not fit_before or not fit_after:
                continue
            vx_b, vy_b, res_b, _, span_b, x_b, y_b = fit_before
            vx_a, vy_a, res_a, _, span_a, x_a, y_a = fit_after
            if span_b < cfg["min_span"] or span_a < cfg["min_span"]:
                continue

            # The bounce instant is often one of the frames the tracker dropped
            # -- it is the blurriest moment in the rally -- so do not require a
            # sample there. Both arcs predict where the ball is at this instant,
            # and at a real bounce they must agree: that agreement doubles as a
            # quality check.
            here = by_frame.get(frame)
            join_error_px = float(np.hypot(x_b - x_a, y_b - y_a))
            # Normalise immediately: speed is already known from the two fits, and
            # gating on raw pixels here would drop exactly the fast interpolated
            # serve bounces this branch exists to rescue.
            # Use TOTAL image speed, not vertical: a crosscourt ball carries most
            # of its travel in x, and dividing by the vertical component alone
            # would inflate its normalised errors and grade it down for moving
            # sideways.
            speed_px = max(float(np.hypot(vx_b, vy_b)), float(np.hypot(vx_a, vy_a)), 1e-3)
            join_error_frames = join_error_px / speed_px
            if here is not None:
                centre = (here[1], here[2])
                contact = (here[3], here[4])
                provenance = "sampled"
            else:
                if join_error_frames > cfg["max_join_error_frames"]:
                    continue
                centre = ((x_b + x_a) / 2.0, (y_b + y_a) / 2.0)
                # Carry the bbox-bottom offset over from the nearest real
                # samples so the contact point stays on the ball's underside.
                near_before = max((f for f in segment if f < frame), default=None)
                near_after = min((f for f in segment if f > frame), default=None)
                offsets = [by_frame[f][4] - by_frame[f][2]
                           for f in (near_before, near_after) if f is not None]
                drop = sum(offsets) / len(offsets) if offsets else 0.0
                contact = (centre[0], centre[1] + drop)
                provenance = "interpolated"
            px_per_ft = scale.at(centre[1])

            vy_b_ft = vy_b / px_per_ft
            vy_a_ft = vy_a / px_per_ft
            dvy_ft = vy_b_ft - vy_a_ft
            residual_px = max(res_b, res_a)

            if vy_b_ft <= 0:                       # must be descending in frame
                continue
            if vy_b_ft > cfg["max_vy_before_ft"]:  # impossible speed: bad track
                continue
            if vy_a_ft > cfg["max_vy_after_ft"]:   # must stop descending
                continue
            if dvy_ft < cfg["min_dvy_ft"]:
                continue
            if residual_px > cfg["max_residual_px"]:
                continue
            if residual_px / speed_px > cfg["max_residual_ratio"]:  # jitter, not an arc
                continue

            world = cv2.perspectiveTransform(
                np.array([[contact]], dtype=np.float32), inv_homography)[0][0]
            world_point = (float(world[0]), float(world[1]))
            margin = cfg["court_margin_ft"]
            if not (-margin <= world_point[0] <= COURT_WIDTH_FT + margin
                    and -margin <= world_point[1] <= COURT_LENGTH_FT + margin):
                continue
            side = "near" if world_point[1] > COURT_NET_Y_FT else "far"

            row = rows_by_frame.get(frame, {})
            reach_px = _strike_distance(row, centre)
            reach_ft = reach_px / px_per_ft if reach_px is not None else None

            # Confidence, not rejection. How well the two arcs meet at the
            # candidate instant separates real bounces from coincidences well
            # (validated bounces sit at p90=41px, unmatched extras at p90=200),
            # but gating on it costs real bounces, so it grades instead.
            # `shape_confidence` grades the trajectory evidence alone -- is this a
            # real reversal, or noise? `confidence` then folds in player
            # proximity, which is about WHAT the event is rather than whether it
            # happened. Keeping them apart matters because a serve landing next
            # to the waiting receiver is a perfectly good bounce that a single
            # blended grade would bury.
            near_player = bool(reach_ft is not None and reach_ft <= cfg["near_player_ft"])
            if join_error_frames > cfg["join_error_low_frames"]:
                shape_confidence = "low"
            elif (dvy_ft >= 2.0 and residual_px / speed_px <= 0.5 and provenance == "sampled"
                  and join_error_frames <= cfg["join_error_high_frames"]):
                shape_confidence = "high"
            else:
                shape_confidence = "medium"
            confidence = "low" if near_player else shape_confidence

            candidates.append({
                "frame": frame,
                "provenance": provenance,
                "join_error_px": round(join_error_px, 2),
                "join_error_frames": round(join_error_frames, 2),
                "point": [round(contact[0], 1), round(contact[1], 1)],
                "world_point": [round(world_point[0], 2), round(world_point[1], 2)],
                "side": side,
                "vy_before_ft": round(vy_b_ft, 3),
                "vy_after_ft": round(vy_a_ft, 3),
                "dvy_ft": round(dvy_ft, 3),
                "residual_px": round(residual_px, 2),
                "player_distance_px": round(reach_px, 1) if reach_px is not None else None,
                "player_distance_ft": round(reach_ft, 2) if reach_ft is not None else None,
                "score": round(dvy_ft / max(residual_px / max(px_per_ft, 1e-3), 0.02), 2),
                # Near a player this may be a racket contact rather than a ground
                # bounce. Image-space velocity cannot settle that on its own (a
                # far-court bounce of a ball travelling away rebounds as steeply
                # as a smash), so flag it for the caller instead of guessing.
                "near_player": near_player,
                "shape_confidence": shape_confidence,
                "confidence": confidence,
                # Two contracts, because they need different things.
                # Rally scoring stays conservative: away from any player, so a
                # racket contact cannot be mistaken for a ground bounce.
                "rally_scoring_eligible": confidence in ("high", "medium") and not near_player,
                # A PRECONDITION, not a verdict, and named so it cannot be read
                # as one: it says only "the trajectory evidence is not weak", and
                # on tennis11 that is true of every detection (41/41), so it
                # filters almost nothing on its own. It exists because the rally
                # contract excludes anything near a player, which would throw away
                # serve landings beside the waiting receiver. A serve path must
                # still add contact anchoring, receiver-side geometry and a
                # serve-flight window on top of this.
                "serve_landing_precondition": shape_confidence in ("high", "medium"),
            })

    # Non-maximum suppression: one bounce per rebound. Rank by shape grade first
    # so a well-evidenced candidate is not displaced by a noisier one that
    # happens to score higher, then by score within a grade.
    rank = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda c: (rank[c["shape_confidence"]], -c["score"]))
    kept = []
    for candidate in candidates:
        clash = next((k for k in kept
                      if abs(candidate["frame"] - k["frame"]) <= cfg["suppress"]), None)
        if clash is None:
            candidate["suppressed"] = 0
            kept.append(candidate)
        else:
            # Keep a count so the review CSV can show what was discarded here.
            clash["suppressed"] = clash.get("suppressed", 0) + 1
    kept.sort(key=lambda c: c["frame"])
    for index, candidate in enumerate(kept, start=1):
        candidate["id"] = f"bounce_{index:03d}"
    return kept


def _load_rows(path):
    import json
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    import argparse, json
    parser = argparse.ArgumentParser(description="Detect bounces from a tracking JSONL.")
    parser.add_argument("--jsonl", required=True, help="track_ball_yolo JSONL log.")
    parser.add_argument("--court-calib-file", required=True, help="Court calibration JSON.")
    parser.add_argument("--out", help="Optional output JSON path.")
    args = parser.parse_args()

    rows = _load_rows(args.jsonl)
    calib = json.load(open(args.court_calib_file))["points"]
    bounces = detect_bounces(rows, calib)
    print(f"{len(bounces)} bounces from {len(rows)} frames")
    for bounce in bounces:
        print(f"  f{bounce['frame']:>5} {bounce['side']:>4} "
              f"world=({bounce['world_point'][0]:>5.1f},{bounce['world_point'][1]:>5.1f}) "
              f"dvy={bounce['dvy_ft']:>5.2f} {bounce['confidence']}"
              f"{' near_player' if bounce['near_player'] else ''}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"bounces": bounces}, handle, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
