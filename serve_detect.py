"""Locate the serve strike itself, from player and ball motion.

The analyzer used to have no serve detection: it assumed the first bounce of a
point was the serve and judged that bounce against the receiver's service box.
Point ranges run start-to-start, so they include the walk-back and setup time
before the serve, and the "first bounce" is regularly a rally bounce instead --
on tennis11 game 1 that produced serve verdicts built from bounces the server
hit mid-rally.

This module finds the serve the other way round: locate the strike, then let the
caller look for the bounce that follows it. Two cues, in order of trust:

1. The toss. The ball leaves the server's hand, rises nearly vertically to an
   apex above their head, and falls back. Nothing else in a point looks like
   that, and the rise is large: 0.5-0.9 of the server's own bounding-box height.
2. The peak reach. At contact the server is stretched to full extension, so the
   top edge of their bounding box reaches its highest point of the motion. This
   pins the contact frame far more precisely than the ball does, because the
   tracker frequently loses the ball at the moment the racket hits it -- on
   tennis11 point 2 the track carries on descending straight through contact.

So the toss finds the serve and the peak reach times it. When the ball is never
tracked through the toss (tennis11 point 5), the peak reach alone still finds
the serve, at lower confidence.

Every candidate must pass one hard physical gate: a serve is struck from behind
the server's own baseline. That single check is what separates a serve from the
mid-rally overhead that otherwise looks identical -- on tennis11 point 6 a rally
lob was being reported as the serve, and its striker stood at world y=63ft,
15ft inside the baseline the real serves were struck from.

Known limits, in the order they are likely to bite:

* Far-court servers are not supported. Verified on tennis11 (near serves) only;
  on tennis9, whose server is far, the detector finds nothing and the caller
  falls back to the old bounce-first path with `serve_motion_fallback` set. The
  cause is calibration, not thresholds: court_calib_tennis7.json puts the far
  baseline about 45px above the white line actually visible in the frame, so a
  far player standing correctly behind it projects to world y=14-16ft instead of
  ~0 and can never pass the gate. Comparing foot to baseline in image space
  instead of world space was tried and fails the same way, for the same reason.
  Do not widen BEHIND_BASELINE_MARGIN_FT to paper over this -- every value loose
  enough to admit tennis9's far player also readmits tennis11's rally lob.
* Singles only. Both cues read `player_near`/`player_far`, so the extra players
  and merged boxes of a doubles clip would break the association.
* Durations are held in seconds and converted with the clip's fps, so frame
  rates other than 30 are handled, but only 30fps footage has been measured.

Handedness does not matter: nothing here reads stroke side, only the toss and
the box.
"""

import cv2
import numpy as np


NEAR_BASELINE_FT = 78.0
FAR_BASELINE_FT = 0.0
COURT_NET_Y_FT = 39.0
# A serve is struck from behind the baseline (a foot fault is a rule violation),
# but the server's feet project through the ground homography with some error
# and players drift onto the line. Measured serves on tennis11 land at world
# y=78.5-80.7ft, while the mid-rally overhead that used to be mistaken for a
# serve sits at y=63ft, so there is a wide gap to place this in.
BEHIND_BASELINE_MARGIN_FT = 4.0

# Toss geometry, all scaled by the server's bounding-box height so the same
# numbers hold for a near-court server (~450px tall) and a far-court one (~85px).
TOSS_MIN_RISE_BOX_FRAC = 0.45
TOSS_APEX_ABOVE_HEAD_FRAC = 0.12
TOSS_MAX_STEP_DRIFT_BOX_FRAC = 0.07
TOSS_MAX_TOTAL_DRIFT_BOX_FRAC = 0.50
TOSS_RISE_NOISE_PX = 2.0
# The ball leaves the server's own hand, so the rise has to begin at their box.
# Without this the far player inherits the near player's toss: it starts 300px
# below them but still tops out above their head, and they are stood behind
# their own baseline to receive, so every other test passes.
TOSS_START_MAX_DISTANCE_BOX_FRAC = 0.90

# How far the box top must rise above the player's resting pose to count as a
# deliberate full extension rather than tracker jitter. The resting pose is a
# high percentile of the box top over a window either side, rather than the
# nearest few frames: the frames immediately around a serve are the trophy
# position and the follow-through, which are themselves raised, and measuring
# against those hides the very motion being looked for.
REACH_MIN_PROMINENCE_BOX_FRAC = 0.20
REACH_BASELINE_PERCENTILE = 0.75
# The peak-reach fallback has no toss to corroborate it, and the extension it
# looks for is a fraction of a body height. On a box this small that fraction is
# down among the tracker's own jitter -- a receiving far-court player clears
# 0.19-0.21 of their box height just moving around -- so the fallback is only
# trusted where the signal is comfortably larger than the noise.
REACH_MIN_BOX_HEIGHT_PX = 150

# A rally can only be under way if the ball came back over the net. Between a
# fault and the second serve it stays on the receiver's side: measured at 5% of
# tracked frames on tennis11's double fault, against 44% across a rally. Below
# this fraction, nothing was returned and the next strike is a second serve.
RALLY_BALL_RETURN_FRACTION = 0.15

# Everything below is a duration, held in seconds and converted with the clip's
# own fps. They were measured on 30fps footage; keeping them as seconds is what
# lets the same detector work on 60fps or 24fps clips.
TOSS_MAX_SECONDS = 0.73
TOSS_MAX_TRACK_GAP_SECONDS = 0.13
# The strike follows the apex while the ball falls to racket height: 8-9 frames
# at 30fps on the measured serves. Allow well past that, but not so far that the
# search runs into the next stroke.
REACH_SEARCH_SECONDS = 0.73
REACH_BASELINE_SECONDS = 2.0
REACH_LOCAL_WINDOW_SECONDS = 0.33
# Two serves of the same point are separated by a re-toss, never by less than
# about a second, and a second serve follows the first within about fifteen.
MIN_SERVE_SEPARATION_SECONDS = 1.0
MAX_SECOND_SERVE_GAP_SECONDS = 15.0
# Too little tracked ball to conclude anything about a rally from.
RALLY_MIN_TRACKED_SECONDS = 0.5

FPS_DEFAULT = 30.0


def frame_window(seconds, fps):
    """Convert a measured duration into a whole number of frames."""
    return max(1, int(round(seconds * (fps or FPS_DEFAULT))))


def ball_contact_point(ball):
    if not ball:
        return None
    bbox = ball.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, _y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))
    center = ball.get("center")
    if isinstance(center, list) and len(center) == 2:
        return (float(center[0]), float(center[1]))
    return None


def project_to_court_world(center, inv_homography):
    if center is None or inv_homography is None:
        return None
    point = np.array([[center]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, inv_homography)[0][0]
    return float(world[0]), float(world[1])


def _player_key(side):
    return "player_near" if side == "near" else "player_far"


def _player_box(by_frame, frame, side):
    row = by_frame.get(frame)
    if not row:
        return None
    return row.get(_player_key(side))


def _box_height(box):
    _, y1, _, y2 = box["bbox"]
    return max(1.0, float(y2 - y1))


def _distance_to_box(x, y, box):
    x1, y1, x2, y2 = box["bbox"]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def _feet_world(box, inv_homography):
    x1, _, x2, y2 = box["bbox"]
    return project_to_court_world(((x1 + x2) / 2.0, float(y2)), inv_homography)


def struck_from_behind_baseline(box, side, inv_homography):
    """Is this player standing behind their own baseline, ready to serve?"""
    world_point = _feet_world(box, inv_homography)
    if not world_point:
        return False, None
    y_world = world_point[1]
    if side == "near":
        return y_world >= NEAR_BASELINE_FT - BEHIND_BASELINE_MARGIN_FT, world_point
    return y_world <= FAR_BASELINE_FT + BEHIND_BASELINE_MARGIN_FT, world_point


def _ball_track(by_frame, start_frame, end_frame):
    track = []
    for frame in range(start_frame, end_frame + 1):
        row = by_frame.get(frame)
        if not row:
            continue
        ball = row.get("ball")
        center = ball.get("center") if ball else None
        if center:
            track.append((frame, float(center[0]), float(center[1])))
    return track


def _toss_apexes(by_frame, track, side, inv_homography, fps):
    """Frames where a near-vertical rise from the server's hand tops out."""
    apexes = []
    count = len(track)
    for index in range(2, count - 2):
        frame, x_apex, y_apex = track[index]
        window = range(max(0, index - 2), min(count, index + 3))
        if not all(track[other][2] >= y_apex for other in window):
            continue
        box = _player_box(by_frame, frame, side)
        if not box:
            continue
        height = _box_height(box)
        _, box_top, _, _ = box["bbox"]

        start = index
        while start > 0:
            prev_frame, prev_x, prev_y = track[start - 1]
            cur_frame, cur_x, cur_y = track[start]
            if cur_frame - prev_frame > frame_window(TOSS_MAX_TRACK_GAP_SECONDS, fps):
                break
            # Walking back through a rise, earlier frames sit lower in the image.
            # The tracker repeats a position often enough that an exact-equality
            # test cuts the toss short, so only stop once the ball was clearly
            # higher earlier, which means the rise has genuinely ended.
            if prev_y < cur_y - TOSS_RISE_NOISE_PX:
                break
            if abs(prev_x - cur_x) > max(8.0, TOSS_MAX_STEP_DRIFT_BOX_FRAC * height):
                break
            if frame - prev_frame > frame_window(TOSS_MAX_SECONDS, fps):
                break
            start -= 1

        rise = track[start][2] - y_apex
        drift = abs(track[start][1] - x_apex)
        if rise < TOSS_MIN_RISE_BOX_FRAC * height:
            continue
        if drift > max(25.0, TOSS_MAX_TOTAL_DRIFT_BOX_FRAC * rise):
            continue
        # The ball has to end up over the server's head; a ball bounced on the
        # court before serving rises too, but tops out below shoulder height.
        if y_apex > box_top - TOSS_APEX_ABOVE_HEAD_FRAC * height:
            continue
        if _distance_to_box(track[start][1], track[start][2], box) > (
            TOSS_START_MAX_DISTANCE_BOX_FRAC * height
        ):
            continue
        behind, world_point = struck_from_behind_baseline(box, side, inv_homography)
        if not behind:
            continue
        apexes.append(
            {
                "apex_frame": frame,
                "toss_start_frame": track[start][0],
                "toss_rise_px": round(rise, 1),
                "toss_drift_px": round(drift, 1),
                "server_box_height_px": round(height, 1),
                "server_feet_world": world_point,
            }
        )
    return apexes


def _peak_reach(by_frame, side, first_frame, last_frame):
    """Frame of maximum extension: the highest the box top gets in the window."""
    best_frame = None
    best_top = None
    for frame in range(first_frame, last_frame + 1):
        box = _player_box(by_frame, frame, side)
        if not box:
            continue
        box_top = float(box["bbox"][1])
        if best_top is None or box_top < best_top:
            best_top = box_top
            best_frame = frame
    return best_frame, best_top


def _reach_prominence(by_frame, side, frame, box_top, fps):
    """How far the box top at `frame` clears the player's resting pose."""
    tops = []
    baseline_frames = frame_window(REACH_BASELINE_SECONDS, fps)
    for other in range(frame - baseline_frames, frame + baseline_frames + 1):
        box = _player_box(by_frame, other, side)
        if box:
            tops.append(float(box["bbox"][1]))
    if len(tops) < baseline_frames:
        return None
    tops.sort()
    resting_top = tops[min(len(tops) - 1, int(REACH_BASELINE_PERCENTILE * len(tops)))]
    return resting_top - box_top


def _reach_candidates(by_frame, side, start_frame, end_frame, inv_homography, fps):
    """Serve-shaped full extensions found without any help from the ball."""
    candidates = []
    frame = start_frame
    while frame <= end_frame:
        box = _player_box(by_frame, frame, side)
        if not box:
            frame += 1
            continue
        height = _box_height(box)
        if height < REACH_MIN_BOX_HEIGHT_PX:
            frame += 1
            continue
        box_top = float(box["bbox"][1])
        # The whole serve motion spans about 25 frames, and the trophy position
        # partway through it is a local peak in its own right. Compare over a
        # window wide enough to contain both, so only the true full extension
        # survives.
        local_frame, local_top = _peak_reach(
            by_frame,
            side,
            max(start_frame, frame - frame_window(REACH_LOCAL_WINDOW_SECONDS, fps)),
            min(end_frame, frame + frame_window(REACH_LOCAL_WINDOW_SECONDS, fps)),
        )
        if local_frame != frame or local_top is None:
            frame += 1
            continue
        prominence = _reach_prominence(by_frame, side, frame, box_top, fps)
        if prominence is None or prominence < REACH_MIN_PROMINENCE_BOX_FRAC * height:
            frame += 1
            continue
        behind, world_point = struck_from_behind_baseline(box, side, inv_homography)
        if not behind:
            frame += 1
            continue
        candidates.append(
            {
                "contact_frame": frame,
                "server_box_height_px": round(height, 1),
                "server_feet_world": world_point,
                "reach_prominence_px": round(prominence, 1),
            }
        )
        frame += frame_window(MIN_SERVE_SEPARATION_SECONDS, fps)
    return candidates


def ball_return_fraction(by_frame, first_frame, last_frame, server, inv_homography, fps):
    """Share of tracked ball frames spent back on the server's own side.

    This is how a fault is told from a rally without needing the bounces: a
    served ball that faults stays on the receiver's side until it is collected,
    while a rally sends it back and forth across the net.
    """
    tracked = 0
    returned = 0
    for frame in range(first_frame + 1, last_frame):
        row = by_frame.get(frame)
        ball = row.get("ball") if row else None
        if not ball or not ball.get("center"):
            continue
        tracked += 1
        world_point = project_to_court_world(
            ball_contact_point(ball) or ball.get("center"), inv_homography
        )
        if not world_point:
            continue
        on_near_side = world_point[1] > COURT_NET_Y_FT
        if on_near_side == (server == "near"):
            returned += 1
    if tracked < frame_window(RALLY_MIN_TRACKED_SECONDS, fps):
        return None, tracked
    return returned / float(tracked), tracked


def _dedupe(motions, fps):
    """Collapse candidates that describe the same strike."""
    ordered = sorted(motions, key=lambda motion: motion["contact_frame"])
    kept = []
    for motion in ordered:
        if kept and motion["contact_frame"] - kept[-1]["contact_frame"] < frame_window(
            MIN_SERVE_SEPARATION_SECONDS, fps
        ):
            if _rank(motion) > _rank(kept[-1]):
                kept[-1] = motion
            continue
        kept.append(motion)
    return kept


def _rank(motion):
    return {"high": 2, "medium": 1, "low": 0}.get(motion.get("confidence"), 0)


def detect_serve_motions_for_point(by_frame, start_frame, end_frame, server, inv_homography, fps=FPS_DEFAULT):
    """Serve strikes inside one point, earliest first."""
    track = _ball_track(by_frame, start_frame, end_frame)
    motions = []
    for apex in _toss_apexes(by_frame, track, server, inv_homography, fps):
        apex_frame = apex["apex_frame"]
        # Search the full reach window even where it runs past the caller's
        # window. Clipping it does not drop the strike, it MOVES it: the contact
        # is the highest the box top gets, so a truncated search reports the
        # edge, or the toss apex, with the same confidence as a real contact.
        # Measured on tennis11 point 1 (true contact 159) -- a window ending at
        # 158 reported 158, one ending at 155 reported 151, the apex. The frames
        # are in by_frame either way, so read them and let the caller decide
        # what to do with a strike that lands past its boundary.
        contact_frame, contact_top = _peak_reach(
            by_frame,
            server,
            apex_frame,
            apex_frame + frame_window(REACH_SEARCH_SECONDS, fps),
        )
        if contact_frame is None:
            continue
        motion = dict(apex)
        motion["contact_frame"] = contact_frame
        motion["source"] = "ball_toss"
        motion["confidence"] = "high"
        motion["reasons"] = ["toss_above_head_behind_baseline", "contact_at_peak_reach"]
        prominence = _reach_prominence(by_frame, server, contact_frame, contact_top, fps)
        motion["reach_prominence_px"] = round(prominence, 1) if prominence is not None else None
        # The strike is the highest the box top gets, so a search cut short by
        # the end of the window reports the edge rather than the peak -- a wrong
        # contact frame carrying the same confidence as a right one. Callers
        # scanning arbitrary or overlapping windows cannot see that from the
        # result, so say it here. Measured on tennis11 point 1 (true contact
        # 159): a window ending at 158 reports 158, one ending at 155 reports
        # 151, which is the toss apex rather than the strike.
        motion["contact_outside_window"] = contact_frame > end_frame
        motions.append(motion)

    motions = _dedupe(motions, fps)
    if motions:
        return motions

    # No toss was ever tracked for this point. Fall back to the extension alone,
    # which still finds the serve but cannot prove a ball was thrown, so it is
    # reported at lower confidence.
    fallback = []
    for candidate in _reach_candidates(by_frame, server, start_frame, end_frame, inv_homography, fps):
        motion = dict(candidate)
        motion["apex_frame"] = None
        motion["toss_start_frame"] = None
        motion["source"] = "peak_reach"
        motion["confidence"] = "medium"
        motion["reasons"] = ["full_extension_behind_baseline", "toss_not_tracked"]
        fallback.append(motion)
    return _dedupe(fallback, fps)


def detect_serve_motions(by_frame, point_ranges, server, inv_homography, fps=FPS_DEFAULT):
    """Map point index (1-based) to the serve strikes detected inside it."""
    motions_by_point = {}
    if not server:
        return motions_by_point
    for index, point_range in enumerate(point_ranges, start=1):
        motions = detect_serve_motions_for_point(
            by_frame,
            point_range["start_frame"],
            point_range["end_frame"],
            server,
            inv_homography,
            fps,
        )
        if not motions:
            continue
        # A second serve follows the first one closely. Anything further out is
        # a later stroke that happens to look like a serve, not this point's
        # second attempt.
        trimmed = [motions[0]]
        for motion in motions[1:]:
            if motion["contact_frame"] - trimmed[-1]["contact_frame"] > frame_window(
                MAX_SECOND_SERVE_GAP_SECONDS, fps
            ):
                break
            trimmed.append(motion)
            if len(trimmed) == 2:
                break
        for attempt_number, motion in enumerate(trimmed, start=1):
            motion["attempt"] = attempt_number
            motion["point_index"] = index
            if attempt_number == 1:
                motion["ball_return_fraction"] = None
                motion["ball_tracked_frames_since_previous"] = None
                continue
            fraction, tracked = ball_return_fraction(
                by_frame,
                trimmed[attempt_number - 2]["contact_frame"],
                motion["contact_frame"],
                server,
                inv_homography,
                fps,
            )
            motion["ball_return_fraction"] = round(fraction, 3) if fraction is not None else None
            motion["ball_tracked_frames_since_previous"] = tracked
            motion["rally_between_serves"] = (
                None if fraction is None else fraction >= RALLY_BALL_RETURN_FRACTION
            )
        motions_by_point[index] = trimmed
    return motions_by_point
