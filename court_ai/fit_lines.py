"""Refine predicted court corners by fitting the full court line model.

CourtNet localises the court well enough to initialise a search but not well
enough to calibrate with: on tennis11 its corners were ~8px off on the interior
lines and ~25px off on the far baseline, which is metres of world error at the
far end.

Snapping only the four outer edges to the nearest bright ridge does not fix it.
That problem is under-constrained -- the far baseline is short, faint, and has
the net tape and the court/apron boundary running parallel to it, so the fitted
far edge can slide onto a neighbour and still look self-consistent (measured:
27px out on the far service line).

Fitting the *whole* line model does fix it. The interior lines -- singles
sidelines, both service lines, the centre service line -- are long, high
contrast, and spread through the court, and no wrong homography can light all of
them up at once while also matching the boundary. Scoring is the mean top-hat
response along each projected line, averaged over lines so that every line must
contribute rather than letting the long sidelines dominate. Optimisation is
Powell from the CourtNet corners, coarse-to-fine over three blur levels so the
search starts in a wide basin and finishes sharp.

Measured on tennis11 (mean |offset| of projected lines from the real lines):
CourtNet 7.8px -> edge-snapping 10.0px -> this 0.4px. Verified against two
features the objective never sees: the projected net-post bases and the baseline
centre marks both land on the real ones.
"""
import numpy as np
import cv2
from scipy.optimize import minimize

from court_ai.court_geometry import CORNERS_WORLD, COURT_LINES

# The painted lines, minus the two short centre marks: a 2ft tick sampled like a
# full line is mostly noise, and keeping the marks out of the objective leaves
# them free as an independent check on the result.
MIN_MODEL_LINE_FT = 10.0
MODEL_LINES = tuple(
    ((x0, y0), (x1, y1))
    for x0, y0, x1, y1 in COURT_LINES
    if np.hypot(x1 - x0, y1 - y0) >= MIN_MODEL_LINE_FT
)

SAMPLES_PER_LINE = 70
MIN_IN_FRAME = 0.55


def _sample_grid():
    pts, owner = [], []
    for i, (a, b) in enumerate(MODEL_LINES):
        a, b = np.array(a), np.array(b)
        for s in np.linspace(0.02, 0.98, SAMPLES_PER_LINE):
            pts.append(a + (b - a) * s)
            owner.append(i)
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2), np.array(owner)


_WORLD_PTS, _OWNER = _sample_grid()


def line_response_map(bgr, line_px=9, sigma=1.5):
    """Full-resolution top-hat response: thin bright structures, i.e. court lines."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    return cv2.GaussianBlur(tophat, (0, 0), sigma).astype(np.float32)


def score_corners(corners, response):
    """Mean line response of the projected court model. Higher is better."""
    corners = np.asarray(corners, dtype=np.float32)
    try:
        homography = cv2.getPerspectiveTransform(CORNERS_WORLD, corners)
    except cv2.error:
        return -1e6
    proj = cv2.perspectiveTransform(_WORLD_PTS, homography).reshape(-1, 2)
    if not np.isfinite(proj).all():
        return -1e6
    h, w = response.shape
    inside = (
        (proj[:, 0] >= 0) & (proj[:, 0] < w - 1)
        & (proj[:, 1] >= 0) & (proj[:, 1] < h - 1)
    )
    # Cameras legitimately crop the near doubles corners, so some samples fall
    # outside the frame; only reject when most of the court has left it.
    if inside.mean() < MIN_IN_FRAME:
        return -1e6
    vals = cv2.remap(
        response,
        proj[inside, 0].reshape(-1, 1),
        proj[inside, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).ravel()
    owner = _OWNER[inside]
    per_line = [
        vals[owner == i].mean()
        for i in range(len(MODEL_LINES))
        if (owner == i).sum() > 8
    ]
    # Every line must be visible enough to vote, otherwise the fit could win by
    # abandoning the far baseline and hugging the long sidelines.
    if len(per_line) < len(MODEL_LINES):
        return -1e6
    return float(np.mean(per_line))


def refine_corners(corners, court_bgr, verbose=False):
    """Locally optimise the 4 corners against the full court line model.

    Returns (refined_corners, score_before, score_after). If the search fails to
    improve on the input, the input is returned unchanged.
    """
    response = line_response_map(court_bgr)
    best = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    start_score = score_corners(best, response)
    best_score = start_score

    for sigma, step in ((6.0, 40.0), (3.0, 16.0), (1.5, 6.0)):
        blurred = cv2.GaussianBlur(response, (0, 0), sigma)
        result = minimize(
            lambda x: -score_corners(x.reshape(4, 2), blurred),
            best.ravel(),
            method="Powell",
            options={"xtol": 0.05, "ftol": 1e-4, "maxiter": 20000,
                     "direc": np.eye(8) * step},
        )
        candidate = result.x.reshape(4, 2)
        value = score_corners(candidate, response)
        if verbose:
            print(f"  refine sigma={sigma:<4} score={value:7.3f} (best {best_score:7.3f})")
        if value > best_score:
            best, best_score = candidate, value

    return best.astype(np.float32), start_score, best_score
