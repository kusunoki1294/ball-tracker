"""Synthetic tennis-court image generator with exact corner labels.

Renders the canonical court geometry through random behind-the-baseline
homographies onto varied backgrounds, with simulated net bands, player
occlusions, blur, and photometric jitter. Each sample yields an image and the
four doubles-corner image coordinates (near_left, near_right, far_right,
far_left), normalized to [0, 1].

Deterministic given a numpy Generator, so the training/val split is stable.
"""
import numpy as np
import cv2
from court_ai.court_geometry import COURT_LINES, NET_LINE, CORNERS_WORLD, L_COURT, W_DOUBLES


def _rand_corners(rng, W, H):
    """Sample plausible image positions for the 4 doubles corners (behind-baseline view)."""
    # near baseline: low in frame, wide
    near_y = rng.uniform(0.60, 0.94) * H
    near_half = rng.uniform(0.34, 0.52) * W       # half-width of near baseline
    cx = rng.uniform(0.42, 0.58) * W              # court horizontal center
    nl = [cx - near_half, near_y]
    nr = [cx + near_half, near_y]
    # far baseline: higher, narrower (perspective), possibly slightly offset
    far_y = rng.uniform(0.10, 0.42) * H
    far_half = near_half * rng.uniform(0.22, 0.55)
    fcx = cx + rng.uniform(-0.06, 0.06) * W
    fl = [fcx - far_half, far_y]
    fr = [fcx + far_half, far_y]
    pts = np.array([nl, nr, fr, fl], dtype=np.float32)
    # small global perspective jitter
    jit = rng.normal(0, 0.012 * W, size=pts.shape).astype(np.float32)
    return pts + jit


def _bg(rng, W, H):
    base = rng.integers(30, 150, size=3)
    img = np.full((H, W, 3), base, dtype=np.uint8)
    # a few random color blocks (trees/fence/houses feel)
    for _ in range(rng.integers(3, 8)):
        x0, y0 = rng.integers(0, W), rng.integers(0, H // 2)
        x1, y1 = min(W, x0 + rng.integers(60, 500)), min(H, y0 + rng.integers(40, 300))
        col = rng.integers(20, 200, size=3).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(3, 12))
    return img


def make_sample(rng, W=960, H=540):
    corners_img = _rand_corners(rng, W, H)
    Hmg = cv2.getPerspectiveTransform(CORNERS_WORLD, corners_img)

    def proj(pts):
        p = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, Hmg).reshape(-1, 2)

    img = _bg(rng, W, H)
    # court surface fill (blue/green-ish), polygon of the 4 corners
    surf = corners_img.astype(np.int32)
    scol = [int(rng.integers(70, 150)), int(rng.integers(60, 130)), int(rng.integers(40, 110))]
    cv2.fillConvexPoly(img, surf[[0, 1, 2, 3]], scol)
    # surround apron slightly different tone (draw a bigger faint quad behind)
    # white lines (thickness scaled to render resolution to match real squished courts)
    line_col = int(rng.integers(190, 255))
    thick = max(1, int(round(W / 256.0 * rng.uniform(1.0, 2.4))))
    for (x0, y0, x1, y1) in COURT_LINES:
        p = proj([(x0, y0), (x1, y1)]).astype(np.int32)
        faint = rng.random() < 0.12  # occasionally fade an interior line
        c = int(line_col * (0.5 if faint else 1.0))
        cv2.line(img, tuple(p[0]), tuple(p[1]), (c, c, c), thick, cv2.LINE_AA)

    # net occlusion band across the court at y=NET
    if rng.random() < 0.9:
        p = proj([(NET_LINE[0], NET_LINE[1]), (NET_LINE[2], NET_LINE[3])]).astype(np.int32)
        band = max(2, int(round(W / 256.0 * rng.uniform(2.5, 7.0))))
        netcol = int(rng.integers(20, 90))
        cv2.line(img, tuple(p[0]), tuple(p[1]), (netcol, netcol, netcol), band, cv2.LINE_AA)

    # player occlusions: a few dark blobs near the court
    for _ in range(rng.integers(0, 3)):
        wc = proj([(rng.uniform(2, W_DOUBLES - 2), rng.uniform(2, L_COURT - 2))])[0]
        ax, ay = int(rng.integers(15, 40)), int(rng.integers(40, 90))
        cv2.ellipse(img, (int(wc[0]), int(wc[1])), (ax, ay), 0, 0, 360,
                    rng.integers(20, 120, size=3).tolist(), -1)

    # photometric jitter + blur + noise
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 2.2))
    noise = rng.normal(0, rng.uniform(3, 14), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    label = (corners_img / np.array([W, H], dtype=np.float32)).astype(np.float32)  # (4,2) in [0,1]
    return img, label, corners_img


if __name__ == "__main__":
    import os
    rng = np.random.default_rng(0)
    out = os.path.join(os.path.dirname(__file__), "_synth_preview")
    os.makedirs(out, exist_ok=True)
    tiles = []
    for i in range(9):
        img, label, corners = make_sample(rng)
        for c in corners.astype(int):
            cv2.circle(img, tuple(c), 8, (0, 0, 255), -1)
        tiles.append(cv2.resize(img, (320, 180)))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
    cv2.imwrite(os.path.join(out, "preview.png"), grid)
    print("wrote", os.path.join(out, "preview.png"))
