"""Domain-invariant input for CourtNet: reduce any court image (synthetic or
real) to a single-channel white-line mask. This strips color, texture,
lighting, and background so a model trained on synthetic courts transfers to
real footage.
"""
import cv2
import numpy as np


def to_linemap(bgr, size=256, line_px=9, min_line_response=35):
    """BGR image -> single-channel float32 line map in [0,1], resized to size.

    Uses a white top-hat (gray - opening) to isolate thin *bright* structures
    (court lines) independent of court/background brightness, so a light court
    and a dark court both reduce to the same clean line pattern. This is the
    domain-invariant representation shared by synthetic and real inputs.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    # Guard against amplifying noise: normalize + Otsu below stretch even a
    # near-uniform image to full range and split it ~in half, turning texture
    # into a dense fake "line map". Real court lines are the brightest thin
    # structures, so the top of the top-hat response is high (measured p99.5 ~67
    # for a real court vs ~23 for heavy sensor noise); if even the strongest few
    # percent of responses are this weak there are no lines, so return an empty
    # map. The floor sits well below any genuine court, so it never alters the
    # output for real or synthetic inputs.
    if np.percentile(tophat, 99.5) < min_line_response:
        return np.zeros((size, size), dtype=np.float32)
    # normalize response and threshold relative to its own strength
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return (m.astype(np.float32) / 255.0)
