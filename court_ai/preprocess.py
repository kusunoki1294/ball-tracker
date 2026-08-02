"""Domain-invariant input for CourtNet: reduce any court image (synthetic or
real) to a single-channel white-line mask. This strips color, texture,
lighting, and background so a model trained on synthetic courts transfers to
real footage.
"""
import cv2
import numpy as np


def to_linemap(bgr, size=256, line_px=9):
    """BGR image -> single-channel float32 line map in [0,1], resized to size.

    Uses a white top-hat (gray - opening) to isolate thin *bright* structures
    (court lines) independent of court/background brightness, so a light court
    and a dark court both reduce to the same clean line pattern. This is the
    domain-invariant representation shared by synthetic and real inputs.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    # normalize response and threshold relative to its own strength
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    return (m.astype(np.float32) / 255.0)
