"""Extract a clean, empty-court image from a match video by median-averaging
sampled frames. Moving players/ball vanish; the static court remains.
"""
import numpy as np
import cv2


def clean_court_image(video_path, n_frames=40):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        raise RuntimeError(f"no frames in {video_path}")
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
