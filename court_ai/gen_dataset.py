"""Pre-generate a cached synthetic court dataset to disk (fast training).

Renders at 16:9 (matching real footage) then resizes to the square model input
exactly as inference does, so synthetic and real courts undergo identical
preprocessing. Saves one .npz with uint8 images and float32 labels.

Usage:
    python -m court_ai.gen_dataset --n 12000 --out court_ai/_data/train.npz
"""
import argparse, os
import numpy as np
import cv2
from court_ai.synth import make_sample
from court_ai.model import INPUT_SIZE

# 16:9 render size (matches 1920x1080 aspect); resized to INPUT_SIZE like inference.
RENDER_W, RENDER_H = 480, 270


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    imgs = np.empty((args.n, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels = np.empty((args.n, 4, 2), dtype=np.float32)
    for i in range(args.n):
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        img, label, _ = make_sample(rng, RENDER_W, RENDER_H)
        imgs[i] = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        labels[i] = label
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.n}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, images=imgs, labels=labels)
    print(f"wrote {args.out}  images={imgs.shape} labels={labels.shape}")


if __name__ == "__main__":
    main()
