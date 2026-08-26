"""Train CourtNet on synthetic tennis-court images.

Data is generated on the fly (infinite variety). Trains on MPS/CPU and saves a
checkpoint to court_ai/_checkpoints/courtnet.pt. Reports validation corner
error in pixels (on a fixed held-out synthetic set).

Usage:
    python -m court_ai.train --steps 4000 --batch 32
"""
import argparse, os, time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from court_ai.synth import make_sample
from court_ai.model import CourtNet, INPUT_SIZE


def _clutter(lm, rng):
    """Add background-like speckle/segments and line dropout to a line map so
    the model learns to ignore real-world clutter (trees, fences, overlays).
    Vectorized speckle keeps this cheap. Only a mild top-bias so it does not
    push the predicted far baseline downward."""
    H, W = lm.shape
    # vectorized speckle, mildly denser toward the top (background)
    n = int(rng.integers(60, 260))
    ys = (rng.beta(1.6, 2.0, n) * H).astype(np.int32).clip(0, H - 1)
    xs = rng.integers(0, W, n)
    lm[ys, xs] = 255
    lm[np.clip(ys + 1, 0, H - 1), xs] = 255  # 2px tall so it survives downscale
    # a few stray background line segments (upper region)
    for _ in range(int(rng.integers(0, 6))):
        p1 = (int(rng.integers(0, W)), int(rng.integers(0, H // 2)))
        p2 = (p1[0] + int(rng.integers(-60, 60)), p1[1] + int(rng.integers(-30, 30)))
        cv2.line(lm, p1, p2, 255, int(rng.integers(1, 3)))
    # random erasing (occlusion of court lines)
    for _ in range(int(rng.integers(0, 3))):
        ex, ey = int(rng.integers(0, W)), int(rng.integers(0, H))
        ew, eh = int(rng.integers(15, 60)), int(rng.integers(15, 60))
        lm[ey:ey+eh, ex:ex+ew] = 0
    return lm


class CachedCourt(Dataset):
    """Cached line-map dataset (npz, 1-channel) with clutter augmentation."""
    def __init__(self, npz_path, augment=True):
        d = np.load(npz_path)
        self.images = d["images"]  # (N,256,256) uint8 line maps
        self.labels = d["labels"]  # (N,4,2) float32
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        lm = self.images[i].copy()
        if self.augment:
            # Seed from the per-worker, per-epoch torch seed (PyTorch reseeds each
            # worker every epoch) combined with the sample index, so a given sample
            # gets fresh clutter each epoch instead of the same fixed pattern.
            rng = np.random.default_rng([torch.initial_seed() & 0xFFFFFFFF, i])
            lm = _clutter(lm, rng)
        x = torch.from_numpy(lm.astype(np.float32) / 255.0).unsqueeze(0)  # (1,256,256)
        y = torch.from_numpy(self.labels[i])
        return x, y


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-data", default="court_ai/_data/train_lm.npz")
    ap.add_argument("--val-data", default="court_ai/_data/val_lm.npz")
    args = ap.parse_args()

    dev = device()
    print(f"device={dev} steps={args.steps} batch={args.batch}")
    ckdir = os.path.join(os.path.dirname(__file__), "_checkpoints")
    os.makedirs(ckdir, exist_ok=True)

    train_ds = CachedCourt(args.train_data, augment=True)
    val_ds = CachedCourt(args.val_data, augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, num_workers=4, drop_last=True, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=64, num_workers=2)

    model = CourtNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    lossfn = torch.nn.SmoothL1Loss(beta=0.01)

    def evaluate():
        model.eval()
        errs = []
        with torch.no_grad():
            for x, y in val_ld:
                pred = model(x.to(dev)).cpu()
                # pixel error at 960x540 reference
                d = (pred - y) * torch.tensor([960.0, 540.0])
                errs.append(torch.linalg.norm(d, dim=-1))  # (B,4)
        e = torch.cat(errs)
        return e.mean().item(), e.median().item()

    model.train()
    t0 = time.time()
    step = 0
    running = 0.0
    done = False
    while not done:
        for x, y in train_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
            if step % 200 == 0:
                me, md = evaluate()
                model.train()
                print(f"step {step:5d}/{args.steps}  loss={running/200:.5f}  "
                      f"val_corner_err mean={me:5.1f}px median={md:5.1f}px  "
                      f"({(time.time()-t0):.0f}s)")
                running = 0.0
                torch.save(model.state_dict(), os.path.join(ckdir, "courtnet.pt"))
            if step >= args.steps:
                done = True
                break
    torch.save(model.state_dict(), os.path.join(ckdir, "courtnet.pt"))
    me, md = evaluate()
    print(f"DONE. final val_corner_err mean={me:.1f}px median={md:.1f}px  saved courtnet.pt")


if __name__ == "__main__":
    main()
