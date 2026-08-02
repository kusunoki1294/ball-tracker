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


class SynthCourt(Dataset):
    def __init__(self, n, seed, W=960, H=540):
        self.n, self.seed, self.W, self.H = n, seed, W, H

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        img, label, _ = make_sample(rng, self.W, self.H)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(label)  # (4,2) normalized
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
    ap.add_argument("--val", type=int, default=256)
    args = ap.parse_args()

    dev = device()
    print(f"device={dev} steps={args.steps} batch={args.batch}")
    ckdir = os.path.join(os.path.dirname(__file__), "_checkpoints")
    os.makedirs(ckdir, exist_ok=True)

    train_ds = SynthCourt(args.steps * args.batch, seed=1)
    val_ds = SynthCourt(args.val, seed=999)
    train_ld = DataLoader(train_ds, batch_size=args.batch, num_workers=4, drop_last=True)
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
            break
    torch.save(model.state_dict(), os.path.join(ckdir, "courtnet.pt"))
    me, md = evaluate()
    print(f"DONE. final val_corner_err mean={me:.1f}px median={md:.1f}px  saved courtnet.pt")


if __name__ == "__main__":
    main()
