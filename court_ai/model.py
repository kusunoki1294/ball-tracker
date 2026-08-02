"""Compact CNN that regresses the 4 doubles-corner image coordinates.

Output is 8 numbers = (near_left, near_right, far_right, far_left) x (x, y),
normalized to roughly [0, 1] (values may fall slightly outside when a corner
sits off-frame). Small enough to train on an M2 (MPS) in minutes.
"""
import torch
import torch.nn as nn

INPUT_SIZE = 256


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CourtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),    # 256 -> 128
            _block(32, 64),   # 128 -> 64
            _block(64, 96),   # 64 -> 32
            _block(96, 128),  # 32 -> 16
            _block(128, 160), # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(160, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.head(self.features(x)).reshape(-1, 4, 2)
