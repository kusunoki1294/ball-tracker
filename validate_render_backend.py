"""Validate that the render backend produces a decodable video."""

import os
import subprocess
import tempfile

import numpy as np

from render_tennis_analysis import FFmpegWriter


def main():
    with tempfile.TemporaryDirectory(prefix="render-backend-") as directory:
        path = os.path.join(directory, "smoke.mp4")
        writer = FFmpegWriter(path, 64, 48, 30.0)
        if not writer.isOpened():
            writer.release()
            raise AssertionError("FFmpeg render writer did not start")
        first = np.zeros((48, 64, 3), dtype=np.uint8)
        second = first.copy()
        second[:, :, 1] = 255
        writer.write(first)
        writer.write(second)
        writer.release()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-of", "json", path],
            check=True, capture_output=True, text=True,
        )
        if '"nb_frames": "2"' not in probe.stdout:
            raise AssertionError(f"render output was not two decodable frames: {probe.stdout}")
        if '"width": 64' not in probe.stdout or '"height": 48' not in probe.stdout:
            raise AssertionError(f"render output has unexpected dimensions: {probe.stdout}")
    print("render backend validation passed (2 decodable frames)")


if __name__ == "__main__":
    main()
