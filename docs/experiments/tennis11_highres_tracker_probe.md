# Tennis11 High-Resolution Tracker Probe

This was a six-second, isolated probe of the game-1 source around source frames
960-1139. It compared the existing tracker behavior with a higher-resolution,
lower-confidence YOLO pass:

    --imgsz 1280 --ball-conf 0.08 --far-ball-conf 0.05 --far-ball-imgsz 1920

The probe completed all 180 frames through the FFmpeg input fallback. The
baseline and probe each produced five offline bounce candidates in this slice.
The probe localized them at frames 966, 989, 1013, 1074, and 1104; the baseline
localized them at 967, 992, 1013, 1075, and 1104. No candidate-free bounce was
recovered, and the small frame shifts would require label re-checking.

Conclusion: higher input resolution and a lower confidence threshold did not
improve bounce recall in this control slice. Do not promote these settings as a
production change based on this probe. A useful next tracker experiment needs a
different ball model, a temporal detector, or a longer labeled comparison.

The probe was intentionally isolated and did not change the shipped JSONL or
analysis artifacts.

The same settings were also tested on an independent six-second slice around
source frames 1350-1529. The probe tracked 66 frames with a ball versus 63 in
the baseline log, but both produced one offline bounce. The probe localized it
at f1505 while the baseline localized it at f1511, so it added no recall and
made the known event's frame alignment worse. This second control does not
justify changing the tracker defaults either.

A third targeted probe used `--imgsz 1920 --ball-conf 0.05 --far-ball-conf 0.03
--far-ball-imgsz 1920` around source frames 1951-2130, covering another group of
candidate-free reversal events. It tracked 148 of 180 frames with a ball and
produced four bounce candidates. None was a new candidate-free reversal, so the
larger inference size and lower threshold still did not demonstrate a recall
gain. This probe also remains isolated from the shipped log and analysis.
