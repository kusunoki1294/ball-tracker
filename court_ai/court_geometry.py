"""Canonical tennis-court geometry in world coordinates.

World frame matches track_ball_yolo's homography model: x in [0, 36] (doubles
width, feet), y in [0, 78] (length, feet), far baseline at y=0, near baseline
at y=78, net at y=39. Corners are the four doubles corners.
"""
import numpy as np

# doubles court is 36 x 78 ft; singles alley 4.5 ft; service line 21 ft from net.
W_DOUBLES = 36.0
L_COURT = 78.0
ALLEY = 4.5
NET_Y = 39.0
SERVICE_OFFSET = 21.0  # from net

SINGLES_L = ALLEY            # 4.5
SINGLES_R = W_DOUBLES - ALLEY  # 31.5
FAR_SERVICE_Y = NET_Y - SERVICE_OFFSET   # 18
NEAR_SERVICE_Y = NET_Y + SERVICE_OFFSET  # 60
CENTER_X = W_DOUBLES / 2.0  # 18

# Painted white lines as world-space segments (x0,y0,x1,y1)
COURT_LINES = [
    # baselines
    (0.0, 0.0, W_DOUBLES, 0.0),
    (0.0, L_COURT, W_DOUBLES, L_COURT),
    # doubles sidelines
    (0.0, 0.0, 0.0, L_COURT),
    (W_DOUBLES, 0.0, W_DOUBLES, L_COURT),
    # singles sidelines
    (SINGLES_L, 0.0, SINGLES_L, L_COURT),
    (SINGLES_R, 0.0, SINGLES_R, L_COURT),
    # service lines (singles width)
    (SINGLES_L, FAR_SERVICE_Y, SINGLES_R, FAR_SERVICE_Y),
    (SINGLES_L, NEAR_SERVICE_Y, SINGLES_R, NEAR_SERVICE_Y),
    # center service line
    (CENTER_X, FAR_SERVICE_Y, CENTER_X, NEAR_SERVICE_Y),
    # center marks on baselines (short ticks)
    (CENTER_X, 0.0, CENTER_X, 2.0),
    (CENTER_X, L_COURT - 2.0, CENTER_X, L_COURT),
]

# Net line (drawn as occlusion band, not a painted white line)
NET_LINE = (0.0, NET_Y, W_DOUBLES, NET_Y)

# Doubles corners in the calibration order: near_left, near_right, far_right, far_left
CORNERS_WORLD = np.array([
    [0.0, L_COURT],        # near_left
    [W_DOUBLES, L_COURT],  # near_right
    [W_DOUBLES, 0.0],      # far_right
    [0.0, 0.0],            # far_left
], dtype=np.float32)
