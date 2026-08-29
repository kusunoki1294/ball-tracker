"""Dependency-free court geometry helpers shared across the tennis pipeline."""


COURT_WORLD_CORNERS_FT = (
    (0.0, 78.0),
    (36.0, 78.0),
    (36.0, 0.0),
    (0.0, 0.0),
)


def ball_contact_point(ball):
    if not ball:
        return None
    bbox = ball.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, _y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))
    center = ball.get("center")
    if isinstance(center, list) and len(center) == 2:
        return (float(center[0]), float(center[1]))
    return None


def order_court_corners(points):
    points = sorted(points, key=lambda p: p[1])
    far = points[:2]
    near = points[2:]
    far_left, far_right = sorted(far, key=lambda p: p[0])
    near_left, near_right = sorted(near, key=lambda p: p[0])
    return [near_left, near_right, far_right, far_left]


def project_to_court_world(center, inv_homography):
    if center is None or inv_homography is None:
        return None
    x, y = center
    h = inv_homography
    denom = (h[2][0] * x) + (h[2][1] * y) + h[2][2]
    if abs(denom) < 1e-12:
        return None
    world_x = ((h[0][0] * x) + (h[0][1] * y) + h[0][2]) / denom
    world_y = ((h[1][0] * x) + (h[1][1] * y) + h[1][2]) / denom
    return float(world_x), float(world_y)


def build_inverse_court_homography(court_calib):
    if not court_calib or court_calib.get("points") is None:
        return None
    return projective_transform(court_calib["points"], COURT_WORLD_CORNERS_FT)


def projective_transform(source_points, target_points):
    rows = []
    values = []
    for (x, y), (u, v) in zip(source_points, target_points):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    solved = solve_linear_system(rows, values)
    if solved is None:
        return None
    a, b, c, d, e, f, g, h = solved
    return ((a, b, c), (d, e, f), (g, h, 1.0))


def solve_linear_system(rows, values):
    matrix = [list(row) + [value] for row, value in zip(rows, values)]
    size = len(values)
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [value / divisor for value in matrix[col]]
        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            matrix[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(matrix[row], matrix[col])
            ]
    return [matrix[row][-1] for row in range(size)]
