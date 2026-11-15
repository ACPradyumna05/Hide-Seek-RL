import math
import numpy as np

def norm(v):
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-6 else np.zeros(2, dtype=np.float32)


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def segment_hits_rect(start, end, rect):
    """Liang-Barsky: true if segment passes through an opaque rectangle."""
    x0, y0 = start
    x1, y1 = end
    rx, ry, rw, rh = rect
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - rx, rx + rw - x0, y0 - ry, ry + rh - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-10:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u0 = max(u0, t)
            else:
                u1 = min(u1, t)
            if u0 > u1:
                return False
    return True
