"""Exact hull collision: pure float geometry, no pygame, no random.

Everything here is deterministic on (pos, angle) so it's lockstep-safe:
both peers run the identical tests on the identical sim state.

Coordinate convention matches the rest of the sim: hull-local coords are
+x = nose, +y = starboard. World queries transform through the ship's
current pos + angle (the same axes() the renderer uses).
"""
import math


def convex_hull(points):
    """Monotone-chain convex hull. Deterministic (sorts by (x, y)).

    Returns hull vertices as a list of (x, y) in CCW order, no repeated
    first point. Collinear points are dropped (cross <= 0 pops them).
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_convex_poly(px, py, poly):
    """True if (px, py) is inside or on the boundary of convex poly.

    Winding-agnostic: inside iff the point is on the same side of every
    edge (all cross products share a sign). On-boundary (cross == 0)
    counts as inside.
    """
    pos = neg = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if c > 0:
            pos += 1
        elif c < 0:
            neg += 1
        if pos and neg:
            return False
    return True


def _on_segment(a, b, p):
    """True if collinear point p lies on segment a-b (inclusive)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _seg_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 intersects segment p3-p4 (touching counts)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 1 if v > 0 else (-1 if v < 0 else 0)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def _seg_intersect_t(p1, p2, a, b):
    """Parameter t in [0,1] where segment p1->p2 crosses a->b, else None."""
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = b[0] - a[0], b[1] - a[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        return None
    t = ((a[0] - p1[0]) * dy2 - (a[1] - p1[1]) * dx2) / denom
    u = ((a[0] - p1[0]) * dy1 - (a[1] - p1[1]) * dx1) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def segment_hits_convex_poly(p1, p2, poly):
    """Sweep segment p1->p2 against convex poly. Returns (hit, hit_point).

    hit_point is the first intersection along the segment (or an endpoint
    if it starts/ends inside). (False, None) if the segment misses.
    """
    if point_in_convex_poly(p1[0], p1[1], poly):
        return True, p1
    if point_in_convex_poly(p2[0], p2[1], poly):
        return True, p2
    best_t, best_pt = None, None
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if _seg_intersect(p1, p2, a, b):
            t = _seg_intersect_t(p1, p2, a, b)
            if t is not None and (best_t is None or t < best_t):
                best_t = t
                best_pt = (p1[0] + (p2[0] - p1[0]) * t,
                           p1[1] + (p2[1] - p1[1]) * t)
    if best_t is not None:
        return True, best_pt
    return False, None


def _project_poly(poly, axis):
    vals = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
    return min(vals), max(vals)


def polys_overlap(poly_a, poly_b):
    """SAT: True if two convex polygons overlap (touching counts)."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            alen = math.hypot(ex, ey)
            if alen < 1e-12:
                continue
            axis = (-ey / alen, ex / alen)   # unit edge normal
            min_a, max_a = _project_poly(poly_a, axis)
            min_b, max_b = _project_poly(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False   # separating axis found -> disjoint
    return True


def _closest_point_on_seg(p, a, b):
    ax, ay = a
    abx, aby = b[0] - ax, b[1] - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return ax, ay
    t = ((p[0] - ax) * abx + (p[1] - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))
    return ax + abx * t, ay + aby * t


def poly_circle_overlap(poly, cx, cy, r):
    """True if circle (cx, cy, r) overlaps convex poly."""
    if point_in_convex_poly(cx, cy, poly):
        return True
    n = len(poly)
    r2 = r * r
    for i in range(n):
        qx, qy = _closest_point_on_seg((cx, cy), poly[i], poly[(i + 1) % n])
        dx, dy = qx - cx, qy - cy
        if dx * dx + dy * dy <= r2:
            return True
    return False


class HullCollision:
    """World-space collision queries for one hull's (convex) polygon.

    The polygon is stored in hull-local coords. Queries transform through
    the ship's current pos + angle so they track the rendered hull.
    """
    def __init__(self, local_poly, inset=1.0):
        # inset scales the polygon toward its centroid (1.0 = exact hull)
        if inset != 1.0:
            cx = sum(p[0] for p in local_poly) / len(local_poly)
            cy = sum(p[1] for p in local_poly) / len(local_poly)
            local_poly = [(cx + (p[0] - cx) * inset,
                           cy + (p[1] - cy) * inset) for p in local_poly]
        self.local_poly = local_poly
        self.circum_radius = max(math.hypot(p[0], p[1]) for p in local_poly)

    def _to_local(self, wx, wy, pos, angle):
        fx, fy = math.cos(angle), math.sin(angle)
        rx, ry = -fy, fx
        dx, dy = wx - pos[0], wy - pos[1]
        return dx * fx + dy * rx, dx * fy + dy * ry

    def _to_world(self, lx, ly, pos, angle):
        fx, fy = math.cos(angle), math.sin(angle)
        rx, ry = -fy, fx
        return pos[0] + fx * lx + rx * ly, pos[1] + fy * lx + ry * ly

    def contains_point(self, wx, wy, pos, angle):
        lx, ly = self._to_local(wx, wy, pos, angle)
        return point_in_convex_poly(lx, ly, self.local_poly)

    def segment_hits(self, p1, p2, pos, angle):
        """p1, p2 are world (x, y). Returns (hit, world_hit_point)."""
        l1 = self._to_local(p1[0], p1[1], pos, angle)
        l2 = self._to_local(p2[0], p2[1], pos, angle)
        hit, lpt = segment_hits_convex_poly(l1, l2, self.local_poly)
        if not hit:
            return False, None
        return True, self._to_world(lpt[0], lpt[1], pos, angle)

    def overlaps_ship(self, other, pos, angle, opos, oangle):
        a = [self._to_world(lx, ly, pos, angle) for (lx, ly) in self.local_poly]
        b = [other._to_world(lx, ly, opos, oangle)
             for (lx, ly) in other.local_poly]
        return polys_overlap(a, b)

    def overlaps_circle(self, cx, cy, r, pos, angle):
        lx, ly = self._to_local(cx, cy, pos, angle)
        return poly_circle_overlap(self.local_poly, lx, ly, r)

    def swept_overlaps_circle(self, prev_pos, prev_angle, pos, angle, cx, cy, r):
        """True if the hull polygon, swept from (prev_pos, prev_angle) to
        (pos, angle), overlaps circle (cx, cy, r).

        Per-step motion is approximated as a pure translation from the
        previous world pose to the current one. The swept volume of a
        convex poly under translation is the convex hull of the two
        endpoint copies, so we hull (P0 + P1) and test the circle against
        it. This subsumes the point-in-time overlaps_circle test (P1 is
        inside the hull). Pure float -> lockstep-safe.
        """
        p0 = [self._to_world(lx, ly, prev_pos, prev_angle)
              for (lx, ly) in self.local_poly]
        p1 = [self._to_world(lx, ly, pos, angle)
              for (lx, ly) in self.local_poly]
        swept = convex_hull(p0 + p1)
        return poly_circle_overlap(swept, cx, cy, r)
