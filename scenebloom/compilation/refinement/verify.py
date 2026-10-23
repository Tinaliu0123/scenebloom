"""Geometric check of each contract relation: is it violated by the current poses?"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scenebloom.utils.schemas import PlacedObject
from scenebloom.utils.geometry import facing_dir
from scenebloom.compilation.fitting.obb import obb_mtv, rect_corners
from scenebloom.compilation.refinement.constraints import Constraint

_HALF_PI = math.pi / 2.0

CONFIRMED = "confirmed"  # the relation is violated
REFUTED = "refuted"
UNVERIFIABLE = "unverifiable"

# Generous on purpose: the repair snaps to cardinals anyway.
FACE_TOLERANCE = math.radians(30.0)

# move_near: largest gap between footprint edges (not centres), about one reach.
NEAR_GAP_MAX = 0.60

# beside / align_with: long axes must agree mod 90 deg (an L-corner pair is aligned).
AXIS_TOLERANCE = math.radians(20.0)
# Footprints squarer than this have no visible long axis; the axis term is skipped.
AXIS_MIN_ASPECT = 1.2

# Wall contact holds when the footprint is within this of the wall plane.
WALL_GAP_MAX = 0.15
# A wall this far off the subject's rear is usually a left/right misread: UNVERIFIABLE.
WALL_YAW_IMPLAUSIBLE = math.radians(60.0)
#: Outward normal (x, z) of each wall; north = +z, east = +x.
WALL_NORMAL: dict[str, tuple[float, float]] = {
    "north": (0.0, 1.0), "south": (0.0, -1.0),
    "east": (1.0, 0.0), "west": (-1.0, 0.0),
}


#: ``face`` on a target with a unique front also needs the subject within a 60 deg
#: half cone in front of it, so a seat behind a desk does not count.
FRONT_SIDE_MIN_COS = math.cos(math.radians(60.0))


def nearest_cardinal_delta(yaw: float) -> float:
    """``min_k |wrap(yaw - k*pi/2)|`` -- distance to the nearest cardinal, radians."""
    remainder = math.fmod(yaw, _HALF_PI)
    if remainder > _HALF_PI / 2.0:
        remainder -= _HALF_PI
    elif remainder < -_HALF_PI / 2.0:
        remainder += _HALF_PI
    return abs(remainder)


def face_front_side_applies(c: "Constraint") -> bool:
    """The front-side term is part of THIS claim's predicate."""
    return c.operator == "face" and bool(getattr(c, "target_has_front", False))


def front_side_cos(target: PlacedObject, xz: tuple[float, float]) -> float:
    """Cosine between ``target``'s front and the direction to ``xz``; 1 in front, -1 behind."""
    tx, tz = _centre_xz(target)
    dx, dz = xz[0] - tx, xz[1] - tz
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return 1.0
    fx, fz = facing_dir(target.pose.rotation_y)
    return max(-1.0, min(1.0, (fx * dx + fz * dz) / norm))


def axis_error(subject: PlacedObject, target: PlacedObject) -> float:
    """``min_k |wrap(yaw_s - yaw_t - k*90deg)|`` in ``[0, 45deg]``, radians."""
    return nearest_cardinal_delta(subject.pose.rotation_y - target.pose.rotation_y)


def has_long_axis(obj: PlacedObject) -> bool:
    """True when the XZ footprint is elongated enough for its yaw to be visible."""
    sx, _, sz = obj.pose.size
    lo, hi = sorted((abs(sx), abs(sz)))
    return lo > 1e-9 and hi / lo >= AXIS_MIN_ASPECT


@dataclass(frozen=True)
class BesideTerms:
    """Terms of the ``beside`` predicate; ``axis`` is None for a near-square subject."""

    gap: float
    dx_local: float
    dz_local: float
    axis: float | None

    @property
    def sideways(self) -> bool:
        return abs(self.dx_local) > abs(self.dz_local)

    @property
    def gap_ok(self) -> bool:
        return self.gap <= NEAR_GAP_MAX

    @property
    def axis_ok(self) -> bool:
        return self.axis is None or self.axis <= AXIS_TOLERANCE

    @property
    def violated(self) -> bool:
        return not (self.gap_ok and self.sideways and self.axis_ok)


def beside_terms(subject: PlacedObject, target: PlacedObject) -> BesideTerms:
    """Measure ``beside`` for the current poses."""
    gap = footprint_gap(subject, target)
    tx, tz = _centre_xz(target)
    sx, sz = _centre_xz(subject)
    yaw = target.pose.rotation_y
    dx_local = (sx - tx) * math.cos(yaw) + (sz - tz) * math.sin(yaw)
    dz_local = -(sx - tx) * math.sin(yaw) + (sz - tz) * math.cos(yaw)
    axis = None
    if has_long_axis(subject):
        axis = axis_error(subject, target)
    return BesideTerms(gap=gap, dx_local=dx_local, dz_local=dz_local, axis=axis)


def wall_plane(wall: str, room_bounds: tuple[float, float, float, float]) -> float:
    """``n . p`` for any point ``p`` on the wall: the wall's plane coordinate."""
    xmin, xmax, zmin, zmax = room_bounds
    return {"north": zmax, "south": -zmin, "east": xmax, "west": -xmin}[wall]


@dataclass(frozen=True)
class WallTerms:
    """Terms of a wall-contact claim."""

    gap: float             # metres from the nearest footprint extent to the wall plane
    yaw_err: float | None  # radians; None for a position-only ("body") claim
    yaw_tol: float

    @property
    def gap_ok(self) -> bool:
        return self.gap <= WALL_GAP_MAX

    @property
    def yaw_ok(self) -> bool:
        return self.yaw_err is None or self.yaw_err <= self.yaw_tol

    @property
    def violated(self) -> bool:
        return not (self.gap_ok and self.yaw_ok)


def _long_axis_dir(obj: PlacedObject) -> tuple[float, float]:
    """World direction of the footprint's long side."""
    sx, _, sz = obj.pose.size
    yaw = obj.pose.rotation_y
    if abs(sx) >= abs(sz):
        return (math.cos(yaw), math.sin(yaw))     # local +x
    return (-math.sin(yaw), math.cos(yaw))        # local +z


def wall_terms(
    subject: PlacedObject,
    wall: str,
    room_bounds: tuple[float, float, float, float],
    interface: str = "back",
) -> WallTerms:
    """Measure ``back_to(wall)``; a negative ``gap`` means through the wall.

    The yaw term: ``back`` must point at the wall, ``structural_long_axis`` must run
    along it, ``body`` has none.
    """
    nx, nz = WALL_NORMAL[wall]
    gap = wall_plane(wall, room_bounds) - max(
        x * nx + z * nz for x, z in _corners(subject)
    )
    if interface == "back":
        fx, fz = facing_dir(subject.pose.rotation_y)
        dot = max(-1.0, min(1.0, -fx * nx - fz * nz))
        return WallTerms(gap=gap, yaw_err=math.acos(dot), yaw_tol=FACE_TOLERANCE)
    if interface == "structural_long_axis" and has_long_axis(subject):
        ux, uz = _long_axis_dir(subject)
        tx, tz = -nz, nx                             # along the wall
        cross = abs(ux * tz - uz * tx)
        dot = abs(ux * tx + uz * tz)
        return WallTerms(
            gap=gap, yaw_err=math.atan2(cross, dot), yaw_tol=AXIS_TOLERANCE
        )
    return WallTerms(gap=gap, yaw_err=None, yaw_tol=0.0)


@dataclass(frozen=True)
class Verdict:
    """One constraint, re-checked against geometry."""

    constraint: Constraint
    verdict: str


def yaw_to_face(from_xz: tuple[float, float], to_xz: tuple[float, float]) -> float:
    """Yaw that points the front of an object at ``from_xz`` towards ``to_xz``.

    ``front(yaw) = (sin yaw, -cos yaw)``, so ``yaw = atan2(dx, -dz)``.
    """
    dx = to_xz[0] - from_xz[0]
    dz = to_xz[1] - from_xz[1]
    return math.atan2(dx, -dz)


def _centre_xz(obj: PlacedObject) -> tuple[float, float]:
    cx, _, cz = obj.pose.center
    return (cx, cz)


def _target_corners(target: PlacedObject) -> list[tuple[float, float]]:
    from scenebloom.compilation.fitting.obb import rect_corners
    cx, _, cz = target.pose.center
    sx, _, sz = target.pose.size
    return rect_corners(cx, cz, sx, sz, target.pose.rotation_y)


def _ray_hits_segment(o, d, a, b) -> bool:
    """Does the ray o + t*d (t >= 0) meet segment ab?"""
    ex, ez = b[0] - a[0], b[1] - a[1]
    den = d[0] * ez - d[1] * ex
    if abs(den) < 1e-12:
        return False
    ax, az = a[0] - o[0], a[1] - o[1]
    t = (ax * ez - az * ex) / den
    u = (ax * d[1] - az * d[0]) / den
    return t >= 0.0 and 0.0 <= u <= 1.0


def nearest_point_on_footprint(pt: tuple[float, float], target: PlacedObject) -> tuple[float, float]:
    """Closest point of ``target``'s oriented footprint to ``pt``."""
    cx, _, cz = target.pose.center
    sx, _, sz = target.pose.size
    yaw = target.pose.rotation_y
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dz = pt[0] - cx, pt[1] - cz
    lx, lz = c * dx + s * dz, -s * dx + c * dz         # world -> local (inverse rotation)
    lx = max(-sx / 2.0, min(sx / 2.0, lx))
    lz = max(-sz / 2.0, min(sz / 2.0, lz))
    return (cx + c * lx - s * lz, cz + s * lx + c * lz)


def facing_error_footprint(obj: PlacedObject, target: PlacedObject) -> float:
    """0 if ``obj`` faces into ``target``'s footprint, else the angle to its nearest corner."""
    o = _centre_xz(obj)
    d = facing_dir(obj.pose.rotation_y)
    cs = _target_corners(target)
    for i in range(4):
        if _ray_hits_segment(o, d, cs[i], cs[(i + 1) % 4]):
            return 0.0
    return min(facing_error(obj, c) for c in cs)


def face_error(obj: PlacedObject, target: PlacedObject) -> float:
    """The ``face`` error, measured against the target's footprint rather than its centre."""
    return facing_error_footprint(obj, target)


def facing_error(obj: PlacedObject, target_xz: tuple[float, float]) -> float:
    """Angle in ``[0, pi]`` between ``obj``'s front and the direction to target."""
    cx, cz = _centre_xz(obj)
    dx, dz = target_xz[0] - cx, target_xz[1] - cz
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return 0.0
    fx, fz = facing_dir(obj.pose.rotation_y)
    dot = max(-1.0, min(1.0, fx * dx / norm + fz * dz / norm))
    return math.acos(dot)


def _corners(obj: PlacedObject) -> list[tuple[float, float]]:
    cx, _, cz = obj.pose.center
    sx, _, sz = obj.pose.size
    return rect_corners(cx, cz, sx, sz, obj.pose.rotation_y)


def footprint_gap(a: PlacedObject, b: PlacedObject) -> float:
    """Distance between two oriented footprints; minus the penetration depth if they overlap."""
    ca, cb = _corners(a), _corners(b)
    mtv = obb_mtv(ca, cb)
    if mtv is not None:
        return -abs(mtv[1])
    # Disjoint: smallest corner-to-edge distance.
    best = float("inf")
    for poly_p, poly_q in ((ca, cb), (cb, ca)):
        for px, pz in poly_p:
            for i in range(len(poly_q)):
                qx, qz = poly_q[i]
                rx, rz = poly_q[(i + 1) % len(poly_q)]
                ex, ez = rx - qx, rz - qz
                seg2 = ex * ex + ez * ez
                t = 0.0 if seg2 < 1e-12 else ((px - qx) * ex + (pz - qz) * ez) / seg2
                t = max(0.0, min(1.0, t))
                best = min(best, math.hypot(px - (qx + t * ex), pz - (qz + t * ez)))
    return best


def _verify_one(
    c: Constraint,
    objects_by_id: dict[str, PlacedObject],
    room_bounds: tuple[float, float, float, float] | None = None,
) -> Verdict:
    subject = objects_by_id.get(c.subject_id)
    if subject is None:
        return Verdict(c, UNVERIFIABLE)

    if c.operator == "back_to" and c.target_type == "wall":
        wall = c.target_wall
        if room_bounds is None or wall not in WALL_NORMAL:
            return Verdict(c, UNVERIFIABLE)
        t = wall_terms(subject, wall, room_bounds, c.subject_interface)
        if t.yaw_err is not None and t.yaw_err > WALL_YAW_IMPLAUSIBLE:
            return Verdict(c, UNVERIFIABLE)
        return Verdict(c, CONFIRMED if t.violated else REFUTED)

    if c.operator == "face":
        target = objects_by_id.get(c.target_id) if c.target_id is not None else None
        if target is None:
            return Verdict(c, UNVERIFIABLE)
        violated = face_error(subject, target) > FACE_TOLERANCE
        if face_front_side_applies(c) and front_side_cos(target, _centre_xz(subject)) < FRONT_SIDE_MIN_COS:
            violated = True
        return Verdict(c, CONFIRMED if violated else REFUTED)

    if c.operator == "move_near":
        target = objects_by_id.get(c.target_id) if c.target_id is not None else None
        if target is None:
            return Verdict(c, UNVERIFIABLE)
        violated = footprint_gap(subject, target) > NEAR_GAP_MAX
        return Verdict(c, CONFIRMED if violated else REFUTED)

    if c.operator == "align_with":
        target = objects_by_id.get(c.target_id) if c.target_id else None
        if target is not None:
            # Object target: the long axes must be parallel or perpendicular.
            if not has_long_axis(subject):
                return Verdict(c, UNVERIFIABLE)
            violated = axis_error(subject, target) > AXIS_TOLERANCE
            return Verdict(c, CONFIRMED if violated else REFUTED)
        if c.target_axis not in ("x", "z"):
            return Verdict(c, UNVERIFIABLE)
        # Zone axis: parallel and perpendicular both count, i.e. the cardinal distance.
        violated = nearest_cardinal_delta(subject.pose.rotation_y) > FACE_TOLERANCE
        return Verdict(c, CONFIRMED if violated else REFUTED)

    if c.operator == "beside":
        target = objects_by_id.get(c.target_id) if c.target_id is not None else None
        if target is None:
            return Verdict(c, UNVERIFIABLE)
        # Close, offset along the target's side axis, and aligned mod 90 deg.
        return Verdict(c, CONFIRMED if beside_terms(subject, target).violated else REFUTED)

    return Verdict(c, UNVERIFIABLE)


def verify_constraints(
    constraints: list[Constraint],
    objects_by_id: dict[str, PlacedObject],
    *,
    room_bounds: tuple[float, float, float, float] | None = None,
) -> list[Verdict]:
    """Re-check every constraint against geometry; one verdict per constraint, in order."""
    return [_verify_one(c, objects_by_id, room_bounds=room_bounds) for c in constraints]
