"""Within-zone fitting: de-collide a zone's objects around its pinned anchor, then size the zone."""

from __future__ import annotations

import math

from scenebloom.utils.schemas import PlacedObject
from scenebloom.compilation.fitting.obb import _project, obb_mtv, rect_corners
from scenebloom.utils.geometry import with_center

#: Room-inward unit normal of each wall (north wall = max Z).
_WALL_INWARD = {"north": (0.0, -1.0), "south": (0.0, 1.0),
                "east": (-1.0, 0.0), "west": (1.0, 0.0)}


def _push_along(a, b, axis, margin: float) -> float:
    """How far ``b`` must move along +axis until its projection clears ``a``."""
    _, amax = _project(a, axis)
    bmin, _ = _project(b, axis)
    return max(0.0, amax - bmin + margin)


#: Contract relations meaning "a stands beside b": the cheapest MTV often slides one behind
#: the other, so these pairs are pushed along the reference object's width axis instead.
LATERAL_OPERATORS = ("beside", "align_with")


#: cos(45 deg): an MTV this close to the lateral axis already counts as sideways.
_LATERAL_COS = 0.7071067811865476


def _lateral_mtv(a, b, mtv, margin: float, ref_yaw: float):
    """Redirect a front/back MTV along the reference object's width axis (its local +x)."""
    lat = (math.cos(ref_yaw), math.sin(ref_yaw))
    (ux, uz), _depth = mtv
    if abs(ux * lat[0] + uz * lat[1]) >= _LATERAL_COS:
        return mtv
    best = None
    for axis in (lat, (-lat[0], -lat[1])):
        d = _push_along(a, b, axis, margin)
        if d > 0.0 and (best is None or d < best[1]):
            best = (axis, d)
    if best is None:
        return mtv
    return best


def _wall_policy_mtv(a, b, ca, cb, mtv, wall: str, margin: float, others=()):
    """Push that moves ``b`` off the pinned ``a`` without moving it toward the wall.

    A wall-ward MTV is redirected along the wall to the side ``b``'s centre already lies on,
    which keeps the image's left/right order, or room-inward if that would hit one of ``others``.
    """
    from scenebloom.compilation.fitting.obb import obb_overlap
    inward = _WALL_INWARD.get(wall)
    (ux, uz), _depth = mtv
    if inward is None:
        return mtv
    dot = ux * inward[0] + uz * inward[1]
    if dot >= -1e-9:
        return mtv
    along = (inward[1], -inward[0])
    off = (cb[0] - ca[0]) * along[0] + (cb[1] - ca[1]) * along[1]
    axis = inward
    if abs(off) > 1e-6:
        lat = along if off > 0.0 else (-along[0], -along[1])
        d_lat = _push_along(a, b, lat, margin)
        moved = [(x + lat[0] * d_lat, z + lat[1] * d_lat) for x, z in b]
        if d_lat > 0.0 and not any(obb_overlap(moved, o, margin=margin) for o in others):
            axis = lat
    d = _push_along(a, b, axis, margin)
    if d <= 0.0:
        return mtv
    return (axis, d)


_SEP_ITERS = 40   # fixed-point cap of the in-zone de-collision

Rect = tuple[float, float, float, float]  # (xmin, xmax, zmin, zmax)


def _obj_corners(o: PlacedObject):
    c, s = o.pose.center, o.pose.size
    return rect_corners(c[0], c[2], s[0], s[2], o.pose.rotation_y)


def _obj_aabb(o: PlacedObject) -> Rect:
    """Axis-aligned bounds of the object's rotated footprint."""
    cs = _obj_corners(o)
    xs = [p[0] for p in cs]
    zs = [p[1] for p in cs]
    return min(xs), max(xs), min(zs), max(zs)


def _clamp_into_zone(o: PlacedObject, rect: Rect) -> PlacedObject:
    """Translate ``o`` so its rotated-footprint AABB sits inside ``rect``."""
    x0, x1, z0, z1 = rect
    ax0, ax1, az0, az1 = _obj_aabb(o)
    hx, hz = (ax1 - ax0) / 2.0, (az1 - az0) / 2.0
    cx, cy, cz = o.pose.center
    fx, fz = (ax0 + ax1) / 2.0, (az0 + az1) / 2.0
    ncx = (x0 + x1) / 2.0 if (x1 - x0) < 2 * hx else min(max(fx, x0 + hx), x1 - hx)
    ncz = (z0 + z1) / 2.0 if (z1 - z0) < 2 * hz else min(max(fz, z0 + hz), z1 - hz)
    dx, dz = ncx - fx, ncz - fz
    if abs(dx) < 1e-9 and abs(dz) < 1e-9:
        return o
    return with_center(o, (cx + dx, cy, cz + dz))


def _decollide_obb_in_zone(
    members: list[PlacedObject], rect: Rect, *, margin: float,
    anchor_cat: str | None = None, wall_orientation: str | None = None,
    lateral_pairs: frozenset | None = None,
    exempt: frozenset = frozenset(),
) -> list[PlacedObject]:
    """Rotation-aware de-collision in ``rect``: the farther object of an overlapping pair moves.

    The ``anchor_cat`` object is never moved. ``lateral_pairs`` (frozensets of two ids) are
    separated sideways; in a wall-backed zone no push goes toward the wall.
    """
    if len(members) < 2:
        return members
    corner = (rect[0], rect[2])
    work = list(members)
    anchor_l = (anchor_cat or "").lower()
    lateral_on = bool(lateral_pairs)
    wall_policy = bool(wall_orientation)

    def _is_anchor(o: PlacedObject) -> bool:
        return bool(anchor_l) and (o.category or "").lower() == anchor_l

    order = sorted(
        range(len(work)),
        key=lambda i: (
            0 if _is_anchor(work[i]) else 1,
            (work[i].pose.center[0] - corner[0]) ** 2
            + (work[i].pose.center[2] - corner[1]) ** 2,
            work[i].instance_id,
        ),
    )
    for _ in range(_SEP_ITERS):
        moved = False
        for ii in range(len(order)):
            for jj in range(ii + 1, len(order)):
                ni, nj = order[ii], order[jj]
                if _is_anchor(work[nj]):
                    continue
                mtv = obb_mtv(_obj_corners(work[ni]), _obj_corners(work[nj]), margin=margin)
                if mtv is None:
                    continue
                if wall_policy and work[nj].instance_id not in exempt:
                    # ``exempt`` objects (licensed behind the anchor) keep the raw push.
                    _ca, _cb = work[ni].pose.center, work[nj].pose.center
                    mtv = _wall_policy_mtv(
                        _obj_corners(work[ni]), _obj_corners(work[nj]),
                        (_ca[0], _ca[2]), (_cb[0], _cb[2]), mtv, wall_orientation, margin,
                        others=[_obj_corners(work[k]) for k in order if k not in (ni, nj)])
                elif lateral_on and frozenset(
                        (work[ni].instance_id, work[nj].instance_id)) in lateral_pairs:
                    mtv = _lateral_mtv(
                        _obj_corners(work[ni]), _obj_corners(work[nj]), mtv, margin,
                        work[ni].pose.rotation_y or 0.0)
                (ux, uz), depth = mtv
                c = work[nj].pose.center
                shifted = with_center(work[nj], (c[0] + ux * depth, c[1], c[2] + uz * depth))
                work[nj] = _clamp_into_zone(shifted, rect)
                moved = True
        if not moved:
            break
    return work


def _fit_zone(
    members: list[PlacedObject], rect0: Rect, *, margin: float, max_dim_m: float,
    anchor_cat: str | None = None, wall_orientation: str | None = None,
    lateral_pairs: frozenset | None = None,
    exempt: frozenset = frozenset(),
) -> tuple[Rect, list[PlacedObject]]:
    """De-collide members without bounds, then set the zone rect to their footprint bbox.

    The rect never shrinks below ``rect0`` and is capped at ``max_dim_m`` per axis.
    """
    # Bounds wide enough that the in-zone clamp never acts.
    big = (rect0[0] - max_dim_m, rect0[1] + max_dim_m,
           rect0[2] - max_dim_m, rect0[3] + max_dim_m)
    work = _decollide_obb_in_zone(members, big, margin=margin, anchor_cat=anchor_cat,
                                  wall_orientation=wall_orientation,
                                  lateral_pairs=lateral_pairs, exempt=exempt)
    ax0 = min(_obj_aabb(o)[0] for o in work) - margin
    ax1 = max(_obj_aabb(o)[1] for o in work) + margin
    az0 = min(_obj_aabb(o)[2] for o in work) - margin
    az1 = max(_obj_aabb(o)[3] for o in work) + margin
    x0, x1 = min(rect0[0], ax0), max(rect0[1], ax1)
    z0, z1 = min(rect0[2], az0), max(rect0[3], az1)
    if x1 - x0 > max_dim_m:
        x1 = x0 + max_dim_m
    if z1 - z0 > max_dim_m:
        z1 = z0 + max_dim_m
    return (x0, x1, z0, z1), work
