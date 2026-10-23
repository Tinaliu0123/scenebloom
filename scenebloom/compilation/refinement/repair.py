"""Pose repair: move or turn an object so a confirmed relation holds."""
# Room bounds and new collisions are hard filters; zone bounds are only a score penalty.
# Overlap with the target is allowed (a chair tucks under its desk) if the centre stays out.

from __future__ import annotations

import math
from dataclasses import dataclass

from scenebloom.utils.schemas import PlacedObject
from scenebloom.compilation.fitting.obb import rect_corners
from scenebloom.compilation.refinement.verify import (
    beside_terms,
    CONFIRMED,
    face_front_side_applies,
    facing_error,
    footprint_gap,
    front_side_cos,
    FRONT_SIDE_MIN_COS,
    Verdict,
    WALL_NORMAL,
    wall_terms,
    yaw_to_face,
    face_error,
    nearest_point_on_footprint,
)

BAND_MARGIN = 0.15  # metres between a seat and the thing it serves
BAND_SAMPLES_PER_EDGE = 5
FACE_TARGET_TOLERANCE = math.radians(20.0)
# Gap kept from non-target neighbours: a 2D gap of 0 can already be a 3D mesh overlap.
COLLISION_MARGIN = 0.02
# Score weights: metres traded against radians of facing error.
W_DISPLACEMENT = 0.8
W_NEAR = 0.4
W_ZONE_OVERSHOOT = 2.0

# Rear-to-wall clearance after a wall repair, the same as the fit's wall glue.
WALL_CLEARANCE = 0.05


#: Relations that read the same with subject and target swapped.
SYMMETRIC_OPERATORS = ("beside", "align_with")


@dataclass(frozen=True)
class RepairOutcome:
    """What happened to one confirmed constraint."""

    new_object: PlacedObject
    applied: bool


def _centre(obj: PlacedObject) -> tuple[float, float]:
    cx, _, cz = obj.pose.center
    return (cx, cz)


def _with_pose(
    obj: PlacedObject,
    *,
    yaw: float | None = None,
    xz: tuple[float, float] | None = None,
) -> PlacedObject:
    pose = obj.pose
    upd: dict = {}
    if yaw is not None:
        upd["rotation_y"] = yaw
    if xz is not None:
        upd["center"] = (xz[0], pose.center[1], xz[1])
    return obj.model_copy(update={"pose": pose.model_copy(update=upd)})


_HALF_PI = math.pi / 2


def _axis_snap(subject: PlacedObject, target: PlacedObject) -> PlacedObject:
    """Turn ``subject`` in place to the nearest ``yaw_t + k*90deg``."""
    yaw_s = subject.pose.rotation_y
    yaw_t = target.pose.rotation_y
    k = round((yaw_s - yaw_t) / _HALF_PI)
    return _with_pose(subject, yaw=yaw_t + k * _HALF_PI)


def propose_yaw_candidates(subject: PlacedObject, target: PlacedObject) -> list[float]:
    """Enumerated yaw candidates for aiming ``subject`` at ``target``."""
    cur = subject.pose.rotation_y
    face = yaw_to_face(_centre(subject), _centre(target))
    # The nearest footprint point: for a seat at a table edge this is the edge normal.
    edge = yaw_to_face(_centre(subject), nearest_point_on_footprint(_centre(subject), target))
    return [cur, cur + math.pi, edge, face, face + math.pi / 2, face - math.pi / 2]


def _corners(obj: PlacedObject) -> list[tuple[float, float]]:
    cx, _, cz = obj.pose.center
    sx, _, sz = obj.pose.size
    return rect_corners(cx, cz, sx, sz, obj.pose.rotation_y)


def _point_in_footprint(pt: tuple[float, float], obj: PlacedObject) -> bool:
    """Winding test against the oriented rectangle."""
    poly = _corners(obj)
    sign = 0
    for i in range(len(poly)):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % len(poly)]
        cross = (bx - ax) * (pt[1] - az) - (bz - az) * (pt[0] - ax)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _in_bounds(obj: PlacedObject, bounds: tuple[float, float, float, float]) -> bool:
    xmin, xmax, zmin, zmax = bounds
    return all(xmin <= x <= xmax and zmin <= z <= zmax for x, z in _corners(obj))


def _overshoot(obj: PlacedObject, bounds: tuple[float, float, float, float]) -> float:
    """Furthest distance any corner sticks out past ``bounds``; 0 if inside."""
    xmin, xmax, zmin, zmax = bounds
    c = _corners(obj)
    return max(
        max(xmin - x for x, _ in c),
        max(x - xmax for x, _ in c),
        max(zmin - z for _, z in c),
        max(z - zmax for _, z in c),
        0.0,
    )


def _collides_with(
    obj: PlacedObject,
    others: list[PlacedObject],
    *,
    ignore: set[str],
    margin: float = 0.0,
) -> set[str]:
    """Ids of ``others`` whose footprint gap to ``obj`` is below ``margin``."""
    return {
        o.instance_id
        for o in others
        if o.instance_id not in ignore and footprint_gap(obj, o) < margin
    }


def _feasible(
    cand: PlacedObject,
    *,
    current: PlacedObject,
    target: PlacedObject,
    others: list[PlacedObject],
    room_bounds: tuple[float, float, float, float] | None,
) -> str | None:
    """``None`` if the candidate passes the hard checks, else the reason it fails."""
    if room_bounds is not None and not _in_bounds(cand, room_bounds):
        return "leaves the room"
    if _point_in_footprint(_centre(cand), target):
        return "centre inside the target (swallowed, not tucked in)"
    ignore = {cand.instance_id, target.instance_id}
    new = _collides_with(cand, others, ignore=ignore, margin=COLLISION_MARGIN) - (
        _collides_with(current, others, ignore=ignore, margin=COLLISION_MARGIN)
    )
    if new:
        return f"newly collides with {sorted(new)}"
    bad = _corridor_entered(cand, current, others, ignore)
    if bad:
        return f"enters another zone's corridor ({bad})"
    return None


def _env_gap(a, b) -> float:
    dx = max(a[0] - b[1], b[0] - a[1], 0.0)
    dz = max(a[2] - b[3], b[2] - a[3], 0.0)
    return math.hypot(dx, dz) if (dx > 0.0 and dz > 0.0) else max(dx, dz)


def _corridor_entered(cand: PlacedObject, current: PlacedObject, others: list[PlacedObject], ignore: set) -> str | None:
    """Another zone's object the move newly brings inside the zone corridor, if any."""
    from scenebloom.compilation.fitting.within_zone import _obj_aabb
    from scenebloom.compilation.fitting.among_zones import ZONE_GAP_M
    need = ZONE_GAP_M - 1e-3
    zc = getattr(cand, "zone_id", None)
    if not zc:
        return None
    ac, acur = _obj_aabb(cand), _obj_aabb(current)
    for o in others:
        if o.instance_id in ignore or not getattr(o, "zone_id", None) or o.zone_id == zc:
            continue
        ao = _obj_aabb(o)
        if _env_gap(ac, ao) < need and _env_gap(acur, ao) >= need:
            return o.instance_id
    return None


def interaction_band_samples(
    target: PlacedObject, subject: PlacedObject, *, margin: float = BAND_MARGIN
) -> list[tuple[float, float]]:
    """Points along the four bands just outside ``target``'s edges, ``margin`` clear of it."""
    tx, tz = _centre(target)
    tsx, _, tsz = target.pose.size
    ssx, _, ssz = subject.pose.size
    yaw = target.pose.rotation_y
    # Target's local +x and +z in world XZ (rect_corners rotates by +yaw).
    ux, uz = math.cos(yaw), math.sin(yaw)
    vx, vz = -math.sin(yaw), math.cos(yaw)

    reach = max(ssx, ssz) / 2.0
    half_u = tsx / 2.0 + reach + margin
    half_v = tsz / 2.0 + reach + margin

    pts: list[tuple[float, float]] = []
    n = BAND_SAMPLES_PER_EDGE
    for sign in (1.0, -1.0):
        for i in range(n):  # bands normal to local x: slide along local z
            t = (i / (n - 1) - 0.5) * tsz if n > 1 else 0.0
            pts.append((tx + ux * half_u * sign + vx * t, tz + uz * half_u * sign + vz * t))
        for i in range(n):  # bands normal to local z: slide along local x
            t = (i / (n - 1) - 0.5) * tsx if n > 1 else 0.0
            pts.append((tx + vx * half_v * sign + ux * t, tz + vz * half_v * sign + uz * t))
    return pts


def _score(
    cand: PlacedObject,
    *,
    target: PlacedObject,
    origin: tuple[float, float],
    zone_bounds: tuple[float, float, float, float] | None = None,
) -> float:
    """Candidate score, higher is better; zone overshoot is a penalty, never a veto."""
    err = face_error(cand, target)
    gap = max(0.0, footprint_gap(cand, target))
    disp = math.hypot(cand.pose.center[0] - origin[0], cand.pose.center[2] - origin[1])
    over = _overshoot(cand, zone_bounds) if zone_bounds is not None else 0.0
    return (
        -err
        - W_NEAR * gap
        - W_DISPLACEMENT * disp
        - W_ZONE_OVERSHOOT * over
    )


def _wall_yaw_for_back(wall: str) -> float:
    """Yaw whose BACK (-front) points along the wall's outward normal."""
    nx, nz = WALL_NORMAL[wall]
    return math.atan2(-nx, nz)


def _wall_yaw_for_axis(subject: PlacedObject, wall: str) -> float:
    """Smallest 90-deg step that lays the footprint's long side along the wall."""
    nx, nz = WALL_NORMAL[wall]
    cur = subject.pose.rotation_y
    for k in (0, 1, -1, 2):
        pts = _corners(_with_pose(subject, yaw=cur + k * _HALF_PI))
        along_n = [x * nx + z * nz for x, z in pts]
        along_w = [-x * nz + z * nx for x, z in pts]
        if max(along_n) - min(along_n) <= max(along_w) - min(along_w) + 1e-9:
            return cur + k * _HALF_PI
    return cur


#: Offsets tried along the wall when the straight slide to it is blocked.
WALL_LATERAL_STEP = 0.10
WALL_LATERAL_MAX = 1.50


def _lateral_offsets() -> list[float]:
    out: list[float] = []
    k = 1
    while k * WALL_LATERAL_STEP <= WALL_LATERAL_MAX + 1e-9:
        out.extend((k * WALL_LATERAL_STEP, -k * WALL_LATERAL_STEP))
        k += 1
    return out


def _feasible_free(
    cand: PlacedObject,
    *,
    current: PlacedObject,
    others: list[PlacedObject],
    room_bounds: tuple[float, float, float, float] | None,
) -> str | None:
    """:func:`_feasible` for a claim with no object target (wall contact)."""
    if room_bounds is not None and not _in_bounds(cand, room_bounds):
        return "leaves the room"
    ignore = {cand.instance_id}
    new = _collides_with(cand, others, ignore=ignore, margin=COLLISION_MARGIN) - (
        _collides_with(current, others, ignore=ignore, margin=COLLISION_MARGIN)
    )
    if new:
        return f"newly collides with {sorted(new)}"
    return None


def _repair_wall_contact(
    v: Verdict,
    subject: PlacedObject,
    others: list[PlacedObject],
    *,
    room_bounds: tuple[float, float, float, float] | None,
    is_anchor: bool,
) -> RepairOutcome:
    """``back_to(wall)``: turn the rear to the wall, then slide it to ``WALL_CLEARANCE``.

    A blocked slide also shifts along the wall, smallest offset first. The anchor keeps its yaw.
    """
    c = v.constraint
    wall = c.target_wall
    if room_bounds is None or wall not in WALL_NORMAL:
        return RepairOutcome(subject, applied=False)
    yaw_only = c.repair_scope == "yaw_only"
    cand = subject
    t = wall_terms(cand, wall, room_bounds, c.subject_interface)
    if not t.yaw_ok and not is_anchor:
        yaw = (_wall_yaw_for_back(wall) if c.subject_interface == "back"
               else _wall_yaw_for_axis(cand, wall))
        cand = _with_pose(cand, yaw=yaw)
    t = wall_terms(cand, wall, room_bounds, c.subject_interface)
    if not t.gap_ok:
        if yaw_only:
            return RepairOutcome(subject, applied=False)
        shift = t.gap - WALL_CLEARANCE
        nx, nz = WALL_NORMAL[wall]
        cx, _, cz = cand.pose.center
        turned = cand
        cand = _with_pose(cand, xz=(cx + nx * shift, cz + nz * shift))
        reason = _feasible_free(cand, current=subject, others=others, room_bounds=room_bounds)
        if reason is not None and "collides" in reason:
            tx, tz = -nz, nx  # along the wall
            for off in _lateral_offsets():
                alt = _with_pose(turned, xz=(cx + nx * shift + tx * off,
                                             cz + nz * shift + tz * off))
                if _feasible_free(alt, current=subject, others=others,
                                  room_bounds=room_bounds) is None:
                    cand = alt
                    reason = None
                    break
        if reason is not None:
            return RepairOutcome(subject, applied=False)
    if cand is subject:
        return RepairOutcome(subject, applied=False)
    if _feasible_free(cand, current=subject, others=others, room_bounds=room_bounds) is not None:
        return RepairOutcome(subject, applied=False)
    return RepairOutcome(cand, applied=True)


def _repair_one(
    v: Verdict,
    objects_by_id: dict[str, PlacedObject],
    *,
    zone_bounds: tuple[float, float, float, float] | None,
    room_bounds: tuple[float, float, float, float] | None,
    immovable: set[str],
    context: list[PlacedObject],
) -> RepairOutcome | None:
    c = v.constraint
    subject = objects_by_id.get(c.subject_id)
    if subject is None:
        return None
    if c.operator == "back_to" and c.target_type == "wall":
        return _repair_wall_contact(
            v, subject, list(objects_by_id.values()) + context,
            room_bounds=room_bounds, is_anchor=c.subject_id in immovable,
        )
    swapped = False
    if (c.subject_id in immovable and c.target_id
            and c.target_id in objects_by_id and c.target_id not in immovable
            and c.operator in SYMMETRIC_OPERATORS):
        # Symmetric relation with an immovable subject: move the target instead.
        subject, swapped = objects_by_id[c.target_id], True
    if c.subject_id in immovable and not swapped:
        return RepairOutcome(subject, applied=False)
    target = objects_by_id.get(c.subject_id if swapped else c.target_id) if c.target_id else None
    if target is None:
        return RepairOutcome(subject, applied=False)

    others = list(objects_by_id.values()) + context
    origin = _centre(subject)
    yaw_only = c.repair_scope == "yaw_only"

    # beside is positional: the fix is a flanking slot along the target's local x.
    # When only the axis term fails, an in-place snap is enough.
    if c.operator == "beside":
        t = beside_terms(subject, target)
        if t.gap_ok and t.sideways and not t.axis_ok:
            cand = _axis_snap(subject, target)
            if _feasible(
                cand, current=subject, target=target, others=others,
                room_bounds=room_bounds,
            ) is None:
                return RepairOutcome(cand, applied=True)
        if yaw_only:
            return RepairOutcome(subject, applied=False)
        yaw_t = target.pose.rotation_y
        cy, sy = math.cos(yaw_t), math.sin(yaw_t)
        tx, tz = _centre(target)
        half = target.pose.size[0] / 2 + subject.pose.size[0] / 2
        flank: list[tuple[float, PlacedObject]] = []
        for side in (1.0, -1.0):
            for slot_gap in (0.05, 0.15, 0.30):
                d = side * (half + slot_gap)
                cand = _with_pose(
                    subject, yaw=yaw_t, xz=(tx + cy * d, tz + sy * d)
                )
                if _feasible(
                    cand, current=subject, target=target, others=others,
                    room_bounds=room_bounds,
                ):
                    continue
                flank.append(
                    (
                        _score(cand, target=target, origin=origin,
                               zone_bounds=zone_bounds),
                        cand,
                    )
                )
        if not flank:
            return RepairOutcome(subject, applied=False)
        return RepairOutcome(max(flank, key=lambda p: p[0])[1], applied=True)

    # align_with is an axis claim, not a facing claim: snap in place or give up.
    if c.operator == "align_with":
        cand = _axis_snap(subject, target)
        if _feasible(
            cand, current=subject, target=target, others=others,
            room_bounds=room_bounds,
        ) is not None:
            return RepairOutcome(subject, applied=False)
        return RepairOutcome(cand, applied=True)

    # face on a target with a front also needs the subject on that front side;
    # turning cannot fix that, so translation is allowed even in a yaw-only scope.
    front_side = face_front_side_applies(c) and not swapped
    on_front_side = (not front_side) or front_side_cos(target, origin) >= FRONT_SIDE_MIN_COS
    if front_side and not on_front_side:
        yaw_only = False
    # Rotate in place first, the minimal edit.
    in_place: list[tuple[float, PlacedObject]] = []
    for yaw in (propose_yaw_candidates(subject, target) if on_front_side else ()):
        cand = _with_pose(subject, yaw=yaw)
        if (face_error(cand, target) if c.operator == "face" else facing_error(cand, _centre(target))) > FACE_TARGET_TOLERANCE:
            continue
        if _feasible(
            cand, current=subject, target=target, others=others,
            room_bounds=room_bounds,
        ):
            continue
        in_place.append(
            (
                _score(cand, target=target, origin=origin, zone_bounds=zone_bounds),
                cand,
            )
        )

    # move_near is about distance, which turning cannot fix.
    if in_place and c.operator != "move_near":
        return RepairOutcome(max(in_place, key=lambda p: p[0])[1], applied=True)

    if yaw_only:
        if in_place:
            return RepairOutcome(max(in_place, key=lambda p: p[0])[1], applied=True)
        return RepairOutcome(subject, applied=False)

    scored: list[tuple[float, PlacedObject]] = []
    for px, pz in interaction_band_samples(target, subject):
        if front_side and front_side_cos(target, (px, pz)) < FRONT_SIDE_MIN_COS:
            continue
        yaw = yaw_to_face((px, pz), _centre(target))
        cand = _with_pose(subject, yaw=yaw, xz=(px, pz))
        if _feasible(
            cand, current=subject, target=target, others=others,
            room_bounds=room_bounds,
        ):
            continue
        scored.append(
            (
                _score(cand, target=target, origin=origin, zone_bounds=zone_bounds),
                cand,
            )
        )
    if not scored:
        return RepairOutcome(subject, applied=False)
    return RepairOutcome(max(scored, key=lambda p: p[0])[1], applied=True)


def repair_verdicts(
    verdicts: list[Verdict],
    objects_by_id: dict[str, PlacedObject],
    *,
    zone_bounds: tuple[float, float, float, float] | None = None,
    room_bounds: tuple[float, float, float, float] | None = None,
    immovable: set[str] | None = None,
    context: list[PlacedObject] | None = None,
) -> list[RepairOutcome]:
    """Repair every CONFIRMED verdict; inputs are never mutated.

    ``zone_bounds`` is a soft preference, ``room_bounds`` the hard limit. ``context`` holds
    the objects outside the zone, which candidates must not collide with either.
    Outcomes with ``applied=False`` are returned too.
    """
    immovable = immovable or set()
    out: list[RepairOutcome] = []
    for v in verdicts:
        if v.verdict != CONFIRMED:
            continue
        res = _repair_one(
            v, objects_by_id, zone_bounds=zone_bounds, room_bounds=room_bounds,
            immovable=immovable, context=context or [],
        )
        if res is not None:
            out.append(res)
    return out
