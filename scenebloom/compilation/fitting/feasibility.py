"""Final feasibility pass: residual OBB de-collision under the wall locks, then room containment."""

from __future__ import annotations

import json
import logging
from typing import Any

from scenebloom.compilation.fitting.obb import obb_mtv, obb_overlap, rect_corners
from scenebloom.utils.geometry import obb_extent_xz, with_center
from scenebloom.utils.schemas import BBox3D, PlacedObject, SceneLayout

logger = logging.getLogger(__name__)

DECOLLIDE_MARGIN = 0.05

# Outward normal of each wall in world XZ.
_WALL_DIR = {"north": (0.0, 1.0), "south": (0.0, -1.0),
             "east": (1.0, 0.0), "west": (-1.0, 0.0)}


def decollide(
    objects: list[PlacedObject],
    *,
    margin: float,
    locked_walls: dict[str, str],
) -> list[PlacedObject]:
    """Separate overlapping footprints along the OBB minimum translation vector.

    ``locked_walls`` maps wall-glued anchor ids to their wall: those anchors never move,
    and nothing is pushed toward their wall.
    """
    if len(objects) < 2:
        return list(objects)

    # Scene min corner; only orders the visit.
    corner = (
        min(o.pose.center[0] - o.pose.size[0] / 2 for o in objects),
        min(o.pose.center[1] - o.pose.size[1] / 2 for o in objects),
        min(o.pose.center[2] - o.pose.size[2] / 2 for o in objects),
    )
    centers = {o.instance_id: o.pose.center for o in objects}
    sizes = {o.instance_id: o.pose.size for o in objects}
    yaws = {o.instance_id: o.pose.rotation_y for o in objects}

    def _dist2(iid: str) -> float:
        cx, cy, cz = centers[iid]
        return (cx - corner[0]) ** 2 + (cy - corner[1]) ** 2 + (cz - corner[2]) ** 2

    def _shift(iid: str, dx: float, dz: float) -> None:
        c = centers[iid]
        centers[iid] = (c[0] + dx, c[1], c[2] + dz)

    def _corners(iid: str):
        c, s = centers[iid], sizes[iid]
        return rect_corners(c[0], c[2], s[0], s[2], yaws[iid])

    def _y_overlap(i: str, j: str) -> bool:
        lo_i, hi_i = centers[i][1] - sizes[i][1] / 2, centers[i][1] + sizes[i][1] / 2
        lo_j, hi_j = centers[j][1] - sizes[j][1] / 2, centers[j][1] + sizes[j][1] / 2
        return (lo_i < hi_j) and (lo_j < hi_i)

    order = sorted((o.instance_id for o in objects), key=lambda i: (_dist2(i), i))
    unresolved: list[tuple[str, str]] = []
    for i, near_id in enumerate(order):
        for far_id in order[i + 1:]:
            ra, rb = _corners(near_id), _corners(far_id)
            if not (_y_overlap(near_id, far_id) and obb_overlap(ra, rb, margin=0.0)):
                continue
            res = obb_mtv(ra, rb, margin=margin)
            if res is None:
                continue
            (ux, uz), depth = res
            vx, vz = ux * depth, uz * depth
            plans = [
                [(far_id, vx, vz)],
                [(far_id, vx / 2, vz / 2), (near_id, -vx / 2, -vz / 2)],
                [(near_id, -vx, -vz)],
            ]

            def _step_ok(iid: str, dx: float, dz: float) -> bool:
                if iid in locked_walls:
                    return False
                other = far_id if iid == near_id else near_id
                w = locked_walls.get(other)
                if w in _WALL_DIR:
                    nx, nz = _WALL_DIR[w]
                    if dx * nx + dz * nz > 1e-9:
                        return False
                return True

            if locked_walls:
                plans = [p for p in plans if all(_step_ok(i, dx, dz) for i, dx, dz in p)]
            if not plans:
                unresolved.append((near_id, far_id))
                continue
            for iid, dx, dz in plans[0]:
                _shift(iid, dx, dz)

    if unresolved:
        logger.info(
            "decollide: %d pair(s) held by a wall lock: %s",
            len(unresolved), ", ".join(f"{n}|{f}" for n, f in unresolved[:5]),
        )
    out = [o if centers[o.instance_id] == o.pose.center else with_center(o, centers[o.instance_id])
           for o in objects]
    return out


def clamp_layout_to_room(
    layout: SceneLayout, *, max_room_m: float = 12.0
) -> SceneLayout:
    """Translate each room's objects as one rigid group into its room bbox.

    Rigid, because per-object clamping re-collides neighbours that de-collision separated.
    A group wider than the room aligns to the min wall and the room grows, up to ``max_room_m``.
    """
    try:
        rooms = list(layout.zoning_plan.rooms)
    except Exception:
        return layout
    if not rooms:
        return layout

    room_of_zone: dict[str, int] = {}
    for ri, room in enumerate(rooms):
        for z in getattr(room, "functional_zones", []):
            room_of_zone[str(z.zone_id)] = ri
    objs_of_room: dict[int, list[PlacedObject]] = {}
    for o in layout.placed_objects:
        ri = room_of_zone.get(str(o.zone_id))
        if ri is not None:
            objs_of_room.setdefault(ri, []).append(o)

    def _axis_shift(umin, umax, lo, hi):
        """Return (shift, regrow) to bring [umin,umax] inside [lo,hi]."""
        span, usz = hi - lo, umax - umin
        if usz <= span:
            if umin < lo:
                return lo - umin, False
            if umax > hi:
                return hi - umax, False
            return 0.0, False
        return lo - umin, True

    new_centers: dict[str, tuple[float, float, float]] = {}
    new_bbox_by_room: dict[int, BBox3D] = {}
    for ri, objs in objs_of_room.items():
        rb = getattr(rooms[ri], "bbox", None)
        if rb is None:
            continue
        lo_x, hi_x = rb.center[0] - rb.size[0] / 2, rb.center[0] + rb.size[0] / 2
        lo_z, hi_z = rb.center[2] - rb.size[2] / 2, rb.center[2] + rb.size[2] / 2
        exts = [(o.pose.center, obb_extent_xz(o)) for o in objs]
        uxmin = min(c[0] - hx for c, (hx, _) in exts)
        uxmax = max(c[0] + hx for c, (hx, _) in exts)
        uzmin = min(c[2] - hz for c, (_, hz) in exts)
        uzmax = max(c[2] + hz for c, (_, hz) in exts)
        dx, gx = _axis_shift(uxmin, uxmax, lo_x, hi_x)
        dz, gz = _axis_shift(uzmin, uzmax, lo_z, hi_z)
        if dx == 0.0 and dz == 0.0 and not gx and not gz:
            continue
        for o in objs:
            cx, cy, cz = o.pose.center
            new_centers[o.instance_id] = (cx + dx, cy, cz + dz)
        if gx or gz:
            nlo_x, nhi_x = min(lo_x, uxmin + dx), max(hi_x, uxmax + dx)
            nlo_z, nhi_z = min(lo_z, uzmin + dz), max(hi_z, uzmax + dz)
            nhi_x = min(nhi_x, nlo_x + max_room_m)
            nhi_z = min(nhi_z, nlo_z + max_room_m)
            new_bbox_by_room[ri] = BBox3D(
                center=((nlo_x + nhi_x) / 2.0, rb.center[1], (nlo_z + nhi_z) / 2.0),
                size=(nhi_x - nlo_x, rb.size[1], nhi_z - nlo_z),
                rotation_y=rb.rotation_y,
            )

    if not new_centers:
        return layout
    new_objs = [with_center(o, new_centers[o.instance_id])
                if o.instance_id in new_centers else o
                for o in layout.placed_objects]
    upd: dict[str, Any] = {"placed_objects": new_objs}
    if new_bbox_by_room:
        new_rooms = [r.model_copy(update={"bbox": new_bbox_by_room[i]})
                     if i in new_bbox_by_room else r for i, r in enumerate(rooms)]
        upd["zoning_plan"] = layout.zoning_plan.model_copy(update={"rooms": new_rooms})
    logger.info("clamp_layout_to_room: shifted %d obj as rigid group(s); %d room(s) regrown",
                len(new_centers), len(new_bbox_by_room))
    return layout.model_copy(update=upd)


def global_feasibility(
    all_packed: list[PlacedObject],
    locked: dict,
    packed_tmpl: dict,
) -> SceneLayout:
    """Resolve cross-zone and post-glue overlaps under the wall locks, then clamp to the room."""
    deco = decollide(all_packed, margin=DECOLLIDE_MARGIN, locked_walls=locked)
    lay5 = SceneLayout.model_validate(
        {
            **packed_tmpl,
            "placed_objects": [json.loads(o.model_dump_json()) for o in deco],
        }
    )
    return clamp_layout_to_room(lay5)
