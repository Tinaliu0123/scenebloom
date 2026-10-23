"""Among-zone assembly: place fitted zone envelopes, growing the room only if they do not fit."""

from __future__ import annotations

import json
import math
import logging
from typing import Optional

from scenebloom.compilation.fitting.within_zone import (
    _fit_zone,
    _obj_aabb,
)
from scenebloom.utils.geometry import with_center

logger = logging.getLogger(__name__)

#: Clearance between zones: two navigation-agent radii (2 x 0.2 m), so the agent can pass.
ZONE_GAP_M = 0.4
WALL_MARGIN = 0.05  # anchor back to wall clearance, metres
MAX_ROOM_M = 12.0

#: Fixed-point passes in ``_solve_axis``: the hi-wall target moves as the room grows.
_SOLVE_PASSES = 5


def zone_geoms(layout: dict) -> dict[str, dict]:
    """zone_id -> {wall, ox, oz, w, d, anchor_category} from zoning polygons."""
    out = {}
    for room in layout["zoning_plan"]["rooms"]:
        for z in room["functional_zones"]:
            xs = [p[0] for p in z["polygon_2d"]]
            zs = [p[1] for p in z["polygon_2d"]]
            out[z["zone_id"]] = {
                "wall": z.get("wall_orientation"),
                "ox": min(xs),
                "oz": min(zs),
                "w": max(xs) - min(xs),
                "d": max(zs) - min(zs),
                "anchor_category": z.get("anchor_category"),
            }
    return out


def room_bbox(layout: dict) -> tuple[float, float, float, float]:
    """(x0, x1, z0, z1) of the first room's box."""
    b = layout["zoning_plan"]["rooms"][0]["bbox"]
    cx, cz = b["center"][0], b["center"][2]
    w, d = b["size"][0], b["size"][2]
    return cx - w / 2, cx + w / 2, cz - d / 2, cz + d / 2


def wall_coord(wall: str, rb: tuple[float, float, float, float]) -> float:
    """World coordinate of a cardinal wall of the room box ``rb``."""
    x0, x1, z0, z1 = rb
    return {"north": z1, "south": z0, "east": x1, "west": x0}[wall]


def _geom_walls(px0, px1, pz0, pz1, rb, tol=1e-3):
    x0, x1, z0, z1 = rb
    h = []
    if abs(pz1 - z1) < tol: h.append("north")
    if abs(pz0 - z0) < tol: h.append("south")
    if abs(px1 - x1) < tol: h.append("east")
    if abs(px0 - x0) < tol: h.append("west")
    return h


def _edge_len(wall, px0, px1, pz0, pz1):
    return (px1 - px0) if wall in ("north", "south") else (pz1 - pz0)


def resolve_wall(vlm, px0, px1, pz0, pz1, rb):
    """The VLM's wall if the zone touches it, else the longest touched edge; None if none."""
    cand = _geom_walls(px0, px1, pz0, pz1, rb)
    if not cand:
        return None
    if vlm in cand:
        return vlm
    return max(cand, key=lambda w: _edge_len(w, px0, px1, pz0, pz1))


def _aabb_group(objs):
    xs0 = min(_obj_aabb(o)[0] for o in objs); xs1 = max(_obj_aabb(o)[1] for o in objs)
    zs0 = min(_obj_aabb(o)[2] for o in objs); zs1 = max(_obj_aabb(o)[3] for o in objs)
    return xs0, xs1, zs0, zs1


def _shift_axis(objs, axis, delta):
    out = []
    for o in objs:
        c = list(o.pose.center); c[axis] += delta
        out.append(with_center(o, (c[0], c[1], c[2])))
    return out


#: wall -> (axis index, direction of the wall along that axis)
_WALL_AXIS_DIR = {"north": (2, 1.0), "south": (2, -1.0), "east": (0, 1.0), "west": (0, -1.0)}
_NONANCHOR_FORWARD_TOL = 0.02


def _half_along(o, axis: int) -> float:
    sx, _, sz = o.pose.size
    c, s = abs(math.cos(o.pose.rotation_y)), abs(math.sin(o.pose.rotation_y))
    return (sx * c + sz * s) / 2.0 if axis == 0 else (sx * s + sz * c) / 2.0


def _glue_anchor_id(objs, wall: str | None, anchor_cat: str | None) -> str | None:
    """Object ``glue_zone_to_wall`` backs onto the wall: the anchor, else the deepest object."""
    ad = _WALL_AXIS_DIR.get(wall or "")
    if ad is None or not objs:
        return None
    axis, d = ad
    if anchor_cat:
        a = next((o for o in objs if o.category == anchor_cat), None)
        if a is not None:
            return a.instance_id
    return max(objs, key=lambda o: (o.pose.center[axis] * d + _half_along(o, axis),
                                    o.pose.size[0] * o.pose.size[2])).instance_id


def _slide_behind_anchor_forward(objs, anchor_id: str, wall: str | None, exempt=frozenset()):
    """Slide non-anchors behind the anchor's back plane onto it; returns (objs, shifts)."""
    ad = _WALL_AXIS_DIR.get(wall or "")
    anchor = next((o for o in objs if o.instance_id == anchor_id), None)
    if ad is None or anchor is None:
        return objs, []
    axis, d = ad
    plane = anchor.pose.center[axis] + d * _half_along(anchor, axis)
    out, shifts = [], []
    for o in objs:
        if o is anchor or o.instance_id in exempt:
            out.append(o); continue
        back = o.pose.center[axis] + d * _half_along(o, axis)
        excess = d * (back - plane)
        if excess > _NONANCHOR_FORWARD_TOL:
            c = list(o.pose.center); c[axis] -= d * excess
            out.append(with_center(o, tuple(c))); shifts.append((o.instance_id, round(excess, 3)))
        else:
            out.append(o)
    return out, shifts


def partition_resize_place(
    by_zone: dict,
    geoms: dict,
    template_dict: dict,
    *,
    forward_exempt: frozenset = frozenset(),
    lateral_pairs: Optional[frozenset] = None,
):
    """Pack all zones into the room; returns ``(all_packed, locked, packed_tmpl)``.

    ``locked`` maps each glued anchor id to its wall. ``forward_exempt`` ids sit behind their
    anchor by a contract relation and are not slid forward; ``lateral_pairs`` are id pairs the
    contract wants side by side.
    """
    from scenebloom.utils.schemas import SceneLayout
    from scenebloom.compilation.lifting.wall_anchor import (
        face_zone_inward,
        glue_zone_to_wall,
    )

    rb0 = room_bbox(template_dict)
    ox, oz = rb0[0], rb0[2]
    W0, D0 = rb0[1] - rb0[0], rb0[3] - rb0[2]

    # Phases 1-2: resolve each zone's wall, then fit the zone locally.
    Z = {}   # zone id -> wall, local objs, envelope size (wx, dz), plan cell, anchor_cat
    for zid, objs in by_zone.items():
        g = geoms.get(zid)
        if not objs or g is None:
            continue
        px0, px1 = g["ox"], g["ox"] + g["w"]
        pz0, pz1 = g["oz"], g["oz"] + g["d"]
        wall = resolve_wall(g.get("wall"), px0, px1, pz0, pz1, rb0)
        # A zone that touches no wall joins the nearest one.
        if wall is None:
            pcx, pcz = (px0 + px1) / 2, (pz0 + pz1) / 2
            wall = min(
                (("north", rb0[3] - pcz), ("south", pcz - rb0[2]),
                 ("east", rb0[1] - pcx), ("west", pcx - rb0[0])),
                key=lambda t: t[1],
            )[0]
        anchor_cat = g.get("anchor_category")
        work = list(objs)
        if wall is not None:
            work = face_zone_inward(
                work, wall_orientation=wall,
                wall_coord=wall_coord(wall, rb0), anchor_category=anchor_cat,
            )
        # A non-anchor behind the anchor is a lift depth error; slide it forward first.
        if wall is not None:
            _aid = _glue_anchor_id(work, wall, anchor_cat)
            if _aid:
                work, forward_shifts = _slide_behind_anchor_forward(
                    work, _aid, wall, exempt=forward_exempt)
                if forward_shifts:
                    logger.info("partition_resize: zone %s moved %d object(s) in front of the "
                                "anchor's back plane: %s",
                                zid[:8], len(forward_shifts), forward_shifts)
        start = _aabb_group(work)
        start = (start[0] - WALL_MARGIN, start[1] + WALL_MARGIN,
                 start[2] - WALL_MARGIN, start[3] + WALL_MARGIN)
        rect, sep = _fit_zone(work, start, margin=WALL_MARGIN,
                              max_dim_m=MAX_ROOM_M, anchor_cat=anchor_cat,
                              wall_orientation=wall,
                              lateral_pairs=lateral_pairs,
                              exempt=forward_exempt)
        wx, dz = rect[1] - rect[0], rect[3] - rect[2]
        Z[zid] = dict(wall=wall, objs=sep, wx=wx, dz=dz,
                      cell=(px0, px1, pz0, pz1), anchor_cat=anchor_cat)

    # Phase 3: each zone pair stays separated along the axis that separated their plan cells;
    # zones that still fit keep their plan position, outgrown ones push neighbours to ZONE_GAP_M.
    zids = list(Z)
    cons = {"x": [], "z": []}
    for a in range(len(zids)):
        for b in range(a + 1, len(zids)):
            ca, cb = Z[zids[a]]["cell"], Z[zids[b]]["cell"]
            sep_x = max(cb[0] - ca[1], ca[0] - cb[1])
            sep_z = max(cb[2] - ca[3], ca[2] - cb[3])
            axis = "x" if sep_x >= sep_z else "z"
            c0, c1 = (0, 1) if axis == "x" else (2, 3)
            ka = (ca[c0] + ca[c1], zids[a])
            kb = (cb[c0] + cb[c1], zids[b])
            first, second = (zids[a], zids[b]) if ka <= kb else (zids[b], zids[a])
            cons[axis].append((first, second))

    def _solve_axis(axis):
        lo = ox if axis == "x" else oz
        R0 = W0 if axis == "x" else D0
        size = {z: (Z[z]["wx"] if axis == "x" else Z[z]["dz"]) for z in zids}
        c0, c1 = (0, 1) if axis == "x" else (2, 3)
        lo_wall, hi_wall = ("west", "east") if axis == "x" else ("south", "north")
        order = sorted(zids, key=lambda z: (Z[z]["cell"][c0] + Z[z]["cell"][c1], z))
        preds = {z: [i for (i, j) in cons[axis] if j == z] for z in zids}
        dim = R0
        pos: dict[str, float] = {}
        for _ in range(_SOLVE_PASSES):
            pos = {}
            for z in order:
                w = Z[z]["wall"]
                if w == lo_wall:
                    t = lo + WALL_MARGIN
                elif w == hi_wall:
                    t = lo + dim - WALL_MARGIN - size[z]
                else:  # perpendicular wall: keep the plan position on this axis
                    t = (Z[z]["cell"][c0] + Z[z]["cell"][c1]) / 2 - size[z] / 2
                t = max(t, lo + WALL_MARGIN)
                for p in preds[z]:
                    t = max(t, pos[p] + size[p] + ZONE_GAP_M)
                pos[z] = t
            need = max(pos[z] + size[z] for z in zids) - lo + WALL_MARGIN
            nxt = max(R0, need)
            if nxt <= dim + 1e-9:
                break
            dim = nxt
        # Wall zones the constraints pushed off their wall target are not glued later.
        onwall = {}
        for z in zids:
            w = Z[z]["wall"]
            if w == lo_wall:
                onwall[z] = pos[z] <= lo + WALL_MARGIN + 1e-6
            elif w == hi_wall:
                onwall[z] = pos[z] >= lo + dim - WALL_MARGIN - size[z] - 1e-6
        return pos, dim, onwall

    pos_x, W, onwall_x = _solve_axis("x")
    pos_z, D, onwall_z = _solve_axis("z")
    onwall = {**onwall_x, **onwall_z}   # a zone's wall lives on exactly one axis
    squeezed = []
    if W > MAX_ROOM_M: squeezed.append(("W", round(W, 2))); W = MAX_ROOM_M
    if D > MAX_ROOM_M: squeezed.append(("D", round(D, 2))); D = MAX_ROOM_M
    if squeezed:
        logger.warning("partition_resize: room side capped at %.1f m (solved %s)",
                       MAX_ROOM_M, squeezed)
    rb = (ox, ox + W, oz, oz + D)

    # Phase 4: place each zone rigidly in its slot, glue its anchor, clamp into the room.
    overrides = {}
    locked = {}
    cx_room, cz_room = ox + W / 2, oz + D / 2
    for zid in zids:
        z = Z[zid]
        a = _aabb_group(z["objs"])
        objs = _shift_axis(z["objs"], 0, pos_x[zid] - a[0])
        objs = _shift_axis(objs, 2, pos_z[zid] - a[2])
        # glue_zone_to_wall ignores the coord when the wall is None.
        _wc = wall_coord(z["wall"], rb) if z["wall"] else 0.0
        glued, anchor_id = glue_zone_to_wall(
            objs, wall_orientation=z["wall"], wall_coord=_wc,
            margin=WALL_MARGIN, anchor_category=z["anchor_cat"],
        )
        if not onwall.get(zid, True):
            # Gluing a zone the solver pushed off its wall would re-create the overlap;
            # the glue call above is still needed for the anchor id.
            logger.info("partition_resize: zone %s keeps its solved position off the %s wall",
                        zid[:8], z["wall"])
            glued = objs
        # The glue can push the envelope past the room; shift it back in rigidly.
        ga = _aabb_group(glued)
        dx = max(0.0, rb[0] - ga[0]) - max(0.0, ga[1] - rb[1])
        dzc = max(0.0, rb[2] - ga[2]) - max(0.0, ga[3] - rb[3])
        if abs(dx) > 1e-9 or abs(dzc) > 1e-9:
            glued = _shift_axis(_shift_axis(glued, 0, dx), 2, dzc)
            logger.info("partition_resize: zone %s containment clamp shifted the glued zone %.2f m",
                        zid[:8], math.hypot(dx, dzc))
        for o in glued:
            overrides[o.instance_id] = o
        if anchor_id:
            locked[anchor_id] = z["wall"]
        z["placed"] = glued

    # Rebuild the layout: new room bbox, and zone polygons around the final footprints.
    seen = set(); new_objs = []
    for z in Z.values():
        for o in z.get("placed", z["objs"]):
            new_objs.append(o); seen.add(o.instance_id)
    for objs in by_zone.values():          # carry any zone-less / geom-less object
        for o in objs:
            if o.instance_id not in seen:
                new_objs.append(o); seen.add(o.instance_id)
    new_objs = [overrides.get(o.instance_id, o) for o in new_objs]

    tmpl = json.loads(json.dumps(template_dict))
    room = tmpl["zoning_plan"]["rooms"][0]
    yc = template_dict["zoning_plan"]["rooms"][0]["bbox"]["center"][1]
    ys = room["bbox"]["size"][1]
    room["bbox"] = {"center": [cx_room, yc, cz_room], "size": [W, ys, D],
                    "rotation_y": room["bbox"].get("rotation_y", 0.0)}
    by_new = {}
    for o in new_objs:
        by_new.setdefault(o.zone_id, []).append(o)
    for zj in room["functional_zones"]:
        objs = by_new.get(zj["zone_id"])
        if not objs:
            continue
        a = _aabb_group(objs)
        g = ZONE_GAP_M
        zj["polygon_2d"] = [[a[0] - g, a[2] - g], [a[1] + g, a[2] - g],
                            [a[1] + g, a[3] + g], [a[0] - g, a[3] + g]]

    lay = SceneLayout.model_validate(
        {**tmpl, "placed_objects": [json.loads(o.model_dump_json()) for o in new_objs]}
    )
    packed_tmpl = json.loads(lay.model_dump_json())
    all_packed = list(lay.placed_objects)
    logger.info("partition_resize: placed %d zones -> room %.2fx%.2fm (was %.2fx%.2f)",
                len(Z), W, D, W0, D0)
    return all_packed, locked, packed_tmpl
