"""Wall-object placement: flush on the zone's wall, placed from the plan rather than lifted."""
# Room walls: north = +Z, south = -Z, east = +X, west = -X. ``center_y`` is the box base.

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

from scenebloom.utils.schemas import BBox3D, PlacedObject
from scenebloom.utils.assets import AssetDB
from scenebloom.utils.assets import room_allows

#: VLM call on a chat message list (system + user turn), returning the raw reply.
LLMMessagesFn = Callable[[list[dict[str, str]]], str]

logger = logging.getLogger(__name__)

# Direction from the room centre to each wall, in (x, z).
_WALL_DIR: dict[str, tuple[float, float]] = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}
# Yaw that makes an object on each wall face the room; at yaw 0 the front is (0, -1).
_INWARD_YAW: dict[str, float] = {
    "north": 0.0,
    "south": math.pi,
    "east": -math.pi / 2.0,
    "west": math.pi / 2.0,
}

# Target vertical centre per category, metres (Holodeck-style height priors).
_DEFAULT_HEIGHT_CENTERS: dict[str, float] = {
    "mirror": 1.55,
    "wall_art": 1.55,
    "wall_clock": 1.95,
}
_FALLBACK_HEIGHT_CENTER = 1.50


@dataclass(frozen=True)
class WallObjectConfig:
    """Placement knobs (all deterministic)."""

    margin_m: float = 0.05
    #: category -> target vertical centre, metres
    height_centers: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_HEIGHT_CENTERS)
    )
    fallback_height_center: float = _FALLBACK_HEIGHT_CENTER
    max_per_zone: int = 4
    #: seeded along-wall positions tried before giving up
    max_retries: int = 6
    collision_margin_m: float = 0.02
    check_furniture_collision: bool = True


@dataclass(frozen=True)
class WallPoseContext:
    """A single requested wall object expressed in a normalized wall frame."""

    room_type: str
    zone_id: str
    category: str
    wall_orientation: str
    wall_width_m: float
    wall_height_m: float
    asset_ids: tuple[str, ...]
    #: (width_m, height_m) per ``asset_ids`` entry; empty means not supplied.
    asset_dims: tuple[tuple[float, float], ...] = ()
    nearby_objects: tuple[str, ...] = ()
    #: Furniture against this wall as (short_id, category, u0, u1), u in [-1, 1].
    anchors: tuple[tuple[str, str, float, float], ...] = ()


@dataclass(frozen=True)
class WallPoseProposal:
    """Proposed wall-local centre. ``u`` and ``v`` are both in ``[-1, 1]``."""

    asset_id: str
    u: float = 0.0
    v: float = 0.0
    #: Short id of the anchor to hang above (the placer then sets u and height), or None.
    above: str | None = None


@dataclass(frozen=True)
class WallZoneContext:
    """What a planner needs to decide what belongs on one wall."""

    room_type: str
    zone_id: str
    zone_label: str
    wall_orientation: str
    wall_width_m: float
    wall_height_m: float
    #: ``(category, hint)`` pairs the planner may choose from, and nothing else.
    catalogue: tuple[tuple[str, str], ...]
    nearby_objects: tuple[str, ...] = ()
    #: What already hangs in this room, so each wall is not planned blind.
    room_state: tuple[str, ...] = ()


@dataclass(frozen=True)
class WallZonePlan:
    """Categories a planner wants on one wall.  Empty tuple = bare wall."""

    categories: tuple[str, ...] = ()


def _format_meshes(ctx: "WallPoseContext") -> str:
    """One line per mesh, with its size when ``asset_dims`` matches ``asset_ids``."""
    dims = ctx.asset_dims
    if len(dims) != len(ctx.asset_ids):
        return f"  {', '.join(ctx.asset_ids)}\n"
    width = max((len(a) for a in ctx.asset_ids), default=0)
    return "".join(
        f"  {aid:<{width}}  {w:.2f} m wide x {h:.2f} m high\n"
        for aid, (w, h) in zip(ctx.asset_ids, dims)
    )


#: Vertical gap between an anchor's top face and the wall object's base.
_ANCHOR_GAP_M = 0.15

#: Furniture stands against a wall when its footprint is within this distance of it, metres.
_AGAINST_WALL_GAP_M = 0.9

#: Max pieces of one category per room; categories not listed default to 1.
_ROOM_CATEGORY_CAP: dict[str, int] = {"wall_art": 2}
_ROOM_CATEGORY_CAP_DEFAULT = 1


def _room_category_cap(category: str) -> int:
    return _ROOM_CATEGORY_CAP.get(category, _ROOM_CATEGORY_CAP_DEFAULT)


def _room_state_lines(room_wall_log: list[tuple[str, str]]) -> tuple[str, ...]:
    """Compress the rolling room log into planner-facing lines."""
    if not room_wall_log:
        return ()
    by_wall: dict[str, list[str]] = {}
    tally: dict[str, int] = {}
    for orientation, cat in room_wall_log:
        by_wall.setdefault(orientation, []).append(cat)
        tally[cat] = tally.get(cat, 0) + 1
    lines = [f"{o} wall: {', '.join(cats)}" for o, cats in sorted(by_wall.items())]
    lines.append("room total: "
                 + ", ".join(
                     f"{c} x{n} (limit {_room_category_cap(c)})"
                     for c, n in sorted(tally.items())))
    return tuple(lines)


def _wall_anchor_candidates(
    placed: list[dict], zone_id: str, frame: "WallFrame",
) -> list[tuple[str, str, float, float, float]]:
    """Zone furniture against this wall as (short_id, category, u0, u1, top_y), u in [-1, 1]."""
    span = max(1e-6, frame.along_max - frame.along_min)
    out: list[tuple[str, str, float, float, float]] = []
    for o in placed:
        if o.get("zone_id") != zone_id:
            continue
        pose = o.get("pose") or {}
        c = pose.get("center") or (0.0, 0.0, 0.0)
        s = pose.get("size") or (0.0, 0.0, 0.0)
        if frame.along_idx == 0:
            along_c, along_half = float(c[0]), float(s[0]) / 2.0
            normal_c, normal_half = float(c[2]), float(s[2]) / 2.0
        else:
            along_c, along_half = float(c[2]), float(s[2]) / 2.0
            normal_c, normal_half = float(c[0]), float(s[0]) / 2.0
        gap_to_wall = abs(normal_c - frame.normal_coord) - normal_half
        if gap_to_wall > _AGAINST_WALL_GAP_M:
            continue
        u0 = 2.0 * (along_c - along_half - frame.along_min) / span - 1.0
        u1 = 2.0 * (along_c + along_half - frame.along_min) / span - 1.0
        if u1 < -1.0 or u0 > 1.0:
            continue
        sid = str(o.get("instance_id", "?"))[:6]
        out.append((sid, str(o.get("category", "?")),
                    max(-1.0, u0), min(1.0, u1),
                    float(c[1]) + float(s[1])))
    return out


def _format_room_state(room_state: tuple[str, ...]) -> str:
    """Render the rolling this-room summary the planner decides against."""
    if not room_state:
        return "Already hanging in this room : nothing yet\n"
    return ("Already hanging in this room :\n"
            + "".join(f"  {line}\n" for line in room_state))


def _format_anchors(anchors: tuple[tuple[str, str, float, float], ...]) -> str:
    """Render the against-this-wall furniture as anchor lines, or nothing."""
    if not anchors:
        return ""
    lines = ["Furniture against this wall (anchor candidates):\n"]
    for sid, cat, u0, u1 in anchors:
        lines.append(f"  id {sid}  {cat}  spans u in [{u0:+.2f}, {u1:+.2f}]\n")
    return "".join(lines)


def _size_hint(dims: list[tuple[float, float]]) -> str:
    """Compact size summary for one category's meshes: a point or a range."""
    if not dims:
        return ""
    ws = [w for w, _ in dims]
    hs = [h for _, h in dims]
    def _span(lo: float, hi: float) -> str:
        return f"{lo:.2f}" if abs(hi - lo) < 5e-3 else f"{lo:.2f}-{hi:.2f}"
    n = f" ({len(dims)} meshes)" if len(dims) > 1 else ""
    return (f"{_span(min(ws), max(ws))} m wide x "
            f"{_span(min(hs), max(hs))} m high{n}")


def _with_cap_wording(category: str, size_hint: str) -> str:
    """Append the per-room frequency to one catalogue hint line."""
    cap = _room_category_cap(category)
    freq = "one per room" if cap == 1 else f"up to {cap} per room"
    return f"{size_hint}; {freq}" if size_hint else freq


def _format_catalogue(catalogue: tuple[tuple[str, str], ...]) -> str:
    """Render the DB-derived catalogue as indented `name  hint` lines."""
    if not catalogue:
        return "  (none)\n"
    width = max(len(c) for c, _ in catalogue)
    return "".join(
        f"  {cat:<{width}}  {hint}\n" if hint else f"  {cat}\n"
        for cat, hint in catalogue
    )


class WallPoseProposer(Protocol):
    def propose(self, ctx: WallPoseContext) -> WallPoseProposal | None: ...


@dataclass(frozen=True)
class VLMWallPoseProposer:
    """One-shot VLM proposal of a wall-local (u, v) or an anchor; None on any failure."""

    llm_fn: LLMMessagesFn

    SYSTEM = (
        "You are placing one wall mounted decoration on one wall of one room.\n"
        "\n"
        "You are given the wall size and the object that will hang on it, with its "
        "measured size. Say roughly where on the wall it should sit.\n"
        "\n"
        "You never output world coordinates, yaw, size or scale. Position is "
        "expressed in wall local coordinates:\n"
        "  u  runs across the wall from left (-1) to right (+1)\n"
        "  v  runs up the wall from bottom (-1) to top (+1)\n"
        "u = 0, v = 0 is the exact centre of the wall. A deterministic placer "
        "converts your (u, v) into a world pose, clamps it inside the wall, and "
        "resolves any overlap with furniture, so a plausible answer is enough.\n"
        "\n"
        "If furniture stands against this wall you will be given each piece "
        "with an id and its span along the wall. Wall decor usually hangs ABOVE "
        "a piece of furniture -- a mirror above the washstand, a painting above "
        "the sofa. When that is the natural choice, answer with that anchor id "
        "and the placer will centre the object over it at the right height; "
        "answer null to place freely by (u, v) instead.\n"
        "\n"
        "Reply with one JSON object and nothing else:\n"
        '  {"asset_id": "<the id given>", "above": "<anchor id or null>", '
        '"u": <number>, "v": <number>}'
    )

    def build_user(self, ctx: WallPoseContext) -> str:
        mesh_line = "Mesh (already chosen) :\n" + _format_meshes(ctx)
        ask = "Choose a plausible spot on this wall."
        return (
            f"Room type      : {ctx.room_type}\n"
            f"Zone           : {ctx.zone_id}\n"
            f"Wall           : the {ctx.wall_orientation} facing wall, "
            f"{ctx.wall_width_m:.2f} m wide and {ctx.wall_height_m:.2f} m high\n"
            f"Object to hang : {ctx.category}\n"
            f"{mesh_line}"
            f"Furniture already standing in this zone : "
            f"{', '.join(ctx.nearby_objects) or 'none'}\n"
            f"{_format_anchors(ctx.anchors)}"
            "\n"
            f"{ask}"
        )

    def build_messages(self, ctx: WallPoseContext) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": self.build_user(ctx)},
        ]

    def propose(self, ctx: WallPoseContext) -> WallPoseProposal | None:
        try:
            raw = self.llm_fn(self.build_messages(ctx))
            lo, hi = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[lo : hi + 1] if lo >= 0 and hi >= lo else raw)
            asset_id = str(data.get("asset_id", ""))
            if asset_id not in ctx.asset_ids:
                return None
            above = data.get("above")
            above = str(above) if above not in (None, "", "null", "none") else None
            if above is not None and above not in {a[0] for a in ctx.anchors}:
                above = None
            u = max(-1.0, min(1.0, float(data.get("u", 0.0) or 0.0)))
            v = max(-1.0, min(1.0, float(data.get("v", 0.0) or 0.0)))
            return WallPoseProposal(asset_id=asset_id, u=u, v=v, above=above)
        except Exception:  # the placer falls back to seeded geometry
            return None


class WallCategoryPlanner(Protocol):
    """Decides WHICH wall objects a zone's wall gets, including none at all."""

    def plan_zone(self, ctx: WallZoneContext) -> WallZonePlan | None:
        """Return the plan, or ``None`` if the call failed / was unusable."""
        ...


@dataclass
class VLMWallCategoryPlanner:
    """Ask a VLM which catalogue categories belong on one wall; an empty wall is a valid answer.

    The catalogue sits in the system turn: in the user turn alone the model still names
    categories outside it. An out-of-catalogue name is logged and leaves the wall bare.
    """

    llm_fn: LLMMessagesFn
    max_per_wall: int = 2

    SYSTEM = (
        "You are an interior designer. For one wall of one room you decide which "
        "wall mounted decorations, if any, should hang on it.\n"
        "\n"
        "CATALOGUE. The list below is the whole of what exists; it is read from "
        "the asset library for THIS room type, so it is short and it changes "
        "from room to room:\n"
        "{catalogue}"
        "Never name anything that is not on that list. Whatever you have in mind "
        "that is missing from it -- a shelf, a sconce, a tapestry, a television, "
        "a hanging plant, a sign -- does not exist here, and naming it produces "
        "nothing at all.\n"
        "\n"
        "THE WHOLE ROOM IS THE UNIT OF TASTE, NOT THIS WALL. You will be told "
        "what already hangs on the other walls of this room, and every "
        "catalogue line says how many of that category ONE room tolerates -- "
        "most belong once, a few twice, never more -- while a category the "
        "room has used up is no longer offered at all. Once anything hangs in "
        "the room prefer a DIFFERENT category or a bare wall over repeating: "
        "a home hangs one of most things, and duplicates read as a showroom.\n"
        "\n"
        "AN EMPTY WALL IS A CORRECT ANSWER. Real rooms have plenty of bare walls. "
        "A wall that is narrow, already crowded with tall furniture, or serves "
        "only as circulation space should normally get nothing. Never add an item "
        "just to fill space.\n"
        "\n"
        "HOW TO DECIDE. Judge from the room type, what the zone is used for, how "
        "much bare wall is available, and which furniture already stands in the "
        "zone. List the items most worth hanging first.\n"
        "\n"
        "YOU CHOOSE CATEGORY NAMES ONLY. Which concrete mesh is used, where on "
        "the wall it hangs and how high it sits are all settled in a later step, "
        "so do not describe them.\n"
        "\n"
        "Reply with one JSON object and nothing else:\n"
        '  {"categories": ["<name>", ...], "reason": "<one sentence>"}\n'
        "Use an empty list when the wall should stay bare."
    )

    def build_user(self, ctx: WallZoneContext) -> str:
        return (
            f"Room type : {ctx.room_type}\n"
            f"Zone      : {ctx.zone_label}, one functional area of that room\n"
            f"Wall      : the {ctx.wall_orientation} facing wall of this zone, "
            f"{ctx.wall_width_m:.2f} m wide and {ctx.wall_height_m:.2f} m high\n"
            f"Furniture already standing in this zone : "
            f"{', '.join(ctx.nearby_objects) or 'none'}\n"
            f"Catalogue available for this wall :\n"
            f"{_format_catalogue(ctx.catalogue)}"
            f"{_format_room_state(ctx.room_state)}"
            "\n"
            f"Choose at most {self.max_per_wall} item(s) for this wall, or an "
            "empty list if it should stay bare."
        )

    def build_system(self, ctx: WallZoneContext) -> str:
        """System turn with this room's catalogue filled in."""
        # ``str.replace``, not ``str.format``: the template contains a literal JSON object.
        return self.SYSTEM.replace(
            "{catalogue}", _format_catalogue(ctx.catalogue))

    def build_messages(self, ctx: WallZoneContext) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.build_system(ctx)},
            {"role": "user", "content": self.build_user(ctx)},
        ]

    def plan_zone(self, ctx: WallZoneContext) -> WallZonePlan | None:
        allowed = {c for c, _ in ctx.catalogue}
        try:
            raw = self.llm_fn(self.build_messages(ctx))
            lo, hi = raw.find("{"), raw.rfind("}")
            data = json.loads(raw[lo : hi + 1] if lo >= 0 and hi >= lo else raw)
            cats = data.get("categories")
            if not isinstance(cats, list):
                return None
        except Exception:
            return None                      # call or parse failure, not an empty plan
        picked, bad = [], []
        for c in cats[: self.max_per_wall]:
            (picked if str(c) in allowed else bad).append(str(c))
        if bad:
            logger.warning("wall planner named %d out-of-catalogue category(ies) "
                           "for zone %s: %s", len(bad), ctx.zone_id, bad)
            return WallZonePlan()
        return WallZonePlan(categories=tuple(picked))


@dataclass(frozen=True)
class WallFrame:
    """SE(2) surface frame of one room wall (axis-aligned room)."""

    orientation: str
    along_idx: int  #: 0 => wall runs along X, 2 => along Z (world coord index)
    along_min: float
    along_max: float
    normal_coord: float  #: wall-plane coordinate on the normal axis
    inward_sign: float  #: +1/-1: direction from wall into the room
    yaw: float  #: inward-facing yaw for objects on this wall


def _seeded_unit(*parts: object) -> float:
    """Deterministic float in [0, 1) from a tuple of parts (no global RNG)."""
    h = hashlib.blake2b(
        "|".join(str(p) for p in parts).encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def wall_frame(
    room_center: tuple[float, float, float],
    room_size: tuple[float, float, float],
    orientation: str,
    *,
    along_lo: float | None = None,
    along_hi: float | None = None,
) -> WallFrame:
    """SE(2) frame of a room wall, optionally limited to the span ``[along_lo, along_hi]``."""
    cx, _, cz = room_center
    sx, _, sz = room_size
    xmin, xmax = cx - sx / 2, cx + sx / 2
    zmin, zmax = cz - sz / 2, cz + sz / 2
    dirx, dirz = _WALL_DIR[orientation]

    if dirz != 0.0:  # north/south: wall runs along X, normal is Z
        along_idx = 0
        room_lo, room_hi = xmin, xmax
        normal_coord = zmax if dirz > 0 else zmin
        inward_sign = -dirz
    else:  # east/west: wall runs along Z, normal is X
        along_idx = 2
        room_lo, room_hi = zmin, zmax
        normal_coord = xmax if dirx > 0 else xmin
        inward_sign = -dirx

    lo = room_lo if along_lo is None else max(room_lo, along_lo)
    hi = room_hi if along_hi is None else min(room_hi, along_hi)
    return WallFrame(
        orientation=orientation,
        along_idx=along_idx,
        along_min=lo,
        along_max=hi,
        normal_coord=normal_coord,
        inward_sign=inward_sign,
        yaw=_INWARD_YAW[orientation],
    )


def _pick_wall_variant(
    candidates: list, *, seed_key: str
):
    """Deterministic choice among one category's wall variants, seeded by ``seed_key``."""
    ordered = sorted(candidates, key=lambda a: a.asset_id)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def _pose_on_wall(
    frame: WallFrame,
    along_coord: float,
    asset_size: tuple[float, float, float],
    center_base_y: float,
) -> BBox3D:
    """Place an asset flush on the wall at ``along_coord`` and base height."""
    depth = asset_size[2]
    normal_center = frame.normal_coord + frame.inward_sign * depth / 2.0
    if frame.along_idx == 0:  # along X, normal Z
        center = (along_coord, center_base_y, normal_center)
    else:  # along Z, normal X
        center = (normal_center, center_base_y, along_coord)
    return BBox3D(center=center, size=asset_size, rotation_y=frame.yaw)


def _base_y(
    target_center: float, height: float, ceiling_h: float, margin: float
) -> float:
    """Convert a target vertical centre to a stored base-y, clamped in-room."""
    base = target_center - height / 2.0
    lo = margin
    hi = max(lo, ceiling_h - height - margin)
    return min(max(base, lo), hi)


def _zone_along_span(
    polygon_2d: list[list[float]], along_idx: int
) -> tuple[float, float] | None:
    """Extent of a zone polygon along the wall axis (world index 0 or 2)."""
    if not polygon_2d:
        return None
    comp = 0 if along_idx == 0 else 1  # polygon vertices are [x, z]
    vals = [v[comp] for v in polygon_2d]
    return min(vals), max(vals)


def _planner_wall_demand(
    planner: "WallCategoryPlanner",
    room_type: str,
    zone: dict,
    zone_id: str,
    orientation: str,
    db: AssetDB,
    ceiling_h: float,
    placed: list,
    frame: "WallFrame",
    room_state: tuple[str, ...] = (),
    room_tally: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Ask the planner what this wall should carry."""
    dims_by_cat: dict[str, list[tuple[float, float]]] = {}
    for rec in db.query_wall_mounted():
        if room_allows(rec.allowed_room_types, room_type):
            dims_by_cat.setdefault(rec.category, []).append(
                (float(rec.bbox_size[0]), float(rec.bbox_size[1])))
    # Categories already at their room cap are not offered.
    tally = room_tally or {}
    dims_by_cat = {
        cat: dims for cat, dims in dims_by_cat.items()
        if tally.get(cat, 0) < _room_category_cap(cat)
    }
    catalogue = tuple(
        (cat, _with_cap_wording(cat, _size_hint(dims)))
        for cat, dims in sorted(dims_by_cat.items())
    )
    if not catalogue:
        return ()
    plan = planner.plan_zone(WallZoneContext(
        room_type=room_type,
        zone_id=zone_id,
        zone_label=str(zone.get("label") or zone.get("zone_label") or "zone"),
        wall_orientation=orientation,
        wall_width_m=max(0.0, frame.along_max - frame.along_min),
        wall_height_m=ceiling_h,
        catalogue=catalogue,
        nearby_objects=tuple(o.get("category", "") for o in placed
                             if o.get("zone_id") == zone_id),
        room_state=room_state,
    ))
    return plan.categories if plan is not None else ()


def plan_wall_objects(
    layout: dict,
    db: AssetDB,
    proposer: WallPoseProposer,
    planner: "WallCategoryPlanner",
    cfg: WallObjectConfig | None = None,
) -> list[PlacedObject]:
    """New wall objects for a raw ``scene_layout.json`` dict, which is not mutated.

    Each wall-backed zone gets its requested wall categories plus the planner's picks.
    """
    cfg = cfg or WallObjectConfig()
    result: list[PlacedObject] = []

    wall_categories = {r.category for r in db.query_wall_mounted()}
    if not wall_categories:
        return result

    layout_id = layout.get("layout_id", "layout")
    zp = layout.get("zoning_plan", {})
    ceiling_h = float(zp.get("ceiling_h", 2.7))

    placed = layout.get("placed_objects", [])
    have: dict[tuple[str, str], int] = {}
    for o in placed:
        key = (o.get("zone_id", ""), o.get("category", ""))
        have[key] = have.get(key, 0) + 1

    for room in zp.get("rooms", []):
        room_type = room.get("room_type", "")
        bbox = room.get("bbox", {})
        room_center = tuple(bbox.get("center", (0.0, 0.0, 0.0)))
        room_size = tuple(bbox.get("size", (0.0, 0.0, 0.0)))
        # Occupied wall rectangles per wall plane, shared by zones on the same wall.
        taken_by_wall: dict[tuple[int, float], list] = {}
        # Per room: what hangs where, which meshes are used, one height per category.
        room_wall_log: list[tuple[str, str]] = []   # (orientation, category)
        used_asset_ids: set[str] = set()
        room_cat_base_y: dict[str, float] = {}

        for zone in room.get("functional_zones", []):
            zone_id = zone.get("zone_id", "")
            declared_orientation = zone.get("wall_orientation")
            if declared_orientation not in _WALL_DIR:
                continue  # interior zone: no room wall to hang on
            # Hang where the anchor furniture actually stands.
            orientation = _anchor_furniture_wall(
                zone, zone_id, placed, room_center, room_size,
                declared_orientation,
            )
            if orientation != declared_orientation:
                logger.info(
                    "wall placer: zone %s hangs on the %s wall (anchor %s stands "
                    "there), not the declared %s",
                    zone_id[:8], orientation, zone.get("anchor_category"),
                    declared_orientation,
                )
            counts = dict(zone.get("category_counts") or {})
            # Planner picks join ``counts``; an explicit zoner request wins.
            room_tally: dict[str, int] = {}
            for _o, _c in room_wall_log:
                room_tally[_c] = room_tally.get(_c, 0) + 1
            demand = _planner_wall_demand(
                planner, room_type, zone, zone_id, orientation,
                db, ceiling_h, placed,
                _wall_frame_for_zone(room_center, room_size, orientation, zone),
                room_state=_room_state_lines(room_wall_log),
                room_tally=room_tally,
            )
            for cat in demand:
                counts.setdefault(cat, 1)
            n_in_zone = 0

            for category, want in counts.items():
                if category not in wall_categories:
                    continue
                gap = int(want) - have.get((zone_id, category), 0)
                for k in range(max(0, gap)):
                    if sum(1 for _o, _c in room_wall_log
                           if _c == category) >= _room_category_cap(category):
                        # Also enforced here: zoner counts bypass the planner's menu.
                        continue
                    if n_in_zone >= cfg.max_per_zone:
                        continue
                    cands = [
                        r
                        for r in db.query_by_category(category)
                        if r.mount_type.value == "wall"
                        and room_allows(r.allowed_room_types, room_type)
                    ]
                    if not cands:
                        continue
                    # A mesh hangs at most once per room.
                    cands = [a for a in cands if a.asset_id not in used_asset_ids]
                    if not cands:
                        continue
                    asset = _pick_wall_variant(
                        cands, seed_key=f"{layout_id}|{zone_id}|{category}|{k}"
                    )
                    frame = _wall_frame_for_zone(
                        room_center, room_size, orientation, zone
                    )
                    taken = taken_by_wall.setdefault(
                        (frame.along_idx, round(frame.normal_coord, 3)), []
                    )
                    anchors = _wall_anchor_candidates(
                        placed, zone_id, frame)
                    proposal = proposer.propose(
                        WallPoseContext(
                            room_type=room_type,
                            zone_id=zone_id,
                            category=category,
                            wall_orientation=orientation,
                            wall_width_m=max(0.0, frame.along_max - frame.along_min),
                            wall_height_m=ceiling_h,
                            # Only the seeded pick is offered; a VLM choice collapses onto one mesh.
                            asset_ids=(asset.asset_id,),
                            asset_dims=(
                                (float(asset.bbox_size[0]), float(asset.bbox_size[1])),
                            ),
                            nearby_objects=tuple(
                                o.get("category", "") for o in placed
                                if o.get("zone_id") == zone_id
                            ),
                            anchors=tuple(
                                (sid, cat, u0, u1)
                                for sid, cat, u0, u1, _top in anchors
                            ),
                        )
                    )
                    # Above an anchor: u = anchor centre, base = anchor top + gap;
                    # a height already set for this category in the room wins.
                    along_override = None
                    base_y_override = room_cat_base_y.get(category)
                    if proposal is not None and proposal.above is not None:
                        hit = next((a for a in anchors if a[0] == proposal.above), None)
                        if hit is not None:
                            _sid, _cat, u0, u1, top = hit
                            mid_u = (u0 + u1) / 2.0
                            along_override = (
                                frame.along_min
                                + (mid_u + 1.0) / 2.0
                                * (frame.along_max - frame.along_min)
                            )
                            if base_y_override is None:
                                base_y_override = min(
                                    top + _ANCHOR_GAP_M,
                                    max(cfg.margin_m,
                                        ceiling_h - asset.bbox_size[1] - cfg.margin_m),
                                )
                    obj = _try_place(
                        frame, asset, layout_id, zone_id, category, k,
                        ceiling_h, cfg, taken, placed, proposal,
                        along_override=along_override,
                        base_y_override=base_y_override,
                    )
                    if obj is None:
                        continue
                    result.append(obj)
                    n_in_zone += 1
                    used_asset_ids.add(obj.asset_id)
                    room_wall_log.append((orientation, category))
                    room_cat_base_y.setdefault(
                        category, float(obj.pose.center[1]))
    return result


def _wall_frame_for_zone(room_center, room_size, orientation, zone) -> WallFrame:
    frame0 = wall_frame(room_center, room_size, orientation)
    span = _zone_along_span(zone.get("polygon_2d") or [], frame0.along_idx)
    if span is None:
        return frame0
    return wall_frame(
        room_center, room_size, orientation, along_lo=span[0], along_hi=span[1]
    )


def _gap_to_wall(pose: dict, room_center, room_size, orientation: str) -> float:
    """Distance from a footprint's near face to the wall plane, metres (negative = through it)."""
    frame = wall_frame(room_center, room_size, orientation)
    c = pose.get("center") or (0.0, 0.0, 0.0)
    s = pose.get("size") or (0.0, 0.0, 0.0)
    normal_idx = 2 if frame.along_idx == 0 else 0
    return abs(float(c[normal_idx]) - frame.normal_coord) - float(s[normal_idx]) / 2.0


def _anchor_furniture_wall(
    zone: dict,
    zone_id: str,
    placed: list[dict],
    room_center,
    room_size,
    declared: str,
) -> str:
    """The declared wall, unless the zone's anchor stands against another wall but not this one."""
    anchor_cat = zone.get("anchor_category")
    if not anchor_cat:
        return declared
    anchors = [
        o for o in placed
        if o.get("zone_id") == zone_id and o.get("category") == anchor_cat
    ]
    if not anchors:
        return declared

    def _footprint_area(o: dict) -> float:
        s = (o.get("pose") or {}).get("size") or (0.0, 0.0, 0.0)
        return float(s[0]) * float(s[2])

    anchor = max(anchors, key=_footprint_area)
    pose = anchor.get("pose") or {}
    gaps = {w: _gap_to_wall(pose, room_center, room_size, w) for w in _WALL_DIR}
    if gaps[declared] <= _AGAINST_WALL_GAP_M:
        return declared
    against = [w for w in _WALL_DIR if gaps[w] <= _AGAINST_WALL_GAP_M]
    if not against:
        return declared
    return min(against, key=lambda w: gaps[w])


def _try_place(
    frame: WallFrame,
    asset,
    layout_id: str,
    zone_id: str,
    category: str,
    k: int,
    ceiling_h: float,
    cfg: WallObjectConfig,
    taken: list[list[tuple[float, float]]],
    existing: list[dict],
    proposal: WallPoseProposal | None = None,
    along_override: float | None = None,
    base_y_override: float | None = None,
) -> PlacedObject | None:
    width = asset.bbox_size[0]  # object faces inward: width lies along the wall
    half = width / 2.0
    lo = frame.along_min + cfg.margin_m + half
    hi = frame.along_max - cfg.margin_m - half
    if hi < lo:
        return None

    default_base_y = _base_y(
        cfg.height_centers.get(category, cfg.fallback_height_center),
        asset.bbox_size[1],
        ceiling_h,
        cfg.margin_m,
    )
    for attempt in range(cfg.max_retries):
        if attempt == 0 and along_override is not None:
            # Above an anchor: the placer sets u and the height.
            along = min(max(along_override, lo), hi)
            center_base_y = (base_y_override if base_y_override is not None
                             else default_base_y)
        elif attempt == 0 and proposal is not None:
            along = lo + ((proposal.u + 1.0) / 2.0) * (hi - lo)
            base_lo = cfg.margin_m
            base_hi = max(base_lo, ceiling_h - asset.bbox_size[1] - cfg.margin_m)
            center_base_y = base_lo + ((proposal.v + 1.0) / 2.0) * (base_hi - base_lo)
            if base_y_override is not None:
                # The room's height for this category wins over the VLM's v.
                center_base_y = base_y_override
        elif attempt == 0:
            along = (lo + hi) / 2.0
            center_base_y = (base_y_override if base_y_override is not None
                             else default_base_y)
        else:
            frac = _seeded_unit(layout_id, zone_id, category, k, attempt)
            along = lo + frac * (hi - lo)
            center_base_y = (base_y_override if base_y_override is not None
                             else default_base_y)
        pose = _pose_on_wall(frame, along, asset.bbox_size, center_base_y)
        rect = _wall_rect(along, center_base_y, asset.bbox_size, cfg.collision_margin_m)
        if any(_rect_overlap(rect, other) for other in taken):
            continue
        if cfg.check_furniture_collision and any(
            _boxes_intersect(pose, o.get("pose") or {}) for o in existing
        ):
            continue
        taken.append(rect)
        return PlacedObject(
                instance_id=_wall_instance_id(layout_id, zone_id, category, k),
                asset_id=asset.asset_id,
                category=asset.category,
                pose=pose,
                zone_id=zone_id,
                placement_method="wall_proposer_vlm" if proposal is not None else "wall_proposer",
            )
    return None


def _wall_rect(
    along: float,
    base_y: float,
    size: tuple[float, float, float],
    margin: float,
) -> list[tuple[float, float]]:
    """Wall-plane (along, y) corners of the asset's rectangle, grown by ``margin``."""
    from scenebloom.compilation.fitting.obb import rect_corners

    return rect_corners(along, base_y + size[1] / 2.0, size[0] + 2 * margin,
                        size[1] + 2 * margin, 0.0)


def _rect_overlap(a, b) -> bool:
    from scenebloom.compilation.fitting.obb import obb_overlap

    return obb_overlap(a, b)


def _pose_aabb(pose) -> tuple[float, float, float, float, float, float]:
    """World AABB for a yawed base-centred BBox3D or its JSON representation."""
    if isinstance(pose, dict):
        center = pose.get("center", (0.0, 0.0, 0.0))
        size = pose.get("size", (0.0, 0.0, 0.0))
        yaw = float(pose.get("rotation_y", 0.0))
    else:
        center, size, yaw = pose.center, pose.size, pose.rotation_y
    cx, by, cz = map(float, center)
    sx, sy, sz = map(float, size)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    wx, wz = c * sx + s * sz, s * sx + c * sz
    return (cx - wx / 2, cx + wx / 2, by, by + sy, cz - wz / 2, cz + wz / 2)


def _boxes_intersect(candidate: BBox3D, existing_pose: dict) -> bool:
    """Strict-volume AABB broad phase; touching faces are allowed."""
    a = _pose_aabb(candidate)
    b = _pose_aabb(existing_pose)
    eps = 1e-6
    return (
        a[0] < b[1] - eps and b[0] < a[1] - eps
        and a[2] < b[3] - eps and b[2] < a[3] - eps
        and a[4] < b[5] - eps and b[4] < a[5] - eps
    )


def _wall_instance_id(layout_id: str, zone_id: str, category: str, k: int) -> str:
    digest = hashlib.blake2b(
        f"{layout_id}|{zone_id}|{category}|{k}".encode(), digest_size=4
    ).hexdigest()
    return f"wall_{digest}"
