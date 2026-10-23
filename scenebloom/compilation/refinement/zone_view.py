"""The per-zone object JSON the contract VLM reads, and the zone and room bounds repair uses."""
# orientation_glyph: arrow = unique front (C1), axis = front mod 180 deg (C2),
# cross = front mod 90 deg (C4), circle = no front.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scenebloom.utils.schemas import (
    FunctionalZoneSpec,
    PlacedObject,
    RoomSpec,
)
from scenebloom.utils.geometry import facing_dir
from scenebloom.utils.assets import (
    AssetMetadataDB,
    AssetOrientationMeta,
)


@dataclass(frozen=True)
class ZoneView:
    """One zone as the VLM sees it; short display ids map back to instance ids."""

    scene_json: dict[str, Any]
    instance_id_by_display_id: dict[str, str]


def zone_bounds_xz(zone: FunctionalZoneSpec) -> tuple[float, float, float, float]:
    """``(xmin, xmax, zmin, zmax)`` of the zone polygon."""
    xs = [p[0] for p in zone.polygon_2d]
    zs = [p[1] for p in zone.polygon_2d]
    return (min(xs), max(xs), min(zs), max(zs))


def room_bounds_xz(room: RoomSpec) -> tuple[float, float, float, float]:
    cx, _, cz = room.bbox.center
    sx, _, sz = room.bbox.size
    return (cx - sx / 2.0, cx + sx / 2.0, cz - sz / 2.0, cz + sz / 2.0)


def glyph_for(meta: AssetOrientationMeta) -> str:
    """Orientation glyph of an asset; no structural axis means ``"circle"``."""
    if not meta.has_structural_axis:
        return "circle"
    return {"C1": "arrow", "C2": "axis", "C4": "cross"}.get(
        meta.rotation_symmetry, "circle"
    )


def _display_ids(zone_objects: list[PlacedObject]) -> dict[str, str]:
    """``instance_id -> "o1"…"oN"`` in zone order; short ordinals instead of uuids."""
    return {o.instance_id: f"o{i + 1}" for i, o in enumerate(zone_objects)}


def _object_entry(
    obj: PlacedObject,
    meta_db: AssetMetadataDB,
    *,
    is_anchor: bool,
    display_id: str,
) -> dict[str, Any]:
    meta = meta_db.get(obj.category)
    cx, _, cz = obj.pose.center
    sx, _, sz = obj.pose.size
    glyph = glyph_for(meta)
    front = list(facing_dir(obj.pose.rotation_y)) if glyph != "circle" else None
    return {
        "id": display_id,
        "instance_id": obj.instance_id,
        "label": obj.category,
        # C2/C4: the reported front is one of several equivalent ones.
        "front_is_ambiguous": glyph in ("axis", "cross"),
        "orientation_glyph": glyph,
        "center_xz": [cx, cz],
        "size_xz": [sx, sz],
        "yaw_deg": math.degrees(obj.pose.rotation_y),
        "semantic_front_world": front,
        "rotation_symmetry": meta.rotation_symmetry,
        "axis_alignment_prior": meta.axis_alignment_prior,
        # The zone anchor owns the zone's wall contact; repairs must not move it.
        "movable": not is_anchor,
    }


def pick_anchor_id(
    zone: FunctionalZoneSpec, zone_objects: list[PlacedObject]
) -> str | None:
    if zone.anchor_category is None:
        return None
    for obj in zone_objects:
        if obj.category == zone.anchor_category:
            return obj.instance_id
    return None


def build_zone_view(
    room: RoomSpec,
    zone: FunctionalZoneSpec,
    objects: list[PlacedObject],
    meta_db: AssetMetadataDB,
) -> ZoneView:
    """The VLM payload for ``zone``; only the objects in ``zone`` are listed."""
    zone_objects = [o for o in objects if o.zone_id == zone.zone_id]
    anchor_inst = pick_anchor_id(zone, zone_objects)
    disp = _display_ids(zone_objects)
    xmin, xmax, zmin, zmax = zone_bounds_xz(zone)
    rxmin, rxmax, rzmin, rzmax = room_bounds_xz(room)

    scene_json: dict[str, Any] = {
        "zone": {
            "id": zone.zone_id,
            "function": zone.label,
            "anchor_id": disp.get(anchor_inst) if anchor_inst else None,
            "bounds_xz": [xmin, xmax, zmin, zmax],
        },
        "room": {"type": room.room_type, "bounds_xz": [rxmin, rxmax, rzmin, rzmax]},
        "objects": [
            _object_entry(
                o,
                meta_db,
                is_anchor=(o.instance_id == anchor_inst),
                display_id=disp[o.instance_id],
            )
            for o in zone_objects
        ],
        "geometry": {
            "walls": [[rxmin, rxmax, rzmin, rzmax]],
            "doors": [],
            "free_space_polygons": [],
        },
    }
    return ZoneView(
        scene_json=scene_json,
        instance_id_by_display_id={v: k for k, v in disp.items()},
    )
