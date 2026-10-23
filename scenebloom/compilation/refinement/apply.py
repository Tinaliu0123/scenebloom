"""Relational refinement: repair the contract relations the layout violates."""

from __future__ import annotations

from typing import Any

from scenebloom.utils.schemas import SceneLayout
from scenebloom.compilation.refinement.zone_view import (
    pick_anchor_id,
    room_bounds_xz,
    zone_bounds_xz,
)
from scenebloom.compilation.refinement.repair import repair_verdicts
from scenebloom.compilation.refinement.verify import verify_constraints


def _serialise(obj) -> dict:
    return {
        "instance_id": obj.instance_id,
        "asset_id": obj.asset_id,
        "category": obj.category,
        "pose": {
            "center": list(obj.pose.center),
            "size": list(obj.pose.size),
            "rotation_y": obj.pose.rotation_y,
        },
        "zone_id": obj.zone_id,
        "placement_method": obj.placement_method,
    }


def repair_cached_relations(layout_json: dict, constraints: list[Any]) -> dict:
    """Repair violated relations zone by zone; returns a new layout dict.

    ``constraints`` name objects by instance id.
    """
    sl = SceneLayout.model_validate(layout_json)
    working = {o.instance_id: o for o in sl.placed_objects}

    zone_of = {o.instance_id: o.zone_id for o in sl.placed_objects}
    rooms_by_zone: dict[str, Any] = {}
    zones_by_id: dict[str, Any] = {}
    for room in sl.zoning_plan.rooms:
        for zone in room.functional_zones:
            zones_by_id[zone.zone_id] = zone
            rooms_by_zone[zone.zone_id] = room

    by_zone: dict[str, list[Any]] = {}
    for c in constraints:
        zone_id = zone_of.get(c.subject_id)
        if zone_id is not None:
            by_zone.setdefault(zone_id, []).append(c)

    for zone_id, zone_constraints in by_zone.items():
        zone = zones_by_id.get(zone_id)
        room = rooms_by_zone.get(zone_id)
        if zone is None or room is None:
            continue
        # Wall claims run after the object relations. A wall slide moves the reference of
        # other relations, so rounds repeat (at most 3) while a wall claim applies.
        object_cs = [c for c in zone_constraints
                     if not (c.operator == "back_to" and c.target_type == "wall")]
        wall_cs = [c for c in zone_constraints if c not in object_cs]
        rounds = [(object_cs, wall_cs)] if wall_cs else [(object_cs, ())]
        seen_rounds = 0
        while rounds:
            objs_cs, walls_cs = rounds.pop(0)
            seen_rounds += 1
            wall_applied = False
            for sub_pass, is_wall in ((objs_cs, False), (walls_cs, True)):
                if not sub_pass:
                    continue
                verdicts = verify_constraints(
                    sub_pass, working, room_bounds=room_bounds_xz(room),
                )
                context = [o for o in working.values()
                           if zone_of.get(o.instance_id) != zone_id]
                # The zone anchor owns the zone's facing and wall contact, so it is immovable.
                zone_objs = [o for o in working.values()
                             if zone_of.get(o.instance_id) == zone_id]
                anchor_inst = pick_anchor_id(zone, zone_objs)
                outcomes = repair_verdicts(
                    verdicts, working,
                    zone_bounds=zone_bounds_xz(zone), room_bounds=room_bounds_xz(room),
                    context=context,
                    immovable={anchor_inst} if anchor_inst else set(),
                )
                applied = [oc for oc in outcomes if oc.applied]
                for oc in applied:
                    working[oc.new_object.instance_id] = oc.new_object
                if is_wall and applied:
                    wall_applied = True
            if wall_applied and seen_rounds < 3:
                rounds.append((objs_cs, walls_cs))

    new_layout = dict(layout_json)
    new_layout["placed_objects"] = [_serialise(working[o.instance_id])
                                    for o in sl.placed_objects]
    return new_layout
