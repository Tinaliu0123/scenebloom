"""The compiler: read the relation contract, then fit, refine and resolve."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from scenebloom.compilation.fitting.among_zones import partition_resize_place, zone_geoms
from scenebloom.compilation.fitting.feasibility import global_feasibility
from scenebloom.compilation.fitting.within_zone import LATERAL_OPERATORS
from scenebloom.compilation.lifting.source_contract import (
    SourceLayoutContract,
    extract_source_contract,
)
from scenebloom.compilation.refinement.apply import repair_cached_relations
from scenebloom.compilation.refinement.constraints import Constraint
from scenebloom.sampling.proposals.strategy import zone_short_id
from scenebloom.utils.schemas import PlacedObject, SceneLayout

logger = logging.getLogger(__name__)

#: Contract relations at or above this confidence are repaired in the refine step.
HARD_CONFIDENCE = 0.75

#: Contract reads per compile, the first included; an unavailable contract is read again.
CONTRACT_MAX_READS = 3
CONTRACT_RETRY_BACKOFF_S = (3.0, 8.0)


def compile_scene_from_pipeline(config, layout: SceneLayout) -> SceneLayout:
    """Compile the pipeline's lifted layout."""
    return compile_scene(
        layout,
        flux_images=_discover_flux_images(Path(config.output_dir), layout),
    )


def compile_scene(
    layout: SceneLayout,
    *,
    flux_images: dict[str, str] | None = None,
    contract_vlm_fn: Callable[[list[dict[str, Any]]], str] | None = None,
) -> SceneLayout:
    """Read the source contract of one layout and fit the layout to it."""
    relations = read_source_contract(layout, flux_images or {}, contract_vlm_fn).relations
    return fit_layout(layout, relations)


def read_source_contract(
    layout: SceneLayout,
    flux_images: dict[str, str],
    vlm_fn: Callable[[list[dict[str, Any]]], str] | None = None,
) -> SourceLayoutContract:
    """Extract the contract, reading it again while it is unavailable."""
    for attempt in range(1, CONTRACT_MAX_READS + 1):
        contract = extract_source_contract(layout, flux_images=flux_images, vlm_fn=vlm_fn)
        if contract.status != "unavailable":
            break
        if attempt < CONTRACT_MAX_READS:
            logger.warning("compiler: source contract unavailable (%s); re-reading %d/%d",
                           contract.reason, attempt + 1, CONTRACT_MAX_READS)
            time.sleep(CONTRACT_RETRY_BACKOFF_S[min(attempt - 1, len(CONTRACT_RETRY_BACKOFF_S) - 1)])
    logger.info("compiler: source contract %s with %d relations",
                contract.status, len(contract.relations))
    return contract


def fit_layout(layout: SceneLayout, relations: Sequence[Constraint] = ()) -> SceneLayout:
    """Fit, refine and resolve one lifted layout; ``relations`` name objects by instance id."""
    template = json.loads(layout.model_dump_json())
    template["placed_objects"] = []
    by_zone: dict[str, list[PlacedObject]] = {}
    for o in layout.placed_objects:
        by_zone.setdefault(o.zone_id, []).append(o)
    geoms = zone_geoms(template)

    # 1. Fit: de-collide each zone's objects, then pack the zones into the room.
    if any(objs and geoms.get(zid) for zid, objs in by_zone.items()):
        objects, locked, template = partition_resize_place(
            by_zone, geoms, template,
            forward_exempt=_forward_exempt_ids(by_zone, geoms, relations),
            lateral_pairs=_lateral_pairs(relations),
        )
    else:
        objects = [o for objs in by_zone.values() for o in objs]
        locked = {}

    # 2. Refine: repair the hard contract relations the fit left violated.
    hard = [c for c in relations if c.confidence >= HARD_CONFIDENCE]
    if hard:
        repaired = repair_cached_relations(
            {**template, "placed_objects": [json.loads(o.model_dump_json()) for o in objects]},
            hard,
        )
        objects = list(SceneLayout.model_validate(repaired).placed_objects)

    # 3. Resolve: separate what still overlaps and keep every object in the room.
    return global_feasibility(objects, locked, template)


def _side_of(subject: PlacedObject, target: PlacedObject) -> str:
    """Which side of ``target``, in its own yaw frame, ``subject``'s centre lies on."""
    yaw = float(target.pose.rotation_y or 0.0)
    fx, fz = math.sin(yaw), -math.cos(yaw)
    dx = subject.pose.center[0] - target.pose.center[0]
    dz = subject.pose.center[2] - target.pose.center[2]
    f = dx * fx + dz * fz
    l = dx * fz - dz * fx
    if abs(f) >= abs(l):
        return "front" if f >= 0 else "back"
    return "left" if l >= 0 else "right"


def _forward_exempt_ids(
    by_zone: dict[str, list[PlacedObject]], geoms: dict, relations: Sequence[Constraint]
) -> frozenset[str]:
    """``face`` subjects of an anchor faced from two or more sides; they may stay behind it."""
    anchor_cats = {g.get("anchor_category") for g in geoms.values()}
    objs = {o.instance_id: o for zone_objs in by_zone.values() for o in zone_objs}
    pairs = [(c.subject_id, c.target_id) for c in relations
             if c.operator == "face" and c.subject_id in objs and c.target_id in objs
             and objs[c.target_id].category in anchor_cats]
    sides: dict[str, set[str]] = {}
    for s, t in pairs:
        sides.setdefault(t, set()).add(_side_of(objs[s], objs[t]))
    return frozenset(s for s, t in pairs if len(sides[t]) >= 2)


def _lateral_pairs(relations: Sequence[Constraint]) -> frozenset[frozenset[str]]:
    """Object pairs the contract wants side by side; zone de-collision separates them sideways."""
    return frozenset(
        frozenset((c.subject_id, c.target_id)) for c in relations
        if c.operator in LATERAL_OPERATORS and c.subject_id and c.target_id
        and c.subject_id != c.target_id
    )


def _discover_flux_images(out_dir: Path, layout: SceneLayout) -> dict[str, str]:
    """zone_id -> the FLUX image named in the zone's proposal.json; zones without one are left out."""
    images: dict[str, str] = {}
    base = out_dir / "proposals" / layout.layout_id
    if not base.is_dir():
        return images
    zone_by_short = {zone_short_id(z.zone_id): z.zone_id
                     for room in layout.zoning_plan.rooms for z in room.functional_zones}
    for d in sorted(base.glob("zone_*")):
        zone_id = zone_by_short.get(d.name.split("zone_")[-1])
        record = d / "proposal.json"
        if zone_id is not None and record.is_file():
            images[zone_id] = str(d / json.loads(record.read_text(encoding="utf-8"))["image"])
    return images
