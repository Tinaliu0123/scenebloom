"""SourceLayoutContract: the qualitative relations the source zone image shows."""
# The VLM names relations only between ids of the zone's object JSON, so it cannot invent
# objects; a VLM error gives status "unavailable", never a silent empty contract.

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scenebloom.compilation.refinement.constraints import Constraint, parse_constraint_response
from scenebloom.compilation.refinement.zone_view import build_zone_view
from scenebloom.utils.assets import AssetMetadataDB
from scenebloom.utils.schemas import SceneLayout
from scenebloom.utils.vlm_openrouter import call_openrouter, image_block

logger = logging.getLogger(__name__)

CONTRACT_SYSTEM_PROMPT = """You are reading a diffusion-generated photograph of one
zone of a room, and writing down the spatial relations it shows.

Inputs:
1. The ORIGINAL generated photograph of this zone. This is the ONLY evidence
   for what the arrangement should look like; read the relations off it.
2. Structured JSON listing the objects reconstructed from that photograph,
   with the ids you must use.

Your task: list the qualitative spatial relations that the PHOTOGRAPH shows
should hold between these objects.

You are NOT being asked whether the current 3D reconstruction satisfies them.
Geometry re-checks every relation and decides all repairs later; your job is
only to say what the source image shows.

<example>
What each operator means, with EXAMPLES of the arrangement it describes. These
are illustrations of the operator, not a checklist and not a description of
this particular photograph: report a relation only if THIS photograph actually
shows it.
* face: seating faces its work or dining surface (chair/dining_chair/armchair
  -> desk, dining_table, coffee_table); seating faces the screen
  (sofa/armchair -> tv). Every chair drawn around a table faces that table.
* align_with: a piece squared up with a bigger neighbour (nightstand -> bed,
  chair -> table edge, side_table -> sofa).
* beside: two pieces standing side by side along the same wall or edge
  (cabinet -> bookshelf, nightstand -> bed, tv_stand -> bookshelf). The
  subject stands at the target's SIDE, not in front of or behind it.
* move_near: a companion piece kept within reach (table_lamp -> sofa/bed,
  nightstand -> bed, stool -> counter).
* back_to (WALL target only): a piece whose BACK stands flat against the wall
  behind the arrangement -- bed headboard, sofa back, wardrobe / bookshelf /
  cabinet / tv_stand / desk against the wall, bathroom_washstand, toilet, the
  long side of a bathtub. There is one wall in question: the far wall the
  camera faces. Not for a piece standing free of it, and not when something
  else stands between the piece and the wall.
</example>

Rules:
* Report only relations clearly visible in the photograph.
* confidence is YOUR calibrated belief that the photograph shows this
  relation: 0.9-1.0 unmistakable, 0.7-0.85 clear, 0.4-0.6 plausible but
  partly occluded or ambiguous. Do not copy the number from this text.
* Do not output world coordinates or numeric yaw values.
* Do not invent object ids; use only ids present in the JSON.
* Use only these operators: face, align_with, move_near, beside, back_to.
* "face" only for subjects whose "front_is_ambiguous" is false; the target may be
  any object in the JSON.
* Return at most 10 relations.
* Do NOT set "subject_interface". It is derived from the operator; writing
  "front" on a distance relation like move_near makes the relation invalid.
* Report a relation whether or not it currently holds: an arrangement the
  photograph clearly shows (four chairs seated around a table, a nightstand
  squared up with its bed) is a relation to record even when nothing looks
  wrong.
* back_to takes a WALL target, never an object: {"type": "wall"}.
* Return JSON only, one entry per relation:
  {"zone_assessment": {"function": "..."},
    "constraints": [{"subject_id": "...",
      "operator": "<one of the operators above>",
      "target": {"type": "object", "id": "..."},
      "confidence": <your calibrated value>, "reason": "..."}]}

"""

#: Per-zone user turn; the constant instructions sit in the system turn so they can be cached.
_CONTRACT_USER = """Objects in this zone:
{scene_json}
"""


@dataclass(frozen=True)
class SourceLayoutContract:
    """The relations read for one compile, naming objects by instance id."""

    status: str                   # "extracted" | "unavailable"
    reason: str = ""
    relations: tuple[Constraint, ...] = ()


def extract_source_contract(
    layout: SceneLayout,
    *,
    flux_images: dict[str, str],
    vlm_fn: Callable[[list[dict[str, Any]]], str] | None = None,
) -> SourceLayoutContract:
    """One VLM read per zone with at least two objects."""
    meta_db = AssetMetadataDB.load()
    fn = vlm_fn or call_openrouter
    relations: list[Constraint] = []
    for room in layout.zoning_plan.rooms:
        for zone in room.functional_zones:
            zone_objs = [o for o in layout.placed_objects
                         if o.zone_id == zone.zone_id]
            if len(zone_objs) < 2:
                continue
            view = build_zone_view(room, zone, zone_objs, meta_db)
            prompt = _CONTRACT_USER.format(
                scene_json=json.dumps(view.scene_json, indent=2),
            )
            # Only the source image is sent: the contract describes it, not the lifted boxes.
            content: list[dict[str, Any]] = []
            flux = flux_images.get(zone.zone_id)
            if flux and Path(flux).is_file():
                block = image_block(Path(flux))
                if block is not None:
                    content.append(block)
            content.append({"type": "text", "text": prompt})
            try:
                reply = fn([{"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
                            {"role": "user", "content": content}])
            except Exception as exc:  # the VLM call failed: an unavailable contract, not an empty one
                return SourceLayoutContract(
                    "unavailable",
                    reason=f"VLM call failed on zone {zone.zone_id}: {exc}",
                    relations=tuple(relations),
                )
            result = parse_constraint_response(
                reply, view.scene_json,
                # wall targets are image-relative; wall_orientation maps them to a room side
                zone_wall_orientation=zone.wall_orientation,
            )
            real = view.instance_id_by_display_id
            for c in result.constraints:
                subj = real.get(c.subject_id)
                targ = real.get(c.target_id) if c.target_id else None
                if subj is None or (c.target_id and targ is None):
                    continue
                relations.append(dataclasses.replace(c, subject_id=subj, target_id=targ))

    return SourceLayoutContract("extracted", relations=tuple(relations))
