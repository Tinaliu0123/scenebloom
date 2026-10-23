"""Validation of the object relations (functional constraints) a VLM proposes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from scenebloom.utils.vlm_openrouter import extract_first_json_object

# Every operator here has a geometric predicate in verify and a repair in repair.
SUPPORTED_OPERATORS = (
    "face",
    "align_with",
    "move_near",
    "beside",
    "back_to",
)

_KEPT_STATUSES = ("violated", "slightly_violated", "satisfied")
_VALID_SCOPES = ("yaw_only", "yaw_and_translation", "no_reliable_repair")

# What each subject interface needs from the asset geometry.
_INTERFACE_REQUIREMENT = {
    "front": "unique_front",
    "back": "unique_front",
    "interaction_side": "unique_front",
    "structural_long_axis": "axis",
    "body": "none",
    "position": "none",
}

_DEFAULT_INTERFACE = {
    "face": "front",
    "back_to": "back",
    "align_with": "structural_long_axis",
    "move_near": "body",
    "beside": "body",
}

#: Scope when the reply omits ``repair_scope``: only positional operators may translate.
_DEFAULT_SCOPE = {
    "face": "yaw_only",
    "back_to": "yaw_only",
    "align_with": "yaw_only",
    "move_near": "yaw_and_translation",
    "beside": "yaw_and_translation",
}

DEFAULT_MIN_CONFIDENCE = 0.6
MAX_RELATIONS = 10

#: A wall target names no side: it is the zone's own wall (``wall_orientation``).
_CARDINALS = ("north", "east", "south", "west")
_WALL_SCOPE = "yaw_and_translation"


def _wall_interface(has_unique_front: bool, has_axis: bool) -> str:
    """Which part of the subject a wall claim is about: back, long side, or body."""
    if has_unique_front:
        return "back"
    if has_axis:
        return "structural_long_axis"
    return "body"


@dataclass(frozen=True)
class Constraint:
    """One declarative functional claim about the zone."""

    subject_id: str
    operator: str
    confidence: float
    repair_scope: str
    subject_interface: str = "front"
    target_id: str | None = None  # None for non-object targets (e.g. zone_axis)
    target_type: str = "object"
    target_axis: str | None = None
    #: The object target has a unique front (a desk has one, a dining table does not).
    target_has_front: bool = False
    # ``target_type == "wall"``: the room cardinal the subject's rear should touch.
    target_wall: str | None = None


@dataclass(frozen=True)
class ConstraintResult:
    """Constraints that survived validation."""

    constraints: list[Constraint] = field(default_factory=list)


def default_interface_for(operator: str) -> str:
    """Interface implied by ``operator`` when the reply omits one."""
    return _DEFAULT_INTERFACE.get(operator, "body")


def interface_requirement(interface: str) -> str:
    """What ``interface`` needs; an unknown interface conservatively needs a unique front."""
    return _INTERFACE_REQUIREMENT.get(interface, "unique_front")


def _geometry_by_id(scene_json: dict[str, Any]) -> dict[str, tuple[bool, bool]]:
    """``id -> (has_unique_front, has_axis)``; C2/C4 glyphs have an axis but no unique front."""
    out: dict[str, tuple[bool, bool]] = {}
    for obj in scene_json.get("objects", []):
        glyph = str(obj.get("orientation_glyph", "circle"))
        if glyph == "circle" or obj.get("semantic_front_world") is None:
            out[str(obj.get("id"))] = (False, False)
            continue
        unique_front = glyph == "arrow" and not obj.get("front_is_ambiguous")
        out[str(obj.get("id"))] = (unique_front, True)
    return out


def _parse_one(
    raw: dict[str, Any],
    *,
    known_ids: set[str],
    geometry: dict[str, tuple[bool, bool]],
    min_confidence: float,
    zone_wall_orientation: str | None = None,
) -> Constraint | None:
    subject_id = str(raw.get("subject_id", ""))
    operator = str(raw.get("operator", ""))
    # "status" is optional: the source-contract prompt does not ask for it.
    status = str(raw.get("status") or "satisfied")

    if operator not in SUPPORTED_OPERATORS:
        return None
    if subject_id not in known_ids:
        return None

    target = raw.get("target") or {}
    # A bare id ("target": "o2") is read as an object target.
    if isinstance(target, str):
        target = {"type": "object", "id": target}
    if not isinstance(target, dict):
        return None
    target_type = str(target.get("type", "object"))
    wall: str | None = None
    # back_to takes only the zone's own (back) wall, and only back_to takes a wall.
    if target_type == "wall":
        if operator != "back_to":
            return None
        if str(target.get("side") or "back").strip().lower() != "back":
            return None
        if zone_wall_orientation not in _CARDINALS:
            return None
        wall = zone_wall_orientation
    elif operator == "back_to":
        return None

    # The operator fixes the interface class; a supplied interface may only refine
    # it, since models often copy "front" onto every relation.
    has_unique_front, has_axis = geometry.get(subject_id, (False, False))
    supplied = str(raw.get("subject_interface") or "")
    fallback = default_interface_for(operator)
    if wall is not None:
        supplied = ""
        fallback = _wall_interface(has_unique_front, has_axis)
    if supplied and (interface_requirement(supplied)
                     != interface_requirement(fallback)):
        supplied = ""
    interface = supplied or fallback
    needs = interface_requirement(interface)
    if needs == "unique_front" and not has_unique_front:
        return None
    if needs == "axis" and not has_axis:
        return None

    if status not in _KEPT_STATUSES:
        return None

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if confidence < min_confidence:
        return None

    # "repair_scope" is optional and defaults by operator; it is checked only when violated.
    scope = str(raw.get("repair_scope") or "")
    if not scope:
        scope = _WALL_SCOPE if wall is not None else _DEFAULT_SCOPE[operator]
    elif status == "satisfied":
        scope = scope if scope in _VALID_SCOPES else _DEFAULT_SCOPE[operator]
    else:
        if scope not in _VALID_SCOPES:
            return None
        if scope == "no_reliable_repair":
            return None

    target_id: str | None = None
    if target_type == "object":
        target_id = str(target.get("id", ""))
        if target_id not in known_ids or target_id == subject_id:
            return None

    return Constraint(
        subject_id=subject_id,
        operator=operator,
        confidence=confidence,
        repair_scope=scope,
        subject_interface=interface,
        target_id=target_id,
        target_type=target_type,
        target_axis=(str(target["axis"]) if target.get("axis") is not None else None),
        target_has_front=(
            bool(geometry.get(target_id, (False, False))[0])
            if target_id is not None else False
        ),
        target_wall=wall,
    )


def parse_constraint_response(
    text: str,
    scene_json: dict[str, Any],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_constraints: int = MAX_RELATIONS,
    zone_wall_orientation: str | None = None,
) -> ConstraintResult:
    """Validate a VLM reply into :class:`Constraint` objects.

    Never raises: unparseable, unknown or non-actionable relations are dropped.
    """
    try:
        payload = json.loads(extract_first_json_object(text))
    except (ValueError, json.JSONDecodeError):
        return ConstraintResult()

    known_ids = {str(o.get("id")) for o in scene_json.get("objects", [])}
    geometry = _geometry_by_id(scene_json)

    # Some replies nest the list under "zone_assessment".
    raw_constraints = payload.get("constraints")
    if not raw_constraints:
        nested = payload.get("zone_assessment")
        if isinstance(nested, dict) and nested.get("constraints"):
            raw_constraints = nested["constraints"]

    kept: list[Constraint] = []
    for raw in raw_constraints or []:
        if not isinstance(raw, dict):
            continue
        parsed = _parse_one(
            raw,
            known_ids=known_ids,
            geometry=geometry,
            min_confidence=min_confidence,
            zone_wall_orientation=zone_wall_orientation,
        )
        if parsed is not None:
            kept.append(parsed)
        if len(kept) >= max_constraints:
            break

    return ConstraintResult(constraints=kept)
