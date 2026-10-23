"""Functional zoning: SceneSpec + retrieved exemplars -> ZoningPlan via one VLM call."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from scenebloom.utils.assets import LIBRARY_METADATA_PATH, AssetDB, variant_keys
from scenebloom.utils.paths import DATA
from scenebloom.utils.schemas import (
    AssetRecord,
    BBox3D,
    FunctionalZoneSpec,
    MountType,
    RoomSpec,
    SceneRecord,
    SceneSpec,
    ZoningPlan,
)
from scenebloom.sampling.proposals.prompts import (
    DEFAULT_CATEGORY_ALIASES,
    SUBJECT_META,
    alias_replacements,
    prompt_named_subjects,
    rendered_mesh,
    split_subjects_by_prompt,
)
from scenebloom.sampling.zoning.retrieval import (
    image_path_for_scene,
    to_rich_json,
)

logger = logging.getLogger(__name__)


_MODEL = "qwen/qwen3-vl-8b-instruct"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_TOKENS = 2048

#: __ALLOWED_CATEGORIES__ is filled below by str.replace (str.format would trip on the JSON braces).
_SYSTEM_PROMPT = """\
You are a furniture layout expert specialising in indoor scene generation.

You will receive up to 3 reference indoor floor plans. Use them only as spatial-layout priors, and design for the room type given in the user prompt. Each reference includes:
  * A top-down rendered image (256 by 256 px, if available).
  * A JSON object with the scene's room_type, object_count, and a list of objects, each with category, centre_xz [x, z] in metres, size_xz [width, depth] in metres, height in metres, and rotation_y in radians.

Your task is to analyse the references and output a JSON zone plan for the room described in the user's prompt. The zone plan partitions the room floor into non-overlapping rectangular zones that together cover the entire floor.

## Output schema (respond with ONLY valid JSON, no markdown fences, no commentary)

{
  "room_type": "<string>",
  "room_width_m": <float, metres>,
  "room_depth_m": <float, metres>,
  "zones": [
    {
      "label": "<string, e.g. sleeping | workspace | seating | dining | storage | circulation | bathing | washing | toilet>",
      "polygon_2d": [[x1,z1],[x2,z2],[x3,z3],[x4,z4]],
      "allowed_categories": ["<category>", ...],
      "category_counts": {"<category>": <int count>, ...},
      "anchor_category": "<category or null>",
      "max_objects": <int>,
      "priority": <int, higher placed first>,
      "diffusion_prompt": "<string, ~1 sentence describing this zone's visual content for a text-to-image diffusion model>",
      "wall_orientation": "<one of: north | south | east | west | null>"
    }
  ]
}

## Rules

### A. Zone geometry

1. polygon_2d must be axis-aligned rectangles (4 vertices, metres, local XZ frame with origin at the room min corner).
2. Zones must tile the full room floor without overlap and without gaps.
3. Each zone's bounding box must fit within [0, room_width_m] by [0, room_depth_m].

### B. Room dimensions

4. Estimate room dimensions from the reference scenes (use the median object spread).
5. Room dimensions should be realistic for the room type, as the reference scenes show it.

### C. What each zone contains

6. You are placing FLOOR STANDING furniture, and these are the only category names that exist for it (you may ONLY use these exact names in allowed_categories): __ALLOWED_CATEGORIES__. Do not use any other category names (e.g. use "chair" not "officechair", "nightstand" not "bedsidetable"). Pick categories appropriate to the room type (e.g. a bathroom uses toilet / bathtub / bathroom_washstand / cabinet; a dining room uses dining_table).
7. The list above is complete for what you decide. If a room would normally also contain something that hangs on a wall or sits on top of another object, that is not missing and is not yours: later stages of this same pipeline furnish walls and surfaces from their own catalogues, after you. Every one of your zones will receive those additions without you asking for them. What you write here is read only against the list above, so a name outside it reserves floor area and a count that nothing can ever fill, and the zone is built emptier than you designed it.
8. category_counts declares HOW MANY of each category the zone should contain. Map each category in allowed_categories to an integer count. Default to 1 (a single instance). Use a count above 1 ONLY when the room type naturally has multiples of that item. Let the reference scenes show you where multiples are natural; decide the count from the room you are actually designing. Do NOT pad a zone with extra distinct categories just to reach a subject count; a single-anchor zone (one category, count 1) is perfectly valid. Use a circulation or empty-floor zone (allowed_categories=[], category_counts={}, anchor_category=null, max_objects=0) to absorb leftover space.
9. anchor_category, if set, must appear in allowed_categories.
10. Size each zone's polygon_2d plausibly for the furniture you assign it: real furniture of these categories occupies real floor area, and the reference scenes show typical rooms of this type, so let them guide how much of the room a zone takes. If the room would be too small to hold an anchor you want to use, increase room_width_m / room_depth_m, and do not skip anchors.

### D. Zone text and wall side

11. diffusion_prompt: a concise description (at most 25 words) of this zone. Refer to each furniture item by its EXACT allowed_categories name with NO adjective before it (write "chair", never "visitor chair" or "comfortable chair"; write "bed", never "hospital bed"), because a downstream system injects the object's trained token right before that name, so any adjective you add corrupts it. State the rough RELATIVE relation between the items so the layout reads naturally. <example>Phrasings of the right shape: "chair in front of desk", "nightstand next to the bed", "lamp beside the sofa". These show the STYLE to imitate, not content to copy: write the relation your own zone actually has.</example> Describe ONLY the furniture listed in THIS zone's category_counts, using the same names: every furniture noun in the diffusion_prompt must have a count in this zone. Leave out furniture that only another zone plans; the same category may still be planned in several zones, each zone describing its own. You may add a few words of material, lighting or ambiance AFTER the objects. Do NOT use absolute-position words ("on the left", "in the corner"), because the polygon already encodes absolute position; use only relative inter-object relations. This field drives a text-to-image model whose output is lifted back into FLOOR-STANDING furniture only, so describe only items that rest on the floor and never mention wall-mounted, hanging, built-in or ceiling objects (no wall-mounted TV, no picture frames, no ceiling lamp). Wall decor is chosen by a separate later stage and must not appear here.
12. Set wall_orientation to the room-frame side the zone's primary wall faces. The room is axis-aligned with x in [0, room_width_m] and z in [0, room_depth_m]. Use this rule:
    * "north" if the zone polygon's high-z edge equals room_depth_m (zone touches the +z wall);
    * "south" if the zone polygon's low-z edge equals 0 (zone touches the -z wall, i.e. z = 0);
    * "east" if the zone polygon's high-x edge equals room_width_m (zone touches the +x wall);
    * "west" if the zone polygon's low-x edge equals 0 (zone touches the -x wall, i.e. x = 0);
    * null for interior zones (e.g. circulation) that do not touch any room wall, OR for corner zones where any choice would distort placement.
    When a zone touches multiple room walls (corner zone), pick the wall along the side where the wall-facing anchor (e.g. bed headboard) belongs, typically the longest shared edge. For circulation or empty zones with anchor_category=null, set wall_orientation=null.
"""


#: Surface objects a diffusion_prompt must not draw as floor furniture (rule 11).
_NOT_FLOOR_FURNITURE: frozenset[str] = frozenset({
    "sink",  # bathroom_washstand is the floor-standing one
    "television",
})


#: Exact synonyms the zoner writes for a pool category, renamed before the vocabulary gate.
_ZONER_CATEGORY_ALIASES: dict[str, str] = {
    "dressing_chair": "chair",
}


def _alias_zone_vocabulary(zone: dict, allowed: set) -> list[tuple[str, str]]:
    """Rename aliased names in ``zone`` in place onto ``allowed`` targets; return the (from, to) pairs."""
    applied: list[tuple[str, str]] = []

    def _tgt(name) -> str | None:
        t = _ZONER_CATEGORY_ALIASES.get(str(name))
        return t if t in allowed else None

    counts = zone.get("category_counts") or {}
    new_counts: dict[str, int] = {}
    for k, v in counts.items():
        t = _tgt(k)
        if t:
            applied.append((str(k), t))
            new_counts[t] = new_counts.get(t, 0) + int(v)
        else:
            new_counts[str(k)] = int(v)
    if counts:
        zone["category_counts"] = new_counts

    cats = zone.get("allowed_categories") or []
    if cats:
        out: list[str] = []
        for c in cats:
            t = _tgt(c)
            if t:
                applied.append((str(c), t))
                if t not in out:
                    out.append(t)
            elif c not in out:
                out.append(str(c))
        zone["allowed_categories"] = out

    anchor = zone.get("anchor_category")
    t = _tgt(anchor) if anchor is not None else None
    if t:
        applied.append((str(anchor), t))
        zone["anchor_category"] = t
    return list(dict.fromkeys(applied))


def _library_nouns() -> dict[str, str]:
    """category -> noun, the mapping the FLUX injector also reads."""
    return {cat: meta["noun"] for cat, meta in SUBJECT_META.items()}


def _floor_records() -> dict[str, AssetRecord]:
    """The first floor-furniture row of ``data/assets.json`` per category."""
    records: dict[str, AssetRecord] = {}
    for r in AssetDB.from_path(DATA / "assets.json").all():
        if r.mount_type is MountType.FLOOR:
            records.setdefault(r.category, r)
    return records


def _library_categories() -> list[str]:
    """Plannable categories: trained-LoRA categories that data/assets.json lists as floor furniture."""
    floor = _floor_records()
    return [c for c in SUBJECT_META if c in floor]


_SYSTEM_PROMPT = _SYSTEM_PROMPT.replace(
    "__ALLOWED_CATEGORIES__", ", ".join(_library_categories())
)


_CATALOG_PREAMBLE = """\
## Asset catalog (concrete assets available for placement)

Read this catalog for SIZES only. The headings are category names and are the only names you may write into allowed_categories, anchor_category or category_counts. The line under a heading says how many meshes exist for that category and the range of their footprints (width by depth); the downstream selector picks which mesh is used, so never write a mesh count or a size into the plan. Your zone must fit the LARGEST footprint in the range.
"""


def _catalog_footprints(category: str, meshes: dict[str, dict]) -> list[tuple[float, float]]:
    """(width, depth) of each mesh the render may load for ``category`` (its own, else a borrowed one)."""
    own = [k for k in (category, *variant_keys(category)) if "footprint" in meshes.get(k, {})]
    keys = own or [rendered_mesh(category)]
    if any("footprint" not in meshes.get(k, {}) for k in keys):
        raise ValueError(f"{category}: no mesh with a footprint in {LIBRARY_METADATA_PATH}")
    return [tuple(meshes[k]["footprint"]) for k in keys]


def _build_asset_catalog_text() -> str:
    """Size catalogue for the user turn: mesh count and footprint range per plannable category."""
    meshes = json.loads(LIBRARY_METADATA_PATH.read_text(encoding="utf-8"))
    floor = _floor_records()

    def _rng(v):
        return f"{min(v):.2f}" if abs(max(v) - min(v)) < 0.005 else f"{min(v):.2f}-{max(v):.2f}"

    parts = [_CATALOG_PREAMBLE]
    for heading in sorted(_library_categories()):
        fps = _catalog_footprints(heading, meshes)
        ws, ds = [w for w, _ in fps], [d for _, d in fps]
        # No clearance added; the partition step re-sizes the room from the measured objects.
        line = f"    {len(fps)} mesh{'es' if len(fps) != 1 else ''}, footprint {_rng(ws)} by {_rng(ds)} m"
        record = floor[heading]
        if record.is_anchor:
            labels = sorted(record.allowed_zone_labels)
            line += "   anchor" + (f" (zone: {', '.join(labels)})" if labels else "")
        parts.append(f"  {heading}:\n{line}\n")
    return "\n".join(parts)


def _head_noun(category: str) -> str:
    return category.split("_")[-1].lower()


#: Max zoner reads; a plan whose prompts and counts disagree is re-asked with a note.
ZONER_MAX_READS = 3
ZONER_REREAD_BACKOFF_S = (3.0, 8.0)
_REREAD_NOTE_HEADER = (
    "Your plan is not self-consistent: in the zones listed below the diffusion_prompt and the "
    "category_counts do not name the same furniture. Fix ONLY these zones -- either give the "
    "named item a count in that zone, or remove it from that zone's diffusion_prompt, whichever "
    "matches the room you designed -- keep every other zone unchanged, and return the complete "
    "JSON again."
)
_REREAD_NOTE_LINES = {
    "unplanned_named": "- zone {label}: the diffusion_prompt names {names} but category_counts has no count for it",
    "planned_unnamed": "- zone {label}: category_counts plans {names} but the diffusion_prompt never names it",
    "nonfloor_named": "- zone {label}: the diffusion_prompt names {names}, which is not floor furniture (rule 11)",
}
_REREAD_NOTE_SPEC = _REREAD_NOTE_HEADER + "\n" + "\n".join(
    _REREAD_NOTE_LINES[k] for k in ("unplanned_named", "planned_unnamed", "nonfloor_named"))


def _zone_inconsistencies(
    counts: dict[str, int],
    prompt: str | None,
    *,
    pool: set[str],
    nouns: dict[str, str],
) -> dict[str, list[str]] | None:
    """Prompt-vs-count mismatches of one zone, found with FreeFuse's matcher; None if consistent."""
    text = prompt or ""
    planned = [c for c in counts if c in pool]
    heads = {_head_noun(c) for c in planned}
    named = set(prompt_named_subjects(text, sorted(pool | set(_NOT_FLOOR_FURNITURE)), nouns))
    unplanned = sorted(c for c in named if c in pool and c not in counts and _head_noun(c) not in heads)
    nonfloor = sorted(c for c in named if c in _NOT_FLOOR_FURNITURE)
    if planned and text:
        named_planned, _ = split_subjects_by_prompt(
            text, planned, nouns, alias_replacements(planned, nouns, DEFAULT_CATEGORY_ALIASES))
        unnamed = sorted(c for c in planned if c not in named_planned)
    else:
        unnamed = sorted(planned) if planned else []
    if not (unplanned or unnamed or nonfloor):
        return None
    return {"unplanned_named": unplanned, "planned_unnamed": unnamed, "nonfloor_named": nonfloor}


def _plan_inconsistencies(plan: ZoningPlan) -> list[dict[str, Any]]:
    """Per-zone inconsistencies of a parsed plan (post vocabulary gate)."""
    pool = set(_library_categories())
    nouns = _library_nouns()
    out: list[dict[str, Any]] = []
    for room in plan.rooms:
        for z in room.functional_zones:
            inc = _zone_inconsistencies(z.category_counts, z.diffusion_prompt, pool=pool, nouns=nouns)
            if inc:
                out.append({"zone_label": z.label, **inc})
    return out


def _reread_note(inconsistencies: list[dict[str, Any]]) -> str:
    lines = [_REREAD_NOTE_HEADER]
    for inc in inconsistencies:
        for key, tmpl in _REREAD_NOTE_LINES.items():
            if inc.get(key):
                lines.append(tmpl.format(label=repr(inc["zone_label"]), names=", ".join(inc[key])))
    return "\n".join(lines)


class LLMZoneDecomposer:
    """Splits a room into functional zones with a multimodal VLM through OpenRouter."""

    def __init__(
        self,
        m3dlayout_root: Path,
        model: str = _MODEL,
        top_k_scenes: int = 3,
    ) -> None:
        self._m3dlayout_root = m3dlayout_root
        self._model = model
        self._top_k_scenes = top_k_scenes
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai SDK required for zone decomposition. "
                "Install with: pip install openai"
            ) from exc
        self._client = openai.OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        return self._client

    def decompose(
        self,
        spec: SceneSpec,
        retrieved_scenes: list[SceneRecord],
    ) -> ZoningPlan:
        """Zone the room from the exemplars; an inconsistent plan is re-read, keeping the best one."""
        scenes = retrieved_scenes[: self._top_k_scenes]
        convo = self._build_messages(spec, scenes)
        best: ZoningPlan | None = None
        best_inc: list[dict[str, Any]] = []

        for attempt in range(1, ZONER_MAX_READS + 1):
            # No temperature is sent (provider default).
            raw_text = self._read_once(convo)
            logger.debug("LLM raw response (read %d):\n%s", attempt, raw_text)
            try:
                plan = self._parse_response(raw_text, spec)
            except ValueError:
                if attempt == 1:
                    raise
                logger.warning("zoner re-read %d returned unparsable JSON; keeping the best earlier read", attempt)
                break
            inc = _plan_inconsistencies(plan)
            if best is None or len(inc) <= len(best_inc):
                best, best_inc = plan, inc       # tie -> latest
            if not inc:
                break
            if attempt == ZONER_MAX_READS:
                break
            note = _reread_note(inc)
            convo = [*convo,
                     {"role": "assistant", "content": [{"type": "text", "text": raw_text}]},
                     {"role": "user", "content": [{"type": "text", "text": note}]}]
            logger.info("zoner plan inconsistent in %d zone(s); re-reading (%d/%d)",
                        len(inc), attempt + 1, ZONER_MAX_READS)
            time.sleep(ZONER_REREAD_BACKOFF_S[min(attempt - 1, len(ZONER_REREAD_BACKOFF_S) - 1)])

        assert best is not None  # attempt 1 either parsed or raised
        return best

    def _read_once(self, messages: list[dict[str, Any]]) -> str:
        """One provider call -> the raw reply text."""
        response = self._get_client().chat.completions.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content

    def _build_messages(
        self,
        spec: SceneSpec,
        scenes: list[SceneRecord],
    ) -> list[dict[str, Any]]:
        """OpenAI chat messages (system, user) for one zoner call."""
        user_content: list[dict[str, Any]] = []

        for i, scene in enumerate(scenes):
            img_path = image_path_for_scene(scene.scene_id, self._m3dlayout_root)
            if img_path is not None:
                img_data = base64.standard_b64encode(img_path.read_bytes()).decode()
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                })

            rich = to_rich_json(scene)
            user_content.append({
                "type": "text",
                "text": (
                    f"Reference scene {i + 1} of {len(scenes)} "
                    f"(score={scene.similarity_score:.3f}):\n"
                    + json.dumps(rich, indent=2)
                ),
            })

        rc = spec.room_constraints[0] if spec.room_constraints else None
        room_type = rc.room_type if rc else "bedroom"
        cats = ", ".join(rc.required_categories) if rc and rc.required_categories else "any"

        user_content.append({
            "type": "text",
            "text": _build_asset_catalog_text(),
        })

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Original prompt: {spec.prompt_raw}\n\n"
                    f"Room type: {room_type}\n"
                    f"Required categories: {cats}\n\n"
                    "Produce the JSON zone plan for this room."
                ),
            }
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(
        self,
        raw_text: str,
        spec: SceneSpec,
    ) -> ZoningPlan:
        """Parse LLM JSON output into a ZoningPlan."""
        # Strip accidental markdown fences
        text = re.sub(r"```[a-z]*\n?", "", raw_text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw_text[:500]}"
            ) from exc

        room_type = data.get("room_type", "bedroom")
        w = float(data.get("room_width_m", 4.5))
        d = float(data.get("room_depth_m", 4.0))
        ceiling_h = 2.7

        room_bbox = BBox3D(
            center=(w / 2, ceiling_h / 2, d / 2),
            size=(w, ceiling_h, d),
        )

        zones: list[FunctionalZoneSpec] = []
        raw_zones = data.get("zones", [])
        all_polys = [[tuple(pt) for pt in zz.get("polygon_2d", [])] for zz in raw_zones]
        for zi, z in enumerate(raw_zones):
            poly = [tuple(pt) for pt in z["polygon_2d"]]
            diffusion_prompt = z.get("diffusion_prompt")
            if diffusion_prompt is not None:
                diffusion_prompt = str(diffusion_prompt).strip() or None
            # Check the declared wall against the polygon (rule 12); a missing one is inferred.
            declared = z.get("wall_orientation")
            if isinstance(declared, str):
                declared = declared.strip().lower() or None
                if declared not in ("north", "south", "east", "west"):
                    declared = None
            wall_orientation, override_note = _reconcile_wall_orientation(
                declared, poly, room_width_m=w, room_depth_m=d,
            )
            # A wall zone that stops short of its wall, with nothing in between, is extended onto it.
            poly, extend_note = _extend_polygon_to_wall(
                wall_orientation, poly,
                [pp for pj, pp in enumerate(all_polys) if pj != zi],
                room_width_m=w, room_depth_m=d,
            )
            if extend_note is not None:
                logger.info("zoner wall_extend: zone %r (%s): %s",
                            z.get("label"), wall_orientation, extend_note)
            if override_note is not None:
                logger.warning(
                    "zoner wall_orientation: zone %r declared %r but its polygon "
                    "touches only %s; using %r",
                    z.get("label"), declared, override_note, wall_orientation,
                )
            # Vocabulary gate: drop (and log) names outside the plannable pool; nothing could fill them.
            _allowed = set(_library_categories())
            # Exact synonyms are renamed first so the gate keeps them.
            _aliased = _alias_zone_vocabulary(z, _allowed)
            if _aliased:
                logger.info("zoner vocabulary alias: zone %r renamed %s",
                            z.get("label"), _aliased)
            _cats = [c for c in (z.get("allowed_categories") or [])
                     if c in _allowed]
            _dropped = [c for c in (z.get("allowed_categories") or [])
                        if c not in _allowed]
            _counts = {str(k): int(v)
                       for k, v in (z.get("category_counts") or {}).items()
                       if str(k) in _allowed}
            _dropped += [str(k) for k in (z.get("category_counts") or {})
                         if str(k) not in _allowed]
            _anchor = z.get("anchor_category")
            if _anchor is not None and _anchor not in _allowed:
                _dropped.append(_anchor)
                _anchor = None
            # Rule 9: an anchor whose category the gate removed is cleared.
            if _anchor is not None and _anchor not in _cats:
                _anchor = None
            if _dropped:
                logger.warning(
                    "zoner vocabulary gate: removed %d category name(s) "
                    "outside the catalog from zone %r: %s",
                    len(set(_dropped)), z.get("label"), sorted(set(_dropped)),
                )
            # Reconcile: named in prompt -> count 1, counted -> allowed, uncounted -> stripped.
            _nouns = _library_nouns()
            _named, _ = split_subjects_by_prompt(
                diffusion_prompt, _cats, _nouns,
                alias_replacements(_cats, _nouns, DEFAULT_CATEGORY_ALIASES),
            )
            _filled = [c for c in _named if c not in _counts]
            for c in _filled:
                _counts[c] = 1
            _anchor_kept = None
            if _anchor is not None and _anchor not in _counts:
                _counts[_anchor] = 1
                _anchor_kept = _anchor
            _added = [c for c in _counts if c not in _cats]
            _stripped = [c for c in _cats if c not in _counts]
            _cats = [c for c in _cats if c in _counts] + _added
            if _filled or _stripped or _added or _anchor_kept:
                logger.info(
                    "zoner counts reconcile: zone %r filled %s, stripped %s, "
                    "added %s, anchor kept %s",
                    z.get("label"), _filled, _stripped, _added, _anchor_kept,
                )
            zones.append(
                FunctionalZoneSpec(
                    zone_id=str(uuid.uuid4()),
                    label=z["label"],
                    polygon_2d=poly,
                    allowed_categories=_cats,
                    category_counts=_counts,
                    anchor_category=_anchor,
                    diffusion_prompt=diffusion_prompt,
                    wall_orientation=wall_orientation,
                )
            )

        if not zones:
            raise ValueError("LLM returned no zones in the plan.")

        room_spec = RoomSpec(
            room_id=str(uuid.uuid4()),
            room_type=room_type,
            bbox=room_bbox,
            functional_zones=zones,
        )

        return ZoningPlan(
            scene_spec=spec,
            rooms=[room_spec],
            ceiling_h=ceiling_h,
        )


#: Largest gap (m) to its wall that a zone keeps; a wider gap is extended.
_WALL_EXTEND_MIN_GAP = 0.3


def _extend_polygon_to_wall(
    wall: str | None,
    polygon: list[tuple[float, float]],
    other_polygons: list[list[tuple[float, float]]],
    room_width_m: float,
    room_depth_m: float,
    *,
    min_gap: float = _WALL_EXTEND_MIN_GAP,
    tol: float = 0.05,
) -> tuple[list[tuple[float, float]], str | None]:
    """Snap the wall-side edge onto ``wall`` if the gap exceeds ``min_gap`` and no zone is in between."""
    if wall not in ("north", "south", "east", "west") or not polygon:
        return polygon, None
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    axis = 0 if wall in ("east", "west") else 1
    coord = {"east": room_width_m, "west": 0.0, "north": room_depth_m, "south": 0.0}[wall]
    vals = xs if axis == 0 else zs
    edge = max(vals) if wall in ("east", "north") else min(vals)
    gap = (coord - edge) if wall in ("east", "north") else (edge - coord)
    if gap <= min_gap:
        return polygon, None
    # Leave the polygon if another zone overlaps the strip between its edge and the wall.
    lo_s, hi_s = (min(zs), max(zs)) if axis == 0 else (min(xs), max(xs))
    lo_n, hi_n = (edge, coord) if wall in ("east", "north") else (coord, edge)
    for other in other_polygons:
        if not other:
            continue
        oxs = [p[0] for p in other]
        ozs = [p[1] for p in other]
        o_along = (min(ozs), max(ozs)) if axis == 0 else (min(oxs), max(oxs))
        o_norm = (min(oxs), max(oxs)) if axis == 0 else (min(ozs), max(ozs))
        along_ov = min(hi_s, o_along[1]) - max(lo_s, o_along[0])
        norm_ov = min(hi_n, o_norm[1]) - max(lo_n, o_norm[0])
        if along_ov > tol and norm_ov > tol:
            return polygon, None
    moved = 0
    out = []
    for x, zz in polygon:
        v = x if axis == 0 else zz
        if abs(v - edge) <= tol:
            moved += 1
            out.append((coord, zz) if axis == 0 else (x, coord))
        else:
            out.append((x, zz))
    if moved == 0:
        return polygon, None
    return out, f"extended {gap:.2f} m to the {wall} wall ({moved} vertices moved)"


def _walls_touched_by_polygon(
    polygon: list[tuple[float, float]],
    room_width_m: float,
    room_depth_m: float,
    tol: float = 0.05,
) -> list[tuple[str, float]]:
    """Walls the polygon's bbox touches, with the shared edge length, longest first."""
    if not polygon:
        return []
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    zone_min_x, zone_max_x = min(xs), max(xs)
    zone_min_z, zone_max_z = min(zs), max(zs)
    zone_width = zone_max_x - zone_min_x
    zone_depth = zone_max_z - zone_min_z

    candidates: list[tuple[str, float]] = []
    if zone_max_z >= room_depth_m - tol:
        candidates.append(("north", zone_width))
    if zone_min_z <= tol:
        candidates.append(("south", zone_width))
    if zone_max_x >= room_width_m - tol:
        candidates.append(("east", zone_depth))
    if zone_min_x <= tol:
        candidates.append(("west", zone_depth))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates


def _reconcile_wall_orientation(
    declared: str | None,
    polygon: list[tuple[float, float]],
    room_width_m: float,
    room_depth_m: float,
    tol: float = 0.05,
) -> tuple[str | None, str | None]:
    """Keep a declared wall the polygon touches (or any, if interior); else the longest touched wall."""
    touched = _walls_touched_by_polygon(polygon, room_width_m, room_depth_m, tol=tol)
    inferred = touched[0][0] if touched else None
    if declared is None:
        return inferred, None
    if not touched or declared in {wall for wall, _ in touched}:
        return declared, None
    return inferred, ",".join(wall for wall, _ in touched)
