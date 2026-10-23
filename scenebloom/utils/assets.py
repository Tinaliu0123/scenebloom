"""Asset catalog, per-zone candidates, orientation profiles and the category -> mesh mapping."""
from __future__ import annotations

import functools
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenebloom.utils.schemas import AssetRecord, FunctionalZoneSpec, PlacedObject, ZoningPlan

logger = logging.getLogger(__name__)


class AssetDB:
    """In-memory catalog of the ``data/assets.json`` records."""

    def __init__(self, records: list[AssetRecord]) -> None:
        self._records = records
        self._by_id: dict[str, AssetRecord] = {r.asset_id: r for r in records}

    @classmethod
    def from_path(cls, path: Path) -> "AssetDB":
        if not path.exists():
            raise FileNotFoundError(f"Asset DB not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        records = [AssetRecord.model_validate(entry) for entry in raw]
        logger.info("AssetDB loaded %d records from %s", len(records), path)
        return cls(records)

    def get(self, asset_id: str) -> AssetRecord:
        if asset_id not in self._by_id:
            raise KeyError(f"Asset '{asset_id}' not in DB")
        return self._by_id[asset_id]

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[AssetRecord]:
        return list(self._records)

    def query_by_category(self, category: str) -> list[AssetRecord]:
        """Return all assets matching category exactly."""
        return [r for r in self._records if r.category == category]

    def query_by_categories(self, categories: list[str]) -> list[AssetRecord]:
        """Return all assets whose category is in the given list."""
        cat_set = set(categories)
        return [r for r in self._records if r.category in cat_set]

    def query_surface_mounted(self) -> list[AssetRecord]:
        """Return all assets that rest on a support surface (SURFACE mount), in DB order."""
        from scenebloom.utils.schemas import MountType
        return [r for r in self._records if r.mount_type is MountType.SURFACE]

    def query_wall_mounted(self) -> list[AssetRecord]:
        """Return all assets that hang on a wall (WALL mount), in DB order."""
        from scenebloom.utils.schemas import MountType
        return [r for r in self._records if r.mount_type is MountType.WALL]


class AssetSelector:
    """Candidate assets of each functional zone in a ZoningPlan."""

    def __init__(self, asset_db: AssetDB) -> None:
        self._db = asset_db

    def select_for_zone(
        self,
        zone: FunctionalZoneSpec,
        room_type: str,
        style_tags: list[str],
    ) -> list[AssetRecord]:
        """Assets of the zone's allowed categories, by style-tag overlap, then anchors first."""
        candidates = self._db.query_by_categories(zone.allowed_categories)

        # Asset zone/room labels are not applied: they are narrower than the VLM's free
        # zone labels (bookshelf=['workspace'] would vanish from a 'media' zone).
        filtered = list(candidates)
        if not filtered:
            logger.debug(
                "AssetSelector: no candidates for zone '%s' (label=%s, room=%s)",
                zone.zone_id, zone.label, room_type,
            )
            return []

        tag_set = set(style_tags)
        ranked = sorted(
            filtered,
            key=lambda a: (len(set(a.style_tags) & tag_set), int(a.is_anchor)),
            reverse=True,
        )
        return ranked

    def select_for_plan(
        self,
        plan: ZoningPlan,
        style_tags: list[str] | None = None,
    ) -> dict[str, list[AssetRecord]]:
        """Return {zone_id: candidates} for every zone; style_tags default to the plan's."""
        if style_tags is None:
            style_tags = list(plan.scene_spec.style_tags)

        result: dict[str, list[AssetRecord]] = {}
        for room in plan.rooms:
            for zone in room.functional_zones:
                result[zone.zone_id] = self.select_for_zone(
                    zone, room.room_type, style_tags
                )
        return result


def room_allows(allowed_room_types: list[str] | None, room_type: str) -> bool:
    """Whether *room_type* is allowed; empty or "all" *allowed_room_types* means anywhere."""
    if not allowed_room_types or "all" in allowed_room_types:
        return True
    # Substring match either way: asset tables say "living", the zoner "living_room".
    rt = room_type.lower()
    return any(a.lower() in rt or rt in a.lower() for a in allowed_room_types)


VALID_SYMMETRIES = ("C1", "C2", "C4", "Cinf")
VALID_PRIORS = ("strong", "weak", "none")

_DEFAULT_META_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "asset_orientation.json"
)


@dataclass(frozen=True)
class AssetOrientationMeta:
    """Orientation attributes of one asset category (``data/asset_orientation.json``)."""

    category: str
    has_structural_axis: bool
    rotation_symmetry: str
    axis_alignment_prior: str  # only "strong" assets are snapped to a cardinal yaw

    def __post_init__(self) -> None:
        if self.rotation_symmetry not in VALID_SYMMETRIES:
            raise ValueError(
                f"rotation_symmetry {self.rotation_symmetry!r} for "
                f"{self.category!r} not in {VALID_SYMMETRIES}"
            )
        if self.axis_alignment_prior not in VALID_PRIORS:
            raise ValueError(
                f"axis_alignment_prior {self.axis_alignment_prior!r} for "
                f"{self.category!r} not in {VALID_PRIORS}"
            )


def _conservative_default(category: str) -> AssetOrientationMeta:
    """Profile for categories absent from the DB: never snapped."""
    return AssetOrientationMeta(
        category=category,
        has_structural_axis=False,
        rotation_symmetry="Cinf",
        axis_alignment_prior="none",
    )


class AssetMetadataDB:
    """Read-only lookup of :class:`AssetOrientationMeta` by category."""

    def __init__(self, entries: dict[str, AssetOrientationMeta]) -> None:
        self._entries = dict(entries)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AssetMetadataDB":
        """Load and validate the JSON table (default: the repo's data file)."""
        meta_path = Path(path) if path is not None else _DEFAULT_META_PATH
        raw = json.loads(meta_path.read_text())
        entries: dict[str, AssetOrientationMeta] = {}
        for row in raw["entries"]:
            meta = AssetOrientationMeta(
                category=row["category"],
                has_structural_axis=bool(row["has_structural_axis"]),
                rotation_symmetry=row["rotation_symmetry"],
                axis_alignment_prior=row["axis_alignment_prior"],
            )
            if meta.category in entries:
                raise ValueError(f"duplicate category {meta.category!r} in {meta_path}")
            entries[meta.category] = meta
        return cls(entries)

    def get(self, category: str) -> AssetOrientationMeta:
        """Return metadata for ``category`` (conservative default if unknown)."""
        return self._entries.get(category) or _conservative_default(category)

    def __len__(self) -> int:
        return len(self._entries)


_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LIBRARY_GLB_DIR: Path = _DATA_DIR / "lora_assets" / "glb"
# Per mesh: GLB ``bounding_box_extents`` and ``center``; floor furniture adds ``footprint``.
LIBRARY_METADATA_PATH: Path = _DATA_DIR / "meshes.json"
_LIBRARY_PREFIX = "scenebloom:"


@functools.lru_cache(maxsize=1)
def _library_metadata() -> dict[str, Any]:
    """Load and cache ``data/meshes.json``."""
    if not LIBRARY_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"library Genesis metadata not found at {LIBRARY_METADATA_PATH}"
        )
    with LIBRARY_METADATA_PATH.open() as fh:
        return json.load(fh)


_LIBRARY_ALIASES: dict[str, str] = {
    # Dining zones are drawn with wooden chairs; the base ``chair`` is an office chair.
    "dining_chair": "chair_01",
}


# Categories without a TRELLIS base mesh render this ArtVIP mesh, the one closest to the
# catalogue size in log-space.
_ARTVIP_MESH_FOR_CATEGORY: dict[str, str] = {
    "cabinet": "cabinet_art01",
    "chest_of_drawers": "chest_of_drawers_art03",
    "desk": "desk_art01",
    "dressing_table": "dressing_table_art02",
    "nightstand": "nightstand_art02",
    "toilet": "toilet_art02",
    "tv_stand": "tv_stand_art01",
    "wardrobe": "wardrobe_art02",
}


def _build_category_to_glb_id() -> dict[str, str]:
    cats = list(_library_metadata().keys())
    mapping = {c: f"{_LIBRARY_PREFIX}{c}" for c in cats}
    for c, r in _ARTVIP_MESH_FOR_CATEGORY.items():
        if r in cats:
            mapping[c] = f"{_LIBRARY_PREFIX}{r}"
    for alias, canon in _LIBRARY_ALIASES.items():
        if canon in mapping:
            mapping[alias] = mapping[canon]
    return mapping


CATEGORY_TO_GLB_ID: dict[str, str] = _build_category_to_glb_id()

GENESIS_COMPATIBLE_CATEGORIES: frozenset[str] = frozenset(CATEGORY_TO_GLB_ID.keys())


# `_<NN>` is a static mesh variant, `_art<NN>` an ArtVIP articulated one.
_VARIANT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?:art)?\d+$")


def variant_keys(category: str) -> list[str]:
    """Sorted mesh-variant keys of the alias-resolved *category*; may be empty."""
    canonical = _LIBRARY_ALIASES.get(category, category)
    return sorted(
        k for k in CATEGORY_TO_GLB_ID
        if (m := _VARIANT_SUFFIX_RE.match(k)) and m.group("base") == canonical
    )


def effective_lookup_key(asset_id: str, category: str) -> str:
    """GLB/metadata key of a placed object: ``asset_id`` if a known variant, else the category."""
    if asset_id != category and asset_id in CATEGORY_TO_GLB_ID:
        return asset_id
    return category


def resolve_glb(category: str) -> Path:
    """Return the GLB path of *category*."""
    try:
        asset_id = CATEGORY_TO_GLB_ID[category]
    except KeyError as exc:
        raise KeyError(
            f"category {category!r} is not Genesis-compatible. "
            f"Supported: {sorted(GENESIS_COMPATIBLE_CATEGORIES)}"
        ) from exc

    stem = asset_id[len(_LIBRARY_PREFIX):]
    library_path = LIBRARY_GLB_DIR / f"{stem}.glb"
    if library_path.is_file():
        return library_path
    raise FileNotFoundError(
        f"library GLB missing for category={category!r}: {library_path}"
    )


def get_object_metadata(asset_id: str) -> dict[str, Any]:
    """Return the mesh metadata (``bounding_box_extents``, ``center``) of ``scenebloom:<key>``."""
    if not asset_id.startswith(_LIBRARY_PREFIX):
        raise KeyError(f"asset_id {asset_id!r} is not a library id ({_LIBRARY_PREFIX}<key>)")
    cat = asset_id[len(_LIBRARY_PREFIX):]
    md = _library_metadata()
    if cat not in md:
        raise KeyError(f"library category {cat!r} not in {LIBRARY_METADATA_PATH}")
    return md[cat]


_QUARTER_TURNS: dict[int, int] = {0: 0, 90: 1, 180: 2, 270: 3}


def _planar_center_offset(
    center_x: float,
    center_z: float,
    rotation_offset_deg: float,
) -> tuple[float, float]:
    """Unscaled planar offset of the bbox centre after the render yaw (subtracted from pos)."""
    # Exact quarter turns only swap and negate, so axis-aligned renders carry no trig rounding.
    turn = rotation_offset_deg % 360
    quarters = _QUARTER_TURNS.get(turn)
    if quarters is None:
        c = math.cos(math.radians(turn))
        s = math.sin(math.radians(turn))
        return c * center_z + s * center_x, c * center_x - s * center_z
    a, b = center_z, center_x
    for _ in range(quarters):
        a, b = b, -a
    return a, b


def _render_yaw_deg(rotation_y: float) -> float:
    """Genesis Z rotation (degrees, in [0, 360)) for a layout yaw in radians."""
    return math.degrees(rotation_y) % 360


def _mesh_scale(
    placed: PlacedObject,
    bbox: Any,
) -> tuple[Any, tuple[float, float, float]]:
    """``(scale for gs.morphs.Mesh, per-axis factors)`` stretching the mesh to the placed box."""
    # GLB-local (x, y_up, z) is (width, height, depth) before yaw.
    factors = tuple(
        want / have if have > 0 else 1.0
        for want, have in zip(placed.pose.size, bbox)
    )
    logger.info(
        "object_mesh_pose: %s scale=(%.3f, %.3f, %.3f)",
        placed.instance_id, *factors,
    )
    return factors, factors


@dataclass(frozen=True)
class ObjectMeshPose:
    """Arguments of ``gs.morphs.Mesh`` for one placed object."""

    asset_id: str
    glb_path: Any
    scale: Any  # per mesh-local axis, applied before the rotation
    pos: tuple  # Genesis frame (x, y_floor, z_height)
    euler: tuple  # degrees, Rotation.from_euler("xyz")


def object_mesh_pose(
    placed: PlacedObject,
) -> ObjectMeshPose | None:
    """GLB, scale and Genesis pose of one placed object; None if its category has no mesh."""
    lookup = effective_lookup_key(placed.asset_id, placed.category)
    if lookup not in CATEGORY_TO_GLB_ID:
        logger.warning(
            "object_mesh_pose: skipping %s (category %r has no GLB mapping)",
            placed.instance_id, placed.category,
        )
        return None

    asset_id = CATEGORY_TO_GLB_ID[lookup]
    glb_path = resolve_glb(lookup)
    meta = get_object_metadata(asset_id)
    bbox = meta["bounding_box_extents"]
    center = meta["center"]
    scale, (fx, fy, fz) = _mesh_scale(placed, bbox)
    yaw = _render_yaw_deg(placed.pose.rotation_y)

    # Rest the bbox bottom at the layout height and move the yawed bbox centre onto the
    # layout centre. Layout (x, y_up, z) -> Genesis (x, y_floor, z_up).
    lift = (bbox[1] / 2 - center[1]) * fy
    dx, dz = _planar_center_offset(center[0], center[2], yaw)
    dx *= fx
    dz *= fz
    x, y_up, z = placed.pose.center
    return ObjectMeshPose(
        asset_id=asset_id,
        glb_path=glb_path,
        scale=scale,
        pos=(x - dx, z - dz, y_up + lift),
        euler=(90, 0, yaw),
    )
