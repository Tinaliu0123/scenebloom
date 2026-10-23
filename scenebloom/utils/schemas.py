"""Pydantic data models of the pipeline: parsed prompt, zoning plan and layout."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_BASE_CFG = ConfigDict(frozen=False, extra="forbid")


class BBox3D(BaseModel):
    """Yawed box in metres; for a placed object ``center`` y is the base, not the centre."""

    model_config = _BASE_CFG

    center: tuple[float, float, float]
    size: tuple[float, float, float]  # (width, height, depth) metres
    rotation_y: float = 0.0  # radians, yaw only

    @field_validator("size")
    @classmethod
    def _positive_size(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s <= 0 for s in v):
            raise ValueError(f"BBox3D.size must be positive, got {v}")
        return v

    @property
    def min_x(self) -> float:
        return self.center[0] - self.size[0] / 2

    @property
    def max_x(self) -> float:
        return self.center[0] + self.size[0] / 2

    @property
    def min_z(self) -> float:
        return self.center[2] - self.size[2] / 2

    @property
    def max_z(self) -> float:
        return self.center[2] + self.size[2] / 2

    @property
    def floor_area_m2(self) -> float:
        return self.size[0] * self.size[2]


class RoomConstraint(BaseModel):
    """Constraints for a single room derived from the prompt."""

    model_config = _BASE_CFG

    room_type: str  # "bedroom", "living_room", "office", …
    required_categories: list[str] = []


class SceneSpec(BaseModel):
    """Structured representation of a text prompt."""

    model_config = _BASE_CFG

    prompt_raw: str
    scene_type: str
    room_constraints: list[RoomConstraint]
    style_tags: list[str]


class SceneRecord(BaseModel):
    """One room of the M3DLayout database."""

    model_config = _BASE_CFG

    scene_id: str
    source_dataset: str  # "3DFront" | "infinigen" | "mp3d"
    room_type: str
    # Raw M3DLayout objects; location is (centre_x, height, centre_z), not a 3D position.
    objects: list[dict[str, Any]]
    style_tags: list[str] = []
    similarity_score: float = 0.0  # set by the retriever


class FunctionalZoneSpec(BaseModel):
    """A functional sub-region of a room (sleeping, seating, workspace, ...)."""

    model_config = _BASE_CFG

    zone_id: str
    label: str  # "sleeping" | "seating" | "workspace" | "dining" | …
    polygon_2d: list[tuple[float, float]]  # XZ floor polygon in metres, CW or CCW
    allowed_categories: list[str]
    category_counts: dict[str, int] = {}  # e.g. {"dining_chair": 4}; absent means one
    anchor_category: str | None = None
    diffusion_prompt: str | None = None  # LLM-authored prompt for per-zone diffusion generation
    # Room side of the zone's wall; lifting turns the image's wall onto it. None: interior zone.
    wall_orientation: str | None = None  # "north" | "south" | "east" | "west" | None


class RoomSpec(BaseModel):
    """One room within the scene."""

    model_config = _BASE_CFG

    room_id: str
    room_type: str
    bbox: BBox3D
    functional_zones: list[FunctionalZoneSpec]


class ZoningPlan(BaseModel):
    """Zoning plan produced by Phase 1: the rooms, each with its functional zones."""

    model_config = _BASE_CFG

    scene_spec: SceneSpec
    rooms: list[RoomSpec]
    ceiling_h: float = 2.7

    @model_validator(mode="after")
    def _at_least_one_room(self) -> "ZoningPlan":
        if not self.rooms:
            raise ValueError("ZoningPlan must contain at least one RoomSpec")
        return self


class MountType(str, Enum):
    FLOOR = "floor"
    WALL = "wall"
    SURFACE = "surface"  # placed on top of another object


class AssetRecord(BaseModel):
    """One row of ``data/assets.json``."""

    model_config = _BASE_CFG

    asset_id: str
    category: str
    bbox_size: tuple[float, float, float]  # full extents, metres
    mount_type: MountType = MountType.FLOOR
    support_surface: str | None = None  # SURFACE mount: the category it rests on
    allowed_room_types: list[str] = []  # surface and wall objects; empty = any
    allowed_zone_labels: list[str] = []  # zone labels of an anchor in the zoner's catalogue
    is_anchor: bool = False
    style_tags: list[str] = []

    @property
    def footprint_size(self) -> tuple[float, float]:
        """(x, z) floor footprint in metres."""
        return (self.bbox_size[0], self.bbox_size[2])


class PlacedObject(BaseModel):
    """An asset instance positioned in the scene."""

    model_config = _BASE_CFG

    instance_id: str
    asset_id: str
    category: str
    pose: BBox3D
    zone_id: str
    # lift3d | wall_proposer_vlm | wall_proposer | surface_manipuland
    placement_method: str = "lift3d"


class SceneLayout(BaseModel):
    """Complete 3-D scene layout produced by Phase 2."""

    model_config = _BASE_CFG

    layout_id: str
    zoning_plan: ZoningPlan
    placed_objects: list[PlacedObject]
