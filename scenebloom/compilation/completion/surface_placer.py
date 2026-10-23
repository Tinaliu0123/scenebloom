"""Small-object (manipuland) placement on furniture tops, after SceneSmith's place_manipuland."""
# Floor plane XZ, up +Y, yaw = ``rotation_y`` about +Y.
# ``BBox3D.center_y`` is the object's base, so a support's top is ``center_y + size_y``.
# Placements are support-local ``(u, v) in [-1, 1]`` plus a local yaw; world poses are derived.

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from scenebloom.utils.schemas import AssetRecord, BBox3D, PlacedObject

from scenebloom.compilation.fitting.obb import (
    Pt,
    obb_overlap,
    rect_corners,
)

__all__ = [
    "SupportRect",
    "SupportContext",
    "SmallObjectProposal",
    "SmallObjectPlacement",
    "PlacementResult",
    "SmallObjectProposer",
    "LLMProposer",
    "get_support_rect",
    "local_to_world",
    "validate_and_settle",
    "place_manipulands",
]

_DEFAULT_MARGIN_M = 0.02  # inset from the support edge, metres
_SETTLE_EPS_M = 0.005  # spawn gap above the surface, metres


def _rotate_xz(x: float, z: float, ang: float) -> tuple[float, float]:
    """Rotate a vector in the XZ plane by ``ang`` radians about +Y (local->world)."""
    c, s = math.cos(ang), math.sin(ang)
    return (x * c - z * s, x * s + z * c)


@dataclass(frozen=True)
class SupportRect:
    """Top rectangle of a furniture OBB in world space; half extents along its local x/z."""

    support_id: str
    cx: float
    cz: float
    y_top: float
    half_w: float
    half_d: float
    yaw: float


def get_support_rect(furniture: PlacedObject) -> SupportRect:
    """Top support rectangle of a placed furniture object (no mesh needed)."""
    cx, cy, cz = furniture.pose.center
    sx, sy, sz = furniture.pose.size
    return SupportRect(
        support_id=furniture.instance_id,
        cx=cx,
        cz=cz,
        y_top=cy + sy,
        half_w=sx / 2.0,
        half_d=sz / 2.0,
        yaw=furniture.pose.rotation_y,
    )


def local_to_world(
    support: SupportRect,
    u: float,
    v: float,
    yaw: float,
    *,
    eps: float = _SETTLE_EPS_M,
) -> tuple[tuple[float, float, float], float]:
    """Map support-local ``(u, v, yaw)`` to a world ``(center_xyz, rotation_y)``.

    ``center_y`` is the object's base, ``eps`` above the support top (no half-height offset).
    """
    lx = u * support.half_w
    lz = v * support.half_d
    dx, dz = _rotate_xz(lx, lz, support.yaw)
    center = (
        support.cx + dx,
        support.y_top + eps,
        support.cz + dz,
    )
    return center, support.yaw + yaw


def _world_footprint(
    support: SupportRect, u: float, v: float, yaw: float, asset: AssetRecord
) -> list[Pt]:
    """World XZ corners of a placement's footprint."""
    (wx, _wy, wz), wyaw = local_to_world(support, u, v, yaw)
    fx, fz = asset.footprint_size
    return rect_corners(wx, wz, fx, fz, wyaw)


@dataclass(frozen=True)
class SupportContext:
    """What a proposer sees for one surface; ``occupied_uv`` holds already accepted ``(u, v)``."""

    room_type: str
    support_category: str
    support: SupportRect
    nearby_categories: tuple[str, ...] = ()
    available_assets: tuple[AssetRecord, ...] = ()
    occupied_uv: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SmallObjectProposal:
    """A proposer's request: put ``asset_id`` at support-local ``(u, v, yaw)``."""

    asset_id: str
    u: float
    v: float
    yaw: float = 0.0


@dataclass(frozen=True)
class SmallObjectPlacement:
    """Accepted placement in support-local coordinates; the world pose is derived."""

    asset_id: str
    u: float
    v: float
    yaw: float  # support-local, radians


@dataclass(frozen=True)
class PlacementResult:
    """Outcome of one support surface: instantiated objects + their placements."""

    placed: list[PlacedObject] = field(default_factory=list)
    placements: list[SmallObjectPlacement] = field(default_factory=list)


class SmallObjectProposer(Protocol):
    """Returns support-local placements for a single surface. Pure/one-shot."""

    def propose(self, ctx: SupportContext) -> list[SmallObjectProposal]: ...


@dataclass(frozen=True)
class LLMProposer:
    """One-shot VLM proposer; ``llm_fn`` maps a prompt to a JSON list of ``(u, v, yaw)`` entries."""

    llm_fn: Callable[[str], str]
    # Safety cap on parsed entries; the VLM decides how many objects to place.
    max_candidates: int = 20
    yaw_in_degrees: bool = True

    def build_prompt(self, ctx: SupportContext) -> str:
        assets = "\n".join(
            f"  {a.asset_id} ({a.category}, {a.footprint_size[0]:.2f} x "
            f"{a.footprint_size[1]:.2f} m)"
            for a in ctx.available_assets
        )
        nearby = ", ".join(ctx.nearby_categories) or "none"
        w, d = ctx.support.half_w * 2, ctx.support.half_d * 2
        occupied = ""
        if ctx.occupied_uv:
            taken = ", ".join(f"({u:.2f},{v:.2f})" for u, v in ctx.occupied_uv)
            occupied = (
                f"Spots already taken on this surface : {taken}\n"
                "Do not overlap them; pick clearly separated positions.\n"
            )
        return (
            "You are arranging small objects on top of one piece of furniture.\n"
            "\n"
            "Position is expressed in surface local coordinates, never in world "
            "coordinates:\n"
            "  u  runs across the surface from one edge (-1) to the other (+1)\n"
            "  v  runs from the near edge (-1) to the far edge (+1)\n"
            "u = 0, v = 0 is the centre of the surface. yaw is in degrees about "
            "the vertical axis.\n"
            "\n"
            f"Room type          : {ctx.room_type}\n"
            f"Supporting surface : {ctx.support_category}, "
            f"{w:.2f} m by {d:.2f} m\n"
            f"Nearby context     : {nearby}\n"
            f"{occupied}"
            f"Objects you may place (each with its footprint):\n{assets}\n"
            "\n"
            "Choose a plausible subset and arrange it. Space the objects so their "
            "footprints do not overlap and so nothing hangs off the edge. You need "
            "not use every object.\n"
            "Reply with ONLY a JSON list, one entry per object placed:\n"
            '  [{"asset": "<asset_id>", "u": <number>, "v": <number>, '
            '"yaw": <degrees>}, ...]'
        )

    def propose(self, ctx: SupportContext) -> list[SmallObjectProposal]:
        raw = self.llm_fn(self.build_prompt(ctx))
        valid_ids = {a.asset_id for a in ctx.available_assets}
        out: list[SmallObjectProposal] = []
        for item in _parse_proposal_json(raw)[: self.max_candidates]:
            asset_id = str(item.get("asset", ""))
            if asset_id not in valid_ids:
                continue
            try:
                u = _clamp(float(item["u"]), -1.0, 1.0)
                v = _clamp(float(item["v"]), -1.0, 1.0)
                yaw = float(item.get("yaw", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if self.yaw_in_degrees:
                yaw = math.radians(yaw)
            out.append(SmallObjectProposal(asset_id=asset_id, u=u, v=v, yaw=yaw))
        return out


def _parse_proposal_json(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, dict):  # tolerate {"objects": [...]}
        data = data.get("objects") or data.get("placements") or []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _within_bounds(
    support: SupportRect, u: float, v: float, yaw: float, asset: AssetRecord, margin: float
) -> bool:
    """All four footprint corners inside the support's inset rectangle, in the support frame."""
    fx, fz = asset.footprint_size
    corners = rect_corners(u * support.half_w, v * support.half_d, fx, fz, yaw)
    return all(
        abs(cx) <= support.half_w - margin and abs(cz) <= support.half_d - margin
        for cx, cz in corners
    )


def validate_and_settle(
    proposals: Sequence[SmallObjectProposal],
    support: SupportRect,
    assets_by_id: dict[str, AssetRecord],
    *,
    zone_id: str,
    occupied: Sequence[list[Pt]] = (),
    margin: float = _DEFAULT_MARGIN_M,
) -> PlacementResult:
    """Greedily accept proposals inside the surface that do not overlap earlier footprints.

    ``occupied`` seeds the overlap test with world XZ corner lists already on the surface.
    """
    result = PlacementResult()
    taken: list[list[Pt]] = [list(c) for c in occupied]
    for i, p in enumerate(proposals):
        asset = assets_by_id.get(p.asset_id)
        if asset is None:
            continue
        (wx, wy, wz), wyaw = local_to_world(support, p.u, p.v, p.yaw)
        if not _within_bounds(support, p.u, p.v, p.yaw, asset, margin):
            continue
        corners = _world_footprint(support, p.u, p.v, p.yaw, asset)
        if any(obb_overlap(corners, oc) for oc in taken):
            continue
        # Stability proxy: the bounds check already put the footprint inside the inset rect.
        taken.append(corners)
        result.placements.append(
            SmallObjectPlacement(asset_id=p.asset_id, u=p.u, v=p.v, yaw=p.yaw)
        )
        result.placed.append(
            PlacedObject(
                instance_id=f"{support.support_id}__{asset.asset_id}__{i}",
                asset_id=asset.asset_id,
                category=asset.category,
                pose=BBox3D(center=(wx, wy, wz), size=asset.bbox_size, rotation_y=wyaw),
                zone_id=zone_id,
                placement_method="surface_manipuland",
            )
        )
    return result


def _placement_corners(
    support: SupportRect, placement: SmallObjectPlacement, asset: AssetRecord
) -> list[Pt]:
    """World XZ corners of an accepted placement."""
    return _world_footprint(
        support, placement.u, placement.v, placement.yaw, asset
    )


def place_manipulands(
    support_furniture: PlacedObject,
    proposer: SmallObjectProposer,
    available_assets: Sequence[AssetRecord],
    *,
    room_type: str,
    nearby_categories: Sequence[str] = (),
    zone_id: str | None = None,
    margin: float = _DEFAULT_MARGIN_M,
    retry_rounds: int = 0,
    preoccupied: Sequence[list[Pt]] = (),
) -> PlacementResult:
    """Propose, map to world and validate the small objects for one support surface.

    Each of the ``retry_rounds`` extra queries asks for still-unplaced assets with accepted
    ``(u, v)`` fed back; ``preoccupied`` holds footprints of furniture already on the surface.
    """
    support = get_support_rect(support_furniture)
    assets_by_id = {a.asset_id: a for a in available_assets}
    zone = zone_id if zone_id is not None else support_furniture.zone_id

    placed_ids: set[str] = set()
    accepted: list[SmallObjectPlacement] = []
    result = PlacementResult()
    for _ in range(retry_rounds + 1):
        remaining = [a for a in available_assets if a.asset_id not in placed_ids]
        if not remaining:
            break
        ctx = SupportContext(
            room_type=room_type,
            support_category=support_furniture.category,
            support=support,
            nearby_categories=tuple(nearby_categories),
            available_assets=tuple(remaining),
            occupied_uv=tuple((p.u, p.v) for p in accepted),
        )
        proposals = proposer.propose(ctx)
        occupied_world = (
            *(list(c) for c in preoccupied),
            *(_placement_corners(support, p, assets_by_id[p.asset_id])
              for p in accepted),
        )
        res = validate_and_settle(
            proposals,
            support,
            assets_by_id,
            zone_id=zone,
            occupied=occupied_world,
            margin=margin,
        )
        for pl, po in zip(res.placements, res.placed):
            placed_ids.add(pl.asset_id)
            accepted.append(pl)
            result.placements.append(pl)
            result.placed.append(po)
        if not res.placements:
            break
    return result
