"""Small-object stage: place manipulands on the tops of supporting furniture."""

from __future__ import annotations

import hashlib
from typing import Callable

from scenebloom.utils.schemas import AssetRecord, SceneLayout
from scenebloom.utils.assets import room_allows
from scenebloom.compilation.completion.surface_placer import (
    SmallObjectProposer,
    get_support_rect,
    place_manipulands,
)


#: Height of the band above a support top that a manipuland occupies (metres).
PREOCCUPY_HEADROOM_M = 0.45


def _footprints_at_height(layout, support, y_top: float, tol: float = 1e-3):
    """XZ footprints of other objects within ``PREOCCUPY_HEADROOM_M`` above the support top."""
    from scenebloom.utils.geometry import y_extent
    from scenebloom.compilation.fitting.obb import rect_corners

    head = PREOCCUPY_HEADROOM_M
    out = []
    for other in layout.placed_objects:
        if other.instance_id == support.instance_id:
            continue
        low, high = y_extent(other)
        if low <= y_top + tol + head and high >= y_top - tol:
            cx, _, cz = other.pose.center
            sx, _, sz = other.pose.size
            out.append(rect_corners(cx, cz, sx, sz, float(other.pose.rotation_y)))
    return out


#: Human reach height (metres); a higher support top holds no small objects.
SUPPORT_MAX_TOP_Y = 1.4

DEFAULT_SUPPORTS = (
    "desk",
    "coffee_table",
    "nightstand",
    "side_table",
    "dining_table",
    "cabinet",
    "tv_stand",
    "chest_of_drawers",
    "dressing_table",
)


def _room_ok(allowed: list[str], room_type: str) -> bool:
    return room_allows(allowed, room_type)


def _pick_variant(candidates: list[AssetRecord], *, seed_key: str) -> AssetRecord:
    """Deterministic choice among one category's variants, seeded by ``seed_key``."""
    ordered = sorted(candidates, key=lambda a: a.asset_id)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def assets_for_support(
    surface_assets: list[AssetRecord],
    support_category: str,
    room_type: str = "",
    *,
    variant_seed: str | None = None,
) -> list[AssetRecord]:
    """Assets fitting the support and room; one variant per category if ``variant_seed`` is set."""
    compatible = [
        asset
        for asset in surface_assets
        if asset.support_surface in (None, "", support_category)
        and _room_ok(asset.allowed_room_types, room_type)
    ]
    if variant_seed is None:
        return compatible

    by_category: dict[str, list[AssetRecord]] = {}
    order: list[str] = []
    for asset in compatible:
        if asset.category not in by_category:
            by_category[asset.category] = []
            order.append(asset.category)  # input order is the proposer's priority
        by_category[asset.category].append(asset)
    return [
        _pick_variant(by_category[category], seed_key=f"{variant_seed}|{category}")
        for category in order
    ]


def place_small_objects(
    layout: SceneLayout,
    surface_assets: list[AssetRecord],
    *,
    room_type: str,
    supports: tuple[str, ...] = DEFAULT_SUPPORTS,
    proposer_factory: Callable[[int], SmallObjectProposer],
    max_candidates: int = 20,
    retry_rounds: int = 0,
) -> SceneLayout:
    """Append surface objects without mutating the furniture-only layout."""

    supports_set = {support.lower() for support in supports}
    max_top_y = SUPPORT_MAX_TOP_Y
    all_categories = [obj.category for obj in layout.placed_objects]
    new_objects = []

    for obj in layout.placed_objects:
        if obj.category.lower() not in supports_set:
            continue
        if get_support_rect(obj).y_top > max_top_y:
            continue
        compatible = assets_for_support(
            surface_assets, obj.category, room_type,
            variant_seed=f"{layout.layout_id}|{obj.instance_id}",
        )
        if not compatible:
            continue
        nearby = [category for category in all_categories if category != obj.category]
        result = place_manipulands(
            obj,
            proposer_factory(max_candidates),
            compatible,
            room_type=room_type,
            nearby_categories=nearby,
            retry_rounds=retry_rounds,
            # Includes manipulands placed on earlier supports in this loop.
            preoccupied=_footprints_at_height(
                layout.model_copy(
                    update={"placed_objects": [*layout.placed_objects, *new_objects]}
                ),
                obj,
                get_support_rect(obj).y_top,
            ),
        )
        new_objects.extend(result.placed)

    return layout.model_copy(
        update={
            "layout_id": f"{layout.layout_id}__smallobj",
            "placed_objects": [*layout.placed_objects, *new_objects],
        }
    )
