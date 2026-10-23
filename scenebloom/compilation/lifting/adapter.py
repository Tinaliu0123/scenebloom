"""Turn lifted camera-frame boxes into PlacedObjects in the room frame."""

from __future__ import annotations

import json
import logging
import math
import uuid
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from scenebloom.utils.paths import DATA
from scenebloom.utils.schemas import BBox3D, PlacedObject

logger = logging.getLogger(__name__)

_ROTATION_HYSTERESIS_M = 0.3  # AABB X-vs-Z gap below this is too noisy to flip 90°


def lift3d_to_placed_objects(
    objects_3d: list[dict[str, Any]],
    zone_id: str,
    room_x_m: float = 5.0,
    room_z_m: float = 6.0,
    floor_y: float = 0.0,
    zone_origin_x_m: float = 0.0,
    zone_origin_z_m: float = 0.0,
    category_to_mount_type: dict[str, str] | None = None,
    wall_orientation: str | None = None,
) -> list[PlacedObject]:
    """Convert lifted boxes (``bbox_min``, ``bbox_max``, ``name``, ``valid_3d``) to PlacedObjects.

    Args:
        room_x_m, room_z_m: width and depth (m) of the zone the image shows.
        zone_origin_x_m, zone_origin_z_m: room-frame position of that zone's min corner.
        category_to_mount_type: {category: "floor" | "wall" | "surface"}; default floor.
        wall_orientation: room side ("north", "south", "east", "west") of the wall the image faces.
    """
    mount_lookup = category_to_mount_type or {}
    # For east/west, the camera's horizontal axis aligns with zone-z.
    perp_extent = room_z_m if wall_orientation in ("east", "west") else room_x_m

    placed: list[PlacedObject] = []
    for obj in objects_3d:
        if not obj.get("valid_3d"):
            continue

        name = obj.get("name", "?")
        mount = mount_lookup.get(name, "floor")
        if mount == "surface":
            logger.info(
                "lift3d adapter: dropping surface-mount %r — deferred to the "
                "small-object stage.",
                name,
            )
            continue
        if mount != "floor":
            raise ValueError(
                f"lift3d adapter: {name!r} is {mount}-mount; only floor objects are lifted"
            )

        bmin = obj["bbox_min"]
        bmax = obj["bbox_max"]

        # Cam-space sizes (lift3d frame: x=right, y=up, z=depth-into-scene)
        cam_size_x = max(bmax[0] - bmin[0], 0.05)
        cam_size_y = max(bmax[1] - bmin[1], 0.05)
        cam_size_z = max(bmax[2] - bmin[2], 0.05)

        if wall_orientation in ("east", "west"):
            zone_size_x = cam_size_z
            zone_size_z = cam_size_x
        else:
            zone_size_x = cam_size_x
            zone_size_z = cam_size_z
        size, rotation_y = _resolve_size_and_yaw(
            obj,
            aabb_size=(zone_size_x, cam_size_y, zone_size_z),
            wall_orientation=wall_orientation,
        )

        # No per-object clamp: wall gluing and containment run later, per room.
        perp = (bmin[0] + bmax[0]) / 2 + perp_extent / 2
        depth = (bmin[2] + bmax[2]) / 2
        cx_local, cz_local = _orient_to_zone_local(
            perp, depth,
            room_x_m=room_x_m, room_z_m=room_z_m,
            wall_orientation=wall_orientation,
        )
        cx = cx_local + zone_origin_x_m
        cz = cz_local + zone_origin_z_m

        # The scene builder rests the mesh bottom at pose.center[1].
        cy = floor_y

        placed.append(PlacedObject(
            instance_id=str(uuid.uuid4())[:8],
            asset_id=obj.get("name", "unknown"),
            category=obj.get("name", "unknown"),
            pose=BBox3D(center=(cx, cy, cz), size=size, rotation_y=rotation_y),
            zone_id=zone_id,
            placement_method="lift3d",
        ))
    return placed


def _principal_axis_rotation_y(
    world_size_x: float,
    world_size_z: float,
) -> float:
    """pi/2 when the AABB is longer in x than in z by more than the hysteresis, else 0."""
    if world_size_x - world_size_z > _ROTATION_HYSTERESIS_M:
        return math.pi / 2
    return 0.0


# Rotation taking a zone's camera (perp, depth) plane onto room (x, z); the yaw offset
# and the branches of ``_orient_to_zone_local`` must both match it.
_ZONE_CARDINAL_RAD = {
    None: 0.0,
    "north": 0.0,
    "south": math.pi,
    "east": -math.pi / 2,
    "west": math.pi / 2,
}


def _zone_cardinal_rad(wall_orientation: str | None) -> float:
    """Cardinal angle for ``wall_orientation``; unknown walls count as north."""
    return _ZONE_CARDINAL_RAD.get(wall_orientation, 0.0)


# Camera-to-room yaw offset per wall, from the same angles as the position rotation.
_WALL_YAW_OFFSET = {
    k: _zone_cardinal_rad(k) for k in (None, "north", "south", "east", "west")
}


def _wrap_to_pi(angle: float) -> float:
    """Wrap a radian angle into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _room_yaw_from_cam(yaw_cam_rad: float, wall_orientation: str | None) -> float:
    """Map an OVMono3D camera-frame yaw into the room frame."""
    # OVMono3D yaw is pi/2, not 0, for a frontal object (long side facing the camera).
    offset = _WALL_YAW_OFFSET.get(wall_orientation, 0.0)
    frontal = math.pi / 2.0
    return _wrap_to_pi((float(yaw_cam_rad) - frontal) + offset)


def _resolve_size_and_yaw(
    obj: dict[str, Any],
    *,
    aabb_size: tuple[float, float, float],
    wall_orientation: str | None,
) -> tuple[tuple[float, float, float], float]:
    """(size, rotation_y) from predicted dims and yaw, else from the AABB and its principal axis."""
    yaw_cam = obj.get("yaw_cam_rad")
    dims = obj.get("dimensions_whd")
    if yaw_cam is not None and dims is not None and len(dims) == 3:
        # dimensions_whd is (width, height, depth), the same order as BBox3D.size
        sx, sy, sz = float(dims[0]), float(dims[1]), float(dims[2])
        size = (max(sx, 0.05), max(sy, 0.05), max(sz, 0.05))
        return size, _room_yaw_from_cam(float(yaw_cam), wall_orientation)
    rotation_y = _principal_axis_rotation_y(
        world_size_x=aabb_size[0],
        world_size_z=aabb_size[2],
    )
    return aabb_size, rotation_y


def _orient_to_zone_local(
    perp: float,
    depth: float,
    *,
    room_x_m: float,
    room_z_m: float,
    wall_orientation: str | None,
) -> tuple[float, float]:
    """Image (perp along the wall, depth toward it) to zone-local (cx, cz) for the wall side."""
    if wall_orientation == "south":
        return room_x_m - perp, room_z_m - depth
    if wall_orientation == "east":
        return depth, room_z_m - perp
    if wall_orientation == "west":
        return room_x_m - depth, perp
    # north or None (default)
    return perp, depth


# Built from the whole catalogue, so a category's mount type never depends on retrieval.
@lru_cache(maxsize=1)
def category_mount_table() -> Mapping[str, str]:
    """``{category: mount_type}`` from ``data/assets.json``, cached (the catalogue is static)."""
    rows = json.loads((DATA / "assets.json").read_text(encoding="utf-8"))
    table: dict[str, str] = {}
    for row in rows:
        table.setdefault(row["category"], row["mount_type"])
    return MappingProxyType(table)


def resolve_mount_table(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The catalogue mount table with ``overrides`` (records about to be placed) on top."""
    resolved = dict(category_mount_table())
    resolved.update(overrides or {})
    return resolved
