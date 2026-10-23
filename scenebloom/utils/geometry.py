"""Box helpers for placed objects; world space is right-handed and Y-up, the floor is XZ."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenebloom.utils.schemas import PlacedObject


def y_extent(obj: "PlacedObject") -> tuple[float, float]:
    """``(y_min, y_max)`` of an object's box; ``center[1]`` is the base, not the centre."""
    cy, sy = obj.pose.center[1], obj.pose.size[1]
    return cy, cy + sy


def obb_extent_xz(obj: "PlacedObject") -> tuple[float, float]:
    """Half-spans (hx, hz) along the room axes of the object's yawed XZ footprint."""
    hx0, hz0 = obj.pose.size[0] / 2.0, obj.pose.size[2] / 2.0
    c = abs(math.cos(obj.pose.rotation_y))
    s = abs(math.sin(obj.pose.rotation_y))
    return hx0 * c + hz0 * s, hx0 * s + hz0 * c


def facing_dir(ry: float) -> tuple[float, float]:
    """Unit vector of an object's front in world XZ; yaw 0 faces -z."""
    return (math.sin(ry), -math.cos(ry))


def with_center(obj: "PlacedObject", center: tuple[float, float, float]) -> "PlacedObject":
    """Return a new PlacedObject translated to ``center``."""
    from scenebloom.utils.schemas import BBox3D

    new_pose = BBox3D(
        center=(float(center[0]), float(center[1]), float(center[2])),
        size=obj.pose.size,
        rotation_y=obj.pose.rotation_y,
    )
    return obj.model_copy(update={"pose": new_pose})
