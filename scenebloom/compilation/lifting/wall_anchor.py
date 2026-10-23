"""Zone wall alignment after the lift: snap parallel, face into the room, glue to the wall."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = [
    "snap_zone_parallel",
    "glue_zone_to_wall",
    "face_zone_inward",
]


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def snap_zone_parallel(placed_objects, *, snap_step_rad: float = math.pi / 2.0):
    """Rigidly rotate the zone so its largest-footprint anchor snaps to the nearest cardinal yaw."""
    try:
        objs = list(placed_objects)
        if not objs:
            return placed_objects
        anchor = max(objs, key=lambda o: o.pose.size[0] * o.pose.size[2])
        cur = float(anchor.pose.rotation_y)
        target = round(cur / snap_step_rad) * snap_step_rad
        delta = _wrap_to_pi(target - cur)
        if abs(delta) < 1e-6:
            logger.info("wall_anchor: snap no-op (anchor=%s already aligned)",
                        anchor.category)
            return placed_objects
        px, _py, pz = anchor.pose.center
        c, s = math.cos(delta), math.sin(delta)
        out = []
        for o in objs:
            x, y, z = o.pose.center
            dx, dz = x - px, z - pz
            new_pose = o.pose.model_copy(update={
                "center": (px + c * dx - s * dz, y, pz + s * dx + c * dz),
                "rotation_y": _wrap_to_pi(float(o.pose.rotation_y) + delta),
            })
            out.append(o.model_copy(update={"pose": new_pose}))
        logger.info(
            "wall_anchor: SNAP zone by %+.1f° (anchor=%s %.1f°→%.1f°, %d objs)",
            math.degrees(delta), anchor.category,
            math.degrees(cur), math.degrees(_wrap_to_pi(target)), len(out),
        )
        return out
    except Exception as exc:
        logger.warning("wall_anchor: snap not applied (%s); placement unchanged", exc)
        return placed_objects


# Wall axis/direction per cardinal: (axis_index_in_center, +1 toward wall).
_GLUE_AXIS = {"north": (2, 1.0), "south": (2, -1.0), "east": (0, 1.0), "west": (0, -1.0)}


def _axis_half_extent(o, axis: int) -> float:
    """Rotation-aware half footprint extent of *o* along room axis 0 (x) / 2 (z)."""
    w, _h, d = o.pose.size
    cy, sy = abs(math.cos(o.pose.rotation_y)), abs(math.sin(o.pose.rotation_y))
    return ((w * cy + d * sy) / 2.0) if axis == 0 else ((w * sy + d * cy) / 2.0)


_INWARD_TARGET = {"south": math.pi, "north": 0.0,
                  "east": -math.pi / 2.0, "west": math.pi / 2.0}


def glue_zone_to_wall(placed_objects, *, wall_orientation: str | None,
                      wall_coord: float, margin: float = 0.05,
                      anchor_category: str | None = None):
    """Translate the zone along the wall normal so the anchor's back sits ``margin`` off the wall.

    ``wall_coord`` is the wall plane's room x (east/west) or z (north/south). Returns the new
    objects and the anchor's instance_id, which de-collision keeps fixed.
    """
    try:
        objs = list(placed_objects)
        axis_dir = _GLUE_AXIS.get(wall_orientation or "")
        if not objs or axis_dir is None:
            return placed_objects, None
        axis, dr = axis_dir

        def _back(o) -> float:
            return dr * o.pose.center[axis] + _axis_half_extent(o, axis)

        # Back the semantic anchor (e.g. sofa) to the wall; noisy depth can put a plant behind it.
        anchor = None
        if anchor_category:
            anchor = next((o for o in objs if o.category == anchor_category), None)
        if anchor is None:
            anchor = max(objs, key=lambda o: (_back(o), o.pose.size[0] * o.pose.size[2]))
        delta = (dr * wall_coord - margin) - _back(anchor)   # along +normal
        shift = dr * delta                                   # signed on the axis
        out = []
        for o in objs:
            c = list(o.pose.center)
            c[axis] += shift
            out.append(o.model_copy(update={
                "pose": o.pose.model_copy(update={"center": (c[0], c[1], c[2])})}))
        logger.info(
            "wall_anchor: GLUE zone to %s wall (anchor=%s, shift=%+.2fm, %d objs)",
            wall_orientation, anchor.category, shift, len(out),
        )
        return out, anchor.instance_id
    except Exception as exc:
        logger.warning("wall_anchor: glue not applied (%s); placement unchanged", exc)
        return placed_objects, None


def face_zone_inward(placed_objects, *, wall_orientation: str | None,
                     wall_coord: float, margin: float = 0.35,
                     anchor_category: str | None = None):
    """Turn a wall-backed zone to face into the room, by its anchor or per object near the wall."""
    try:
        objs = list(placed_objects)
        axis_dir = _GLUE_AXIS.get(wall_orientation or "")
        if not objs or axis_dir is None:
            return placed_objects
        axis, dr = axis_dir
        # Room-inward normal in the (x, z) plane = opposite the wall direction.
        inward = (-dr, 0.0) if axis == 0 else (0.0, -dr)

        # With an anchor: rotate the whole zone about its centre (run before glue).
        anchor = (next((o for o in objs if o.category == anchor_category), None)
                  if anchor_category else None)
        if anchor is not None and wall_orientation in _INWARD_TARGET:
            delta = _wrap_to_pi(_INWARD_TARGET[wall_orientation]
                                - float(anchor.pose.rotation_y))
            if abs(delta) < 1e-6:
                return placed_objects
            xs = [o.pose.center[0] for o in objs]
            zs = [o.pose.center[2] for o in objs]
            cx, cz = (min(xs) + max(xs)) / 2.0, (min(zs) + max(zs)) / 2.0
            c, s = math.cos(delta), math.sin(delta)
            out = []
            for o in objs:
                px, py, pz = o.pose.center
                dx, dz = px - cx, pz - cz
                nx, nz = cx + dx * c - dz * s, cz + dx * s + dz * c
                out.append(o.model_copy(update={"pose": o.pose.model_copy(update={
                    "center": (nx, py, nz),
                    "rotation_y": _wrap_to_pi(float(o.pose.rotation_y) + delta),
                })}))
            logger.info(
                "wall_anchor: FACE-INWARD rotated zone %+.0f° about centre "
                "(anchor=%s, %s wall, %d objs)",
                math.degrees(delta), anchor_category, wall_orientation, len(out),
            )
            return out

        # Without one: flip objects that back onto the wall but face it.
        out, flipped = [], []
        for o in objs:
            gap = dr * wall_coord - (dr * o.pose.center[axis]
                                     + _axis_half_extent(o, axis))
            yaw = o.pose.rotation_y
            # yaw 0 faces (0, -1) in (x, z), i.e. south
            front = (math.sin(yaw), -math.cos(yaw))
            if gap < margin and front[0] * inward[0] + front[1] * inward[1] < 0.0:
                out.append(o.model_copy(update={"pose": o.pose.model_copy(
                    update={"rotation_y": _wrap_to_pi(yaw + math.pi)})}))
                flipped.append(o.category)
            else:
                out.append(o)
        if flipped:
            logger.info(
                "wall_anchor: FACE-INWARD flipped %s on %s wall (%d objs)",
                flipped, wall_orientation, len(out),
            )
        return out
    except Exception as exc:
        logger.warning("wall_anchor: face-inward not applied (%s); placement unchanged", exc)
        return placed_objects
