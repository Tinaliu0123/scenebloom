"""OBB footprints in the XZ plane: corners, SAT overlap and minimum translation."""

from __future__ import annotations

import math

__all__ = [
    "rect_corners",
    "obb_overlap",
    "obb_mtv",
]

Pt = tuple[float, float]


def rect_corners(cx: float, cz: float, sx: float, sz: float, yaw: float) -> list[Pt]:
    """4 corners (XZ) of a ``sx×sz`` rectangle centred at (cx,cz), rotated ``yaw``."""
    hx, hz = sx / 2.0, sz / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (cx + c * dx - s * dz, cz + s * dx + c * dz)
        for dx, dz in ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))
    ]


def _edge_normals(corners: list[Pt]) -> list[Pt]:
    """Unit normals of a rectangle's first two edges."""
    axes: list[Pt] = []
    for i in (0, 1):
        ex = corners[i + 1][0] - corners[i][0]
        ez = corners[i + 1][1] - corners[i][1]
        n = (-ez, ex)
        L = math.hypot(n[0], n[1]) or 1.0
        axes.append((n[0] / L, n[1] / L))
    return axes


def _project(corners: list[Pt], axis: Pt) -> tuple[float, float]:
    ds = [c[0] * axis[0] + c[1] * axis[1] for c in corners]
    return min(ds), max(ds)


def obb_mtv(a: list[Pt], b: list[Pt], *, margin: float = 0.0) -> tuple[Pt, float] | None:
    """SAT translation ``((ux, uz), depth)`` moving ``b`` off ``a``, axis a -> b; None if clear."""
    best_axis: Pt | None = None
    best_depth = math.inf
    for axis in _edge_normals(a) + _edge_normals(b):
        amin, amax = _project(a, axis)
        bmin, bmax = _project(b, axis)
        overlap = min(amax, bmax) - max(amin, bmin) + margin
        if overlap <= 0:
            return None
        if overlap < best_depth:
            best_depth = overlap
            best_axis = axis
    if best_axis is None:
        return None
    ca = (sum(p[0] for p in a) / 4.0, sum(p[1] for p in a) / 4.0)
    cb = (sum(p[0] for p in b) / 4.0, sum(p[1] for p in b) / 4.0)
    if (cb[0] - ca[0]) * best_axis[0] + (cb[1] - ca[1]) * best_axis[1] < 0:
        best_axis = (-best_axis[0], -best_axis[1])
    return best_axis, best_depth


def obb_overlap(a: list[Pt], b: list[Pt], *, margin: float = 0.0) -> bool:
    """True if the two rectangles overlap, with optional clearance ``margin``."""
    return obb_mtv(a, b, margin=margin) is not None
