"""Slot-fill relabel for the OVMono3D lift: fill planned-but-empty slots from discarded boxes."""
# Imported in the ovmono3d env (Python 3.8): stdlib only, typing-module annotations.
from __future__ import annotations

import math
from typing import Callable, Any, Dict, List, Optional, Tuple

#: accepted (short, long) ratio of a box's footprint to the slot category's catalogue footprint
SIZE_BAND = (0.6, 1.6)

#: Categories the detector confuses, detector nouns included; a label is only rewritten within its group.
CONFUSABLE_GROUPS = (
    ("sofa", "armchair", "chair", "dining_chair", "officechair"),
    ("cabinet", "wardrobe", "bookshelf", "chest_of_drawers",
     "tv_stand", "bathroom_washstand", "dressing_table", "dresser"),
    ("desk", "dining_table", "coffee_table", "side_table", "nightstand"),
    ("bed",),
)

_GROUP_OF = {}  # type: Dict[str, str]
for _members in CONFUSABLE_GROUPS:
    for _m in _members:
        _GROUP_OF[_m] = _members[0]


def group_of(category: str) -> Optional[str]:
    """Name of the confusable group ``category`` belongs to (None = none)."""
    return _GROUP_OF.get(category)


def sorted_footprint(x: float, z: float) -> Tuple[float, float]:
    """(short, long) sides of a footprint."""
    a, b = abs(float(x)), abs(float(z))
    return (a, b) if a <= b else (b, a)


def detection_footprint(det: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(short, long) floor footprint of a detection, from its dims or else its corners."""
    dims = det.get("dimensions_w_l_h")
    # dimensions_w_l_h is (width, height, depth) despite its name
    if dims and len(dims) >= 3:
        return sorted_footprint(dims[0], dims[2])
    corners = det.get("bbox3d_corners")
    if not corners:
        return None
    xs = [float(c[0]) for c in corners]
    zs = [float(c[2]) for c in corners]
    return sorted_footprint(max(xs) - min(xs), max(zs) - min(zs))


def within_band(
    fp: Tuple[float, float],
    ref: Tuple[float, float],
    band: Tuple[float, float] = SIZE_BAND,
) -> bool:
    lo, hi = band
    for got, want in zip(fp, ref):
        if want <= 0.0:
            return False
        r = got / want
        if r < lo or r > hi:
            return False
    return True


def _band_distance(fp: Tuple[float, float], ref: Tuple[float, float]) -> float:
    """How far from a perfect size match, in log-ratio (0 = exact)."""
    return sum(abs(math.log(max(g, 1e-6) / max(w, 1e-6))) for g, w in zip(fp, ref))


def fill_slots(
    dets: List[Dict[str, Any]],
    kept: "set",
    count_by_label: Dict[str, int],
    footprint_by_label: Optional[Dict[str, Tuple[float, float]]] = None,
    group_by_label: Optional[Dict[str, str]] = None,
    *,
    band: Tuple[float, float] = SIZE_BAND,
    allow: "Optional[Callable[[int, str], Optional[str]]]" = None,
    overlaps: "Optional[Callable[[int, int], bool]]" = None,
) -> Dict[int, str]:
    """Fill empty planned slots from discarded boxes; returns ``{index: new label}``.

    Greedy by score; a box fills a slot of its confusable group whose catalogue footprint it
    matches within ``band``. ``allow(i, label)`` returns a reason to block, else None.
    """
    requested = {str(k): int(v) for k, v in (count_by_label or {}).items()}
    kept_by_label = {}  # type: Dict[str, int]
    for i in kept:
        lbl = str(dets[i].get("label", "unknown"))
        kept_by_label[lbl] = kept_by_label.get(lbl, 0) + 1
    missing = {
        lbl: n - kept_by_label.get(lbl, 0)
        for lbl, n in requested.items()
        if n - kept_by_label.get(lbl, 0) > 0
    }

    relabels = {}  # type: Dict[int, str]
    refs = footprint_by_label or {}
    groups = group_by_label or {}
    if not (missing and refs and groups):
        return relabels
    leftover = [
        i for i, d in enumerate(dets)
        if i not in kept and d.get("bbox3d_corners")
        # a cross-label duplicate never comes back under a third label
        and "suppressed_by" not in d
    ]
    leftover.sort(key=lambda i: (-float(dets[i].get("score", 0.0)), i))
    for i in leftover:
        det = dets[i]
        own = str(det.get("label", "unknown"))
        own_group = groups.get(own)
        fp = detection_footprint(det)
        if own_group is None or fp is None:
            continue
        if overlaps is not None and any(overlaps(i, j) for j in relabels):
            continue
        best = None  # type: Optional[Tuple[float, str]]
        for lbl, n in missing.items():
            if n <= 0 or lbl == own or groups.get(lbl) != own_group:
                continue
            ref = refs.get(lbl)
            if ref is None or not within_band(fp, tuple(ref), band):
                continue
            if allow is not None and allow(i, lbl):
                continue
            key = (_band_distance(fp, tuple(ref)), lbl)
            if best is None or key < best:
                best = key
        if best is None:
            continue
        target = best[1]
        missing[target] -= 1
        relabels[i] = target
    return relabels
