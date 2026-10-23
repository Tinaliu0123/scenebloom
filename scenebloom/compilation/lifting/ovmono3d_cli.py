"""OVMono3D lift backend: per-zone 3D boxes from the generated zone images."""
# --spec: {"config_file", "weights", "depthpro_ckpt", "threshold", "zones": [{"zone_id",
#   "images", "labels", "anchor_label"?, "{count,size,footprint,group}_by_label"?}]}
# OVMono3D predicts in the Omni3D camera frame: x right, y down, z forward (metres).
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from scenebloom.compilation.lifting.slot_fill import fill_slots


def _load_depthpro(ckpt: Path):
    import dataclasses

    import torch
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    if not ckpt.is_file():
        raise SystemExit(f"depth_pro checkpoint not found: {ckpt}")
    config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=device, precision=torch.float16
    )
    model.eval()
    return model, transform


def _depthpro_npz(model, transform, image_path: Path, out_npz: Path) -> None:
    """Estimate metric depth + K for one image and write the NPZ ``ovmono3d_infer`` reads."""
    import numpy as np
    import torch
    import depth_pro

    image, _, f_px = depth_pro.load_rgb(str(image_path))
    image_t = transform(image)
    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().astype(np.float32)
    focal = float(prediction.get("focallength_px", f_px))
    h, w = depth.shape[:2]
    K = np.array(
        [[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez(out_npz, depth=depth, K=K)


def _build_args(
    *,
    input_folder: Path,
    labels_file: Path,
    threshold: float,
    weights: str,
    output_dir: Path,
    config_file: str,
) -> argparse.Namespace:
    """Construct the argparse.Namespace that ovmono3d_infer's _setup / _do_inference expect."""
    return argparse.Namespace(
        config_file=config_file,
        input_folder=str(input_folder),
        labels_file=str(labels_file),
        focal_length=0,
        principal_point=[],
        threshold=float(threshold),
        output_json=str(output_dir / "_unused.json"),
        eval_only=True,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:12345",
        opts=[
            "MODEL.ROI_HEADS.NAME", "ROIHeads3DGDINO",
            "MODEL.WEIGHTS", weights,
            "INPUT.USE_DEPTH", "True",
            "OUTPUT_DIR", str(output_dir),
        ],
    )


def _ransac_line(pts, n_iter: int = 250, dist_thresh: float = 0.15):
    """RANSAC a 2D line through (N,2) points; returns ((p0, d), inlier_mask)."""
    import numpy as np

    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(0)
    n = len(pts)
    if n < 2:
        return None, None
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        d = pts[j] - pts[i]
        norm = float(np.hypot(*d))
        if norm < 1e-6:
            continue
        d = d / norm
        normal = np.array([-d[1], d[0]])
        dist = np.abs((pts - pts[i]) @ normal)
        inliers = dist < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return None, None
    # Refit on inliers (total least squares via SVD).
    sel = pts[best_inliers]
    centroid = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - centroid)
    d = vt[0]
    normal = np.array([-d[1], d[0]])
    dist = np.abs((pts - centroid) @ normal)
    inliers = dist < dist_thresh
    return (centroid, d), inliers


def _estimate_walls(depth, K, stride: int = 8) -> dict[str, Any]:
    """Wall layout from depth: RANSAC lines through the farthest above-floor point per column."""
    import numpy as np

    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    z = depth[vs, us].astype(np.float64)
    valid = z > 0.5
    if valid.sum() < 100:
        return {"mode": "none", "walls": []}
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z  # y-down: floor = large positive y
    y_floor = np.percentile(y[valid], 95)
    band = valid & (y < y_floor - 0.4) & (y > y_floor - 2.6)

    pts = []
    for col in range(us.shape[1]):
        m = band[:, col]
        if not m.any():
            continue
        zi = z[:, col][m]
        xi = x[:, col][m]
        k = int(np.argmax(zi))
        pts.append((float(xi[k]), float(zi[k])))
    if len(pts) < 20:
        return {"mode": "none", "walls": []}
    pts = np.asarray(pts)

    walls = []
    remaining = pts
    for _ in range(2):
        if len(remaining) < 15:
            break
        line, inliers = _ransac_line(remaining)
        if line is None:
            break
        frac = float(inliers.mean())
        if frac < 0.25 or int(inliers.sum()) < 12:
            break
        centroid, d = line
        normal = np.array([-d[1], d[0]])
        walls.append({
            "distance_m": float(abs(normal @ centroid)),
            "inlier_frac": round(frac, 3),
            "line_d": [float(d[0]), float(d[1])],
        })
        remaining = remaining[~inliers]

    if not walls:
        return {"mode": "none", "walls": []}
    if len(walls) == 2:
        d0 = np.asarray(walls[0]["line_d"])
        d1 = np.asarray(walls[1]["line_d"])
        ang = np.degrees(np.arccos(np.clip(abs(float(d0 @ d1)), 0.0, 1.0)))
        if ang >= 60.0:
            mode = "corner"
        else:
            # Two near-parallel lines = one wall + clutter; keep the dominant.
            mode = "frontal"
            walls = walls[:1]
    else:
        mode = "frontal"
    return {"mode": mode, "walls": walls}


#: footprint IoU above which two boxes of different labels can be one object
CROSS_LABEL_IOU = 0.5
#: ... if their heights also overlap by this fraction of the shorter box
CROSS_LABEL_Y_OVERLAP = 0.5


def _corner_aabb(det: dict[str, Any]) -> tuple[float, ...] | None:
    corners = det.get("bbox3d_corners") or []
    if not corners:
        return None
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    zs = [float(c[2]) for c in corners]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _footprint_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iz = max(0.0, min(a[5], b[5]) - max(a[4], b[4]))
    inter = ix * iz
    union = (a[1] - a[0]) * (a[5] - a[4]) + (b[1] - b[0]) * (b[5] - b[4]) - inter
    return inter / union if union > 0.0 else 0.0


def _y_overlap_frac(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ov = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    h = min(a[3] - a[2], b[3] - b[2])
    return ov / h if h > 0.0 else 0.0


#: share of the lower-score footprint inside the kept footprint
CROSS_LABEL_NESTED_IOS = 0.6
#: share of the lower-score box's camera-x span covered by the kept box
CROSS_LABEL_NESTED_COLUMN = 0.8
#: a part's footprint is at most this times the kept box's footprint
CROSS_LABEL_NESTED_AREA_SLACK = 1.05
#: max combined height (m) of a part stacked above or below the kept box
CROSS_LABEL_NESTED_MAX_HEIGHT = 2.3


def _footprint_area(a: tuple[float, ...]) -> float:
    return max(0.0, a[1] - a[0]) * max(0.0, a[5] - a[4])


def _footprint_inter(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iz = max(0.0, min(a[5], b[5]) - max(a[4], b[4]))
    return ix * iz


def _column_cover_frac(hi: tuple[float, ...], lo: tuple[float, ...]) -> float:
    """Share of ``lo``'s camera-x span that ``hi``'s span covers."""
    span = lo[1] - lo[0]
    if span <= 0.0:
        return 0.0
    return max(0.0, min(hi[1], lo[1]) - max(hi[0], lo[0])) / span


def _nested_duplicate(hi: tuple[float, ...], lo: tuple[float, ...]) -> bool:
    """``lo`` (the lower-score box) is a part of ``hi`` (the kept box)."""
    a_lo = _footprint_area(lo)
    a_hi = _footprint_area(hi)
    if a_lo <= 0.0 or a_hi <= 0.0:
        return False
    if _footprint_inter(hi, lo) / a_lo < CROSS_LABEL_NESTED_IOS:
        return False
    if _column_cover_frac(hi, lo) < CROSS_LABEL_NESTED_COLUMN:
        return False
    if a_lo > a_hi * CROSS_LABEL_NESTED_AREA_SLACK:
        return False
    union_h = max(hi[3], lo[3]) - min(hi[2], lo[2])
    return (_y_overlap_frac(hi, lo) >= CROSS_LABEL_Y_OVERLAP
            or union_h <= CROSS_LABEL_NESTED_MAX_HEIGHT)


#: share of the lower-score box's projected area inside the kept box's projection
CROSS_LABEL_NESTED_2D_IOS = 0.7


def _project_bbox2d(det: dict[str, Any], K: Any) -> list[float] | None:
    """``[u0, v0, u1, v1]`` of the 3D box's corners projected with ``K``."""
    corners = det.get("bbox3d_corners") or []
    if not corners or K is None:
        return None
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    us: list[float] = []
    vs: list[float] = []
    for x, y, z in corners:
        z = max(float(z), 0.05)
        us.append(fx * float(x) / z + cx)
        vs.append(fy * float(y) / z + cy)
    return [min(us), min(vs), max(us), max(vs)]


def _nested_duplicate_2d(hi: dict[str, Any], lo: dict[str, Any]) -> bool:
    """``lo`` (lower score) is a part of ``hi`` in the image."""
    a, b = hi.get("bbox2d_projected"), lo.get("bbox2d_projected")
    if not a or not b:
        return False
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    if area_b <= 0.0 or area_a <= 0.0 or area_b > area_a * CROSS_LABEL_NESTED_AREA_SLACK:
        return False
    inter = (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
             * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))
    return inter / area_b >= CROSS_LABEL_NESTED_2D_IOS


def _nested_leftovers(
    dets: list[dict[str, Any]], kept: set[int] | list[int]
) -> dict[int, int]:
    """``{leftover: kept}`` for discarded boxes nested in a kept box of the same label or group."""
    kept_idx = [i for i in kept if dets[i].get("bbox3d_corners")]
    out: dict[int, int] = {}
    for j, det in enumerate(dets):
        if j in kept or "suppressed_by" in det or not det.get("bbox3d_corners"):
            continue
        lo = _corner_aabb(det)
        gj = _label_group(det.get("label"))
        for i in sorted(kept_idx, key=lambda k: -float(dets[k].get("score", 0.0))):
            same = (dets[i].get("label") == det.get("label")
                    or (gj is not None and gj == _label_group(dets[i].get("label"))))
            if not same:
                continue
            if _nested_duplicate(_corner_aabb(dets[i]), lo) or _nested_duplicate_2d(dets[i], det):
                out[j] = i
                break
    return out


def _label_group(label: Any) -> str | None:
    """Confusable group of a detector label, trying "tv_stand", "tvstand" and "tv stand"."""
    from scenebloom.compilation.lifting.slot_fill import group_of
    text = str(label or "").strip().lower()
    for cand in (text.replace(" ", "_"), text.replace(" ", ""), text):
        g = group_of(cand)
        if g is not None:
            return g
    return None


def cross_label_suppressed(
    dets: list[dict[str, Any]],
    idxs: list[int] | None = None,
) -> dict[int, int]:
    """Cross-label duplicates within a confusable group: ``{suppressed: higher-score index}``."""
    cand = [i for i in (idxs if idxs is not None else range(len(dets)))
            if dets[i].get("bbox3d_corners")]
    cand.sort(key=lambda i: (-float(dets[i].get("score", 0.0)), i))
    boxes = {i: _corner_aabb(dets[i]) for i in cand}
    out: dict[int, int] = {}
    for pos, i in enumerate(cand):
        if i in out:
            continue
        for j in cand[pos + 1:]:
            if j in out or dets[i].get("label") == dets[j].get("label"):
                continue
            gi = _label_group(dets[i].get("label"))
            if gi is None or gi != _label_group(dets[j].get("label")):
                continue
            hi, lo = boxes[i], boxes[j]   # cand is sorted by score: i >= j
            if ((_footprint_iou(hi, lo) > CROSS_LABEL_IOU
                    and _y_overlap_frac(hi, lo) >= CROSS_LABEL_Y_OVERLAP)
                    or _nested_duplicate(hi, lo)
                    or _nested_duplicate_2d(dets[i], dets[j])):
                out[j] = i
    return out


#: Exported boxes behind a frontal back wall (thin objects in noisy depth) slide toward the
#: camera onto the wall plane; the wall needs this RANSAC inlier fraction to be trusted.
WALL_DEPTH_CLAMP_MIN_INLIER = 0.5
WALL_DEPTH_CLAMP_MIN_EXCESS = 0.02  # metres behind the plane before we act


def _frontal_wall_depth(wall_estimate: dict[str, Any] | None) -> float | None:
    """Camera-frame depth of the trusted frontal back wall, else ``None``."""
    if not wall_estimate or wall_estimate.get("mode") != "frontal":
        return None
    walls = wall_estimate.get("walls") or []
    if not walls:
        return None
    w = walls[0]
    if float(w.get("inlier_frac", 0.0)) < WALL_DEPTH_CLAMP_MIN_INLIER:
        return None
    return float(w["distance_m"])


#: A box shorter than this times the label's catalogue height is a part of another object
#: (a bookshelf's shelf as "cabinet") and is not a candidate.
HEIGHT_MIN_RATIO = 0.45


def _height_ratio(det: dict[str, Any], ref: Any) -> float | None:
    """box height / reference height, or None when either is unavailable."""
    dims = det.get("dimensions_w_l_h")
    if not dims or len(dims) < 3 or ref is None or len(ref) < 3:
        return None
    ref_h = float(ref[2])
    if ref_h <= 0.0:
        return None
    return float(dims[1]) / ref_h


#: keep order is score / (1 + w * logdist); it reorders one label's boxes, never drops one
SIZE_TIEBREAK_WEIGHT = 1.0


def _size_logdist(det: dict[str, Any], ref: Any) -> float:
    """Summed |log ratio| of (short, long, height) vs the catalogue; 0 without a reference."""
    dims = det.get("dimensions_w_l_h")
    if not ref or not dims or len(dims) < 3:
        return 0.0
    # dimensions_w_l_h is (width, height, depth) despite its name
    w, h, d = abs(float(dims[0])), abs(float(dims[1])), abs(float(dims[2]))
    short, long_ = (w, d) if w <= d else (d, w)
    total = 0.0
    for got, want in ((short, ref[0]), (long_, ref[1]), (h, ref[2])):
        if want <= 0.0 or got <= 0.0:
            continue
        total += abs(math.log(got / want))
    return total


def _keep_rank(det: dict[str, Any], ref: Any) -> float:
    """Score discounted by size mismatch, so a part box (a drawer) does not outrank the unit."""
    score = float(det.get("score", 0.0))
    return score / (1.0 + SIZE_TIEBREAK_WEIGHT * _size_logdist(det, ref))


def _kept_indices(
    dets: list[dict[str, Any]],
    count_by_label: dict[str, int] | None = None,
    size_by_label: dict[str, Any] | None = None,
) -> set[int]:
    """Dedup: the top ``count_by_label[label]`` boxes (default 1) per label."""
    with_corners = [i for i, d in enumerate(dets) if d.get("bbox3d_corners")]
    # Cross-label suppression first, so a duplicate never takes its own label's slot.
    suppressed = cross_label_suppressed(dets, with_corners)
    with_corners = [i for i in with_corners if i not in suppressed]
    counts = count_by_label or {}
    by_label: dict[str, list[int]] = {}
    for i in with_corners:
        by_label.setdefault(dets[i].get("label", "unknown"), []).append(i)
    kept: set[int] = set()
    sizes = size_by_label or {}
    for label, idxs in by_label.items():
        ref = sizes.get(label)
        if ref is not None:
            eligible = []
            for i in idxs:
                r = _height_ratio(dets[i], ref)
                if r is None or r >= HEIGHT_MIN_RATIO:
                    eligible.append(i)
            idxs = eligible
        idxs.sort(key=lambda i: (-_keep_rank(dets[i], ref), i))
        keep_n = max(1, int(counts.get(label, 1)))
        kept.update(idxs[:keep_n])
    return kept


def _omni3d_to_objects(
    dets: list[dict[str, Any]],
    count_by_label: dict[str, int] | None = None,
    relabels: dict[int, str] | None = None,
    size_by_label: dict[str, Any] | None = None,
    wall_estimate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Kept and relabelled detections as adapter objects: y flipped up, AABB of the corners."""
    kept = _kept_indices(dets, count_by_label, size_by_label)
    relabels = relabels or {}
    wall_z = _frontal_wall_depth(wall_estimate)
    out: list[dict[str, Any]] = []
    for i, d in enumerate(dets):
        if i not in kept and i not in relabels:
            continue
        corners = d["bbox3d_corners"]
        xs = [c[0] for c in corners]
        ys = [-c[1] for c in corners]   # y-down -> y-up
        zs = [c[2] for c in corners]
        if wall_z is not None and max(zs) > wall_z + WALL_DEPTH_CLAMP_MIN_EXCESS:
            shift = max(zs) - wall_z          # slide toward the camera
            zs = [z - shift for z in zs]
        obj = {
            "name": relabels.get(i, d.get("label", "unknown")),
            "valid_3d": True,
            "bbox_min": [min(xs), min(ys), min(zs)],
            "bbox_max": [max(xs), max(ys), max(zs)],
            "yaw_cam_rad": d.get("yaw_rad"),
            "dimensions_whd": d.get("dimensions_w_l_h"),
        }
        if i in relabels:
            obj["relabelled_from"] = d.get("label", "unknown")
        out.append(obj)
    return out


def _zone_size_by_label(zone: dict[str, Any]) -> dict[str, Any]:
    """``{label: (short, long, height)}``; footprint-only labels get height 0 (term skipped)."""
    out: dict[str, Any] = {}
    for k, v in (zone.get("size_by_label") or {}).items():
        if v and len(v) >= 3:
            out[str(k)] = (float(v[0]), float(v[1]), float(v[2]))
    for k, v in (zone.get("footprint_by_label") or {}).items():
        if str(k) not in out and v and len(v) >= 2:
            out[str(k)] = (float(v[0]), float(v[1]), 0.0)
    return out


#: A relabelled box must not overlap a kept box, be tall enough for the target, sit within
#: ELEVATION_M (m) of the lowest kept box's bottom, and not be its own label's best box.
SLOT_FILL_GUARD_IOU = 0.3
SLOT_FILL_GUARD_HEIGHT_MIN_RATIO = 0.45
SLOT_FILL_GUARD_ELEVATION_M = 0.5


def _iou2d(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def _det_bottom_cam(det: dict[str, Any]) -> float | None:
    """Camera-frame (y-down) bottom of the box: larger = lower."""
    c = det.get("center_cam")
    dims = det.get("dimensions_w_l_h")
    if not c or not dims or len(dims) < 2:
        return None
    return float(c[1]) + float(dims[1]) / 2.0


def _slot_fill_guard(
    dets: list[dict[str, Any]],
    kept: set[int],
    size_by_label: dict[str, Any] | None,
):
    """Build the ``allow(index, target)`` callable for ``fill_slots``."""
    sizes = size_by_label or {}
    kept_list = [dets[j] for j in kept]
    bottoms = [b for b in (_det_bottom_cam(d) for d in kept_list) if b is not None]
    floor = max(bottoms) if bottoms else None
    best_by_label: dict[str, float] = {}
    for d in dets:
        if d.get("bbox3d_corners"):
            lbl = str(d.get("label", "unknown"))
            best_by_label[lbl] = max(best_by_label.get(lbl, float("-inf")), float(d.get("score", 0.0)))

    def allow(i: int, target: str) -> str | None:
        det = dets[i]
        aabb = _corner_aabb(det)
        for k in kept_list:
            if "bbox2d_projected" in det and "bbox2d_projected" in k:
                if _iou2d(det["bbox2d_projected"], k["bbox2d_projected"]) >= SLOT_FILL_GUARD_IOU:
                    return "overlap"
            else:
                ka = _corner_aabb(k)
                if aabb and ka and _footprint_iou(aabb, ka) >= SLOT_FILL_GUARD_IOU:
                    return "overlap"
        r = _height_ratio(det, sizes.get(target))
        if r is not None and r < SLOT_FILL_GUARD_HEIGHT_MIN_RATIO:
            return "height"
        bottom = _det_bottom_cam(det)
        if floor is not None and bottom is not None and floor - bottom > SLOT_FILL_GUARD_ELEVATION_M:
            return "elevated"
        if float(det.get("score", 0.0)) >= best_by_label.get(str(det.get("label", "unknown")), float("inf")):
            return "top-of-source"
        return None

    return allow


def _slot_fill_overlaps(dets: list[dict[str, Any]]):
    """``overlaps(i, j)`` for ``fill_slots``: 2D IoU of projections, else footprint IoU."""
    def overlaps(i: int, j: int) -> bool:
        a, b = dets[i], dets[j]
        if "bbox2d_projected" in a and "bbox2d_projected" in b:
            return _iou2d(a["bbox2d_projected"], b["bbox2d_projected"]) >= SLOT_FILL_GUARD_IOU
        aa, bb = _corner_aabb(a), _corner_aabb(b)
        return bool(aa and bb and _footprint_iou(aa, bb) >= SLOT_FILL_GUARD_IOU)
    return overlaps


def _zone_slot_fill(
    dets: list[dict[str, Any]],
    kept: set[int],
    zone: dict[str, Any],
    count_by_label: dict[str, int],
    size_by_label: dict[str, Any] | None = None,
) -> dict[int, str]:
    """Slot fill for one image's detections, driven by the zone spec fields."""
    fp_by_label = {
        str(k): (float(v[0]), float(v[1]))
        for k, v in (zone.get("footprint_by_label") or {}).items()
    }
    group_by_label = {
        str(k): str(v) for k, v in (zone.get("group_by_label") or {}).items()
    }
    return fill_slots(
        dets, kept, count_by_label, fp_by_label, group_by_label,
        allow=_slot_fill_guard(dets, kept, size_by_label),
        overlaps=_slot_fill_overlaps(dets),
    )


def _postprocess_image(
    dets: list[dict[str, Any]],
    zone: dict[str, Any],
    count_by_label: dict[str, int],
    K: Any = None,
    wall_estimate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dedup + slot-fill one image's detections; returns the adapter objects."""
    if K is not None:
        # image-space boxes for the nested-2D rule and the slot-fill guard
        for d in dets:
            b = _project_bbox2d(d, K)
            if b is not None:
                d["bbox2d_projected"] = [round(v, 1) for v in b]
    suppressed = cross_label_suppressed(dets)
    for i, j in suppressed.items():
        # fill_slots never relabels a suppressed box
        dets[i]["suppressed_by"] = j
    size_by_label = _zone_size_by_label(zone)
    kept = _kept_indices(dets, count_by_label, size_by_label)
    # A leftover nested in a kept box is a part of it (a vanity's doors), not a spare object.
    for j, i in _nested_leftovers(dets, kept).items():
        if j not in suppressed:
            dets[j]["suppressed_by"] = i
    relabels = _zone_slot_fill(dets, kept, zone, count_by_label, size_by_label)
    return _omni3d_to_objects(dets, count_by_label, relabels, size_by_label,
                              wall_estimate=wall_estimate)


#: A missing anchor is re-detected alone at this threshold; the best box is kept only if
#: its footprint area / catalogue area lies in ANCHOR_RESCUE_BAND.
ANCHOR_RESCUE_THRESHOLD = 0.10
ANCHOR_RESCUE_BAND = (0.45, 1.0 / 0.45)


def _anchor_status(
    dets: list[dict[str, Any]], objects: list[dict[str, Any]], anchor_label: str | None,
) -> str | None:
    """None if the anchor is present; else "not_returned" by the detector or "dropped" later."""
    if not anchor_label:
        return None
    if any(o.get("name") == anchor_label for o in objects):
        return None
    if any(d.get("label") == anchor_label for d in dets):
        return "dropped"
    return "not_returned"


def _footprint_ratio(det: dict[str, Any], catalogue: list[float] | None) -> float | None:
    """Box footprint area over the catalogue (short, long) area; None if unknown."""
    dims = det.get("dimensions_w_l_h") or det.get("dimensions_whd") or []
    if len(dims) < 3 or not catalogue or len(catalogue) < 2:
        return None
    ref = float(catalogue[0]) * float(catalogue[1])
    if ref <= 0.0:
        return None
    return (float(dims[0]) * float(dims[2])) / ref


def _mark_out_of_band_anchors(objects: list[dict[str, Any]], anchor_label: str | None,
                              size_by_label: dict[str, Any] | None) -> None:
    """Set ``size_source="measured_out_of_band"`` on anchors outside ANCHOR_RESCUE_BAND."""
    if not anchor_label:
        return
    ref = (size_by_label or {}).get(anchor_label)
    if not ref or len(ref) < 2:
        return
    lo, hi = ANCHOR_RESCUE_BAND
    for o in objects:
        if o.get("name") != anchor_label:
            continue
        r = _footprint_ratio(o, list(ref[:2]))
        if r is None or lo <= r <= hi:
            continue
        o["size_source"] = "measured_out_of_band"


def _infer_stage(
    m2: Any, cfg: Any, model: Any, stage: Path, labels_map: dict[str, list[str]],
    threshold: float, weights: str, config_file: str,
) -> dict[str, list[dict[str, Any]]]:
    """One OVMono3D pass over the images staged in ``stage``; returns detections by stem."""
    (stage / "labels.json").write_text(json.dumps(labels_map), encoding="utf-8")
    args = _build_args(
        input_folder=stage,
        labels_file=stage / "labels.json",
        threshold=threshold,
        weights=weights,
        output_dir=stage / "_out",
        config_file=config_file,
    )
    import torch
    with torch.no_grad():
        inf = m2._do_inference(args, cfg, model)
    return inf["predictions_per_image"]


def _rescue_anchor(
    m2: Any, cfg: Any, model: Any, stage: Path, stem: str, zone: dict[str, Any],
    anchor_label: str, weights: str, config_file: str,
    wall_estimate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Re-detect the missing anchor on one image; [] unless its footprint is in band."""
    sub = stage / f"_rescue_{stem}"
    sub.mkdir(exist_ok=True)
    for ext in (".png", ".npz"):
        src = stage / f"{stem}{ext}"
        dst = sub / f"{stem}{ext}"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    preds = _infer_stage(m2, cfg, model, sub, {stem: [anchor_label]},
                         ANCHOR_RESCUE_THRESHOLD, weights, config_file)
    dets = [d for d in (preds.get(stem) or []) if d.get("label") == anchor_label]
    if not dets:
        return []
    best = max(dets, key=lambda d: float(d.get("score", 0.0)))
    ratio = _footprint_ratio(best, (zone.get("footprint_by_label") or {}).get(anchor_label))
    if ratio is None:
        return []        # fail closed: cannot judge, do not accept
    lo, hi = ANCHOR_RESCUE_BAND
    if not (lo <= ratio <= hi):
        return []
    return _omni3d_to_objects([best], {anchor_label: 1}, None,
                              _zone_size_by_label(zone), wall_estimate)


def _run_zone(
    m2: Any,
    cfg: Any,
    model: Any,
    dp_model: Any,
    dp_transform: Any,
    zone: dict[str, Any],
    threshold: float,
    weights: str,
    config_file: str,
) -> dict[str, list[dict[str, Any]]]:
    """Stage one zone's images, run DepthPro + OVMono3D, return {image_path: objects_3d}."""
    import numpy as np

    images = [Path(p).resolve() for p in zone["images"]]
    labels = list(zone.get("labels") or [])
    count_by_label = {
        str(k): int(v) for k, v in (zone.get("count_by_label") or {}).items()
    }
    results: dict[str, list[dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(prefix="ovmono3d_zone_") as td:
        stage = Path(td)
        stem_to_image: dict[str, Path] = {}
        labels_map: dict[str, list[str]] = {}
        for idx, img in enumerate(images):
            stem = f"img{idx:03d}"
            staged_png = stage / f"{stem}.png"
            staged_png.symlink_to(img)
            _depthpro_npz(dp_model, dp_transform, img, stage / f"{stem}.npz")
            labels_map[stem] = labels
            stem_to_image[stem] = img
        preds_per_image = _infer_stage(m2, cfg, model, stage, labels_map,
                                       threshold, weights, config_file)
        anchor_label = zone.get("anchor_label") or None
        for stem, dets in preds_per_image.items():
            img = stem_to_image.get(stem)
            if img is None:
                continue
            # Close the npz before tempdir cleanup: an open handle on NFS makes rmtree fail.
            K = None
            depth = None
            npz_path = stage / f"{stem}.npz"
            if npz_path.is_file():
                with np.load(npz_path) as npz:
                    K = npz["K"].tolist()
                    depth = npz["depth"]
            wall_est = (
                _estimate_walls(depth, K) if (depth is not None and K is not None)
                else None
            )
            objects = _postprocess_image(
                dets, zone, count_by_label, K=K, wall_estimate=wall_est
            )
            missing = _anchor_status(dets, objects, anchor_label)
            if missing is None:
                _mark_out_of_band_anchors(objects, anchor_label, _zone_size_by_label(zone))
            elif missing == "not_returned":
                objects = [*objects, *_rescue_anchor(
                    m2, cfg, model, stage, stem, zone, anchor_label,
                    weights, config_file, wall_est)]
            results[str(img)] = objects
    return results


def load_ovmono_models(config_file: str, weights: str, depthpro_ckpt: Path,
                       threshold: float) -> tuple:
    """Load DepthPro and OVMono3D once; the tuple goes to :func:`process_spec`."""
    # Imported here: it chdirs into the OVMono3D checkout and loads detectron2.
    from scenebloom.compilation.lifting import ovmono3d_infer as m2

    print(f"ovmono3d_cli: loading DepthPro ({Path(depthpro_ckpt).name}) ...", file=sys.stderr)
    dp_model, dp_transform = _load_depthpro(Path(depthpro_ckpt))

    print("ovmono3d_cli: loading OVMono3D-LIFT model ...", file=sys.stderr)
    base_args = _build_args(
        input_folder=Path("/tmp"),
        labels_file=Path("/tmp/labels.json"),
        threshold=threshold,
        weights=weights,
        output_dir=Path(tempfile.mkdtemp(prefix="ovmono3d_setup_")),
        config_file=config_file,
    )
    cfg = m2._setup(base_args)
    model = m2.build_model(cfg)
    m2.DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True,
    )
    return m2, cfg, model, dp_model, dp_transform


def process_spec(models: tuple, spec: dict) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Run every zone of ``spec``; returns ``{zone_id: {image_path: [object]}}``."""
    m2, cfg, model, dp_model, dp_transform = models
    config_file = spec.get("config_file", "configs/OVMono3D_dinov2_SFP.yaml")
    weights = spec["weights"]
    threshold = float(spec.get("threshold", 0.30))
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for zone in spec["zones"]:
        zid = zone["zone_id"]
        print(f"ovmono3d_cli: zone {zid} ({len(zone['images'])} image(s)) ...", file=sys.stderr)
        out[zid] = _run_zone(
            m2, cfg, model, dp_model, dp_transform,
            zone=zone, threshold=threshold, weights=weights, config_file=config_file,
        )
    return out


def _load_for(spec: dict) -> tuple:
    return load_ovmono_models(
        spec.get("config_file", "configs/OVMono3D_dinov2_SFP.yaml"),
        spec["weights"], Path(spec["depthpro_ckpt"]),
        float(spec.get("threshold", 0.30)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = process_spec(_load_for(spec), spec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(len(v) for z in out.values() for v in z.values())
    print(f"ovmono3d_cli: wrote {args.output} ({len(out)} zone(s), {total} image-result(s))",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
