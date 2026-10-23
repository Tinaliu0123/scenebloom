"""FreeFuseStrategy: per-zone FreeFuse FLUX images lifted into one room layout."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from scenebloom.utils.config import PipelineConfig
from scenebloom.utils.paths import CHECKPOINTS
from scenebloom.utils.schemas import AssetRecord, SceneLayout, ZoningPlan
from scenebloom.sampling.proposals.prompts import (
    DEFAULT_CATEGORY_ALIASES,
    LORA_ROOT,
    SUBJECT_META,
    alias_replacements,
    choose_mesh,
    split_subjects_by_prompt,
)
from scenebloom.compilation.lifting.slot_fill import group_of, sorted_footprint


def zone_short_id(zone_id: str) -> str:
    """8-char zone directory name, ``sha1(zone_id)[:8]``; a plain id prefix would collide."""
    return hashlib.sha1(zone_id.encode("utf-8")).hexdigest()[:8]


logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Best-effort JSON write; a failure is logged, never raised."""
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("FreeFuseStrategy: could not write %s: %s", path, exc)


_LIBRARY_CATS = list(SUBJECT_META)
# category -> open-vocab detector noun, e.g. chair -> "office chair".
_LIBRARY_NOUN = {cat: meta["noun"] for cat, meta in SUBJECT_META.items()}

_VALID_SUBJECTS = frozenset(_LIBRARY_CATS)

_DEFAULT_CATEGORY_MAP: dict[str, str] = {c: c for c in _LIBRARY_CATS}
_DEFAULT_CATEGORY_MAP.update(DEFAULT_CATEGORY_ALIASES)


_MAX_SUBJECTS_PER_ZONE = 5


def derive_zone_seed(global_seed: int, room_type: str | None, zone_id: str, attempt: int = 0) -> int:
    """Stable 31-bit FLUX seed per zone, so two zones with the same prompt still differ."""
    payload = "|".join([str(int(global_seed)), room_type or "", str(zone_id),
                        str(int(attempt))])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF


def _anchor_bookkeeping(lift_specs: list[dict[str, Any]],
                        batch_lift: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, bool]]:
    """Per anchored zone: anchor missing from the lift, or outside the catalogue size band."""
    out: dict[str, dict[str, bool]] = {}
    for zs in lift_specs:
        anchor = zs.get("anchor_category")
        if not anchor:
            continue
        zid = zs["zone_id"]
        anchors = [o for per in (batch_lift.get(zid) or {}).values() for o in per
                   if o.get("name") == anchor]
        out[zid] = {
            "missing_after": not anchors,
            "out_of_band": any(o.get("size_source") == "measured_out_of_band" for o in anchors),
        }
    return out


class FreeFuseTotalFailure(RuntimeError):
    """No zone of the scene yields a lifted FreeFuse proposal."""


class FreeFuseStrategy:
    """Scene generation strategy: FreeFuse multi-LoRA FLUX per zone, lifted with OVMono3D."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        lora_root: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._lora_root = Path(lora_root or LORA_ROOT)
        self._output_dir = Path(output_dir or (self._config.output_dir / "proposals"))
        self._category_map = _DEFAULT_CATEGORY_MAP

    def generate(
        self,
        plan: ZoningPlan,
        zone_candidates: dict[str, list[AssetRecord]],
    ) -> SceneLayout:
        """Draw one FreeFuse image per zone and lift all zones into one SceneLayout."""
        layout_id = f"layout_{uuid.uuid4().hex[:8]}"
        room = plan.rooms[0] if plan.rooms else None
        if room is None or not room.functional_zones:
            raise FreeFuseTotalFailure("the zoning plan has no zones")

        run_dir = self._output_dir / layout_id
        run_dir.mkdir(parents=True, exist_ok=True)

        zone_specs: list[dict[str, Any]] = []
        zone_metadata: dict[str, tuple[Any, list[str]]] = {}
        for zone in room.functional_zones:
            zone_short = zone_short_id(zone.zone_id)
            cands = zone_candidates.get(zone.zone_id, [])
            zone_subjects = self._zone_subjects(cands)

            if len(zone_subjects) < 1:
                logger.warning(
                    "FreeFuseStrategy: zone %s has no valid subjects; "
                    "skipping FreeFuse for this zone",
                    zone_short,
                )
                continue

            zone_dir = run_dir / f"zone_{zone_short}"
            zone_dir.mkdir(parents=True, exist_ok=True)
            # Subjects := allowed ∩ prompt-named.
            zone_subjects, prompt_aliases = self._subjects_named_in_prompt(
                zone, zone_short, zone_subjects
            )
            # Fix each subject's mesh before FLUX runs; pipeline.py reads variant_choice.json back.
            subject_meshes = {
                subj: choose_mesh(subj, zone.zone_id, int(self._config.random_seed))
                for subj in zone_subjects
            }
            _write_json(zone_dir / "variant_choice.json", {
                "zone_id": zone.zone_id,
                "subject_meshes": subject_meshes,
            })
            zone_specs.append({
                "zone_id": zone.zone_id,
                "subjects": zone_subjects,
                "run_dir": str(zone_dir),
                "diffusion_prompt": getattr(zone, "diffusion_prompt", None),
                "prompt_aliases": prompt_aliases,
                "subject_meshes": subject_meshes,
            })
            zone_metadata[zone.zone_id] = (zone, zone_subjects)

        if not zone_specs:
            raise FreeFuseTotalFailure("no zone has a subject with a LoRA adapter")

        _room_type = getattr(room, "room_type", None) if room is not None else None
        for _zs in zone_specs:
            _zs["seeds"] = [derive_zone_seed(int(getattr(self._config, "random_seed", 0) or 0),
                                             _room_type, _zs["zone_id"], 0)]

        try:
            zone_images = self._run_freefuse(zone_specs)
        except Exception as exc:
            raise FreeFuseTotalFailure(
                f"FreeFuse produced no images for any zone ({exc})"
            ) from exc

        if not zone_images:
            raise FreeFuseTotalFailure("FreeFuse returned zero zone images")

        lift_specs: list[dict[str, Any]] = []
        for zone_id, image_paths in zone_images.items():
            if zone_id not in zone_metadata or not image_paths:
                continue
            zone, zone_subjects = zone_metadata[zone_id]
            lift_specs.append({
                "zone_id": zone_id,
                "image_paths": [Path(p) for p in image_paths[:1]],
                "subjects": zone_subjects,
                "category_counts": getattr(zone, "category_counts", None),
                # catalogue footprints for the ovmono3d slot-fill relabel
                "candidates": zone_candidates.get(zone_id, []),
                "anchor_category": getattr(zone, "anchor_category", None),
            })
        if not lift_specs:
            raise FreeFuseTotalFailure("FreeFuse returned no image for a planned zone")
        batch_lift = self._run_ovmono3d_zones(lift_specs)

        # Regenerate once (seed attempt=1) any zone whose anchor is missing or out of size band.
        anchor_status = _anchor_bookkeeping(lift_specs, batch_lift)
        _by_id = {zs["zone_id"]: zs for zs in zone_specs}
        for _lz in lift_specs:
            _zid = _lz["zone_id"]
            _st = anchor_status.get(_zid)
            if not _st or not (_st["missing_after"] or _st["out_of_band"]) or _zid not in _by_id:
                continue
            _orig_lift, _orig_imgs = batch_lift.get(_zid, {}), zone_images.get(_zid)
            _seed1 = derive_zone_seed(int(getattr(self._config, "random_seed", 0) or 0),
                                      _room_type, _zid, attempt=1)
            logger.info("FreeFuseStrategy: zone %s anchor %r still missing; regenerating once "
                        "(seed %d) and lifting again", zone_short_id(_zid),
                        _lz["anchor_category"], _seed1)
            try:
                _imgs = self._run_freefuse([{**_by_id[_zid], "seeds": [_seed1]}])
            except Exception as exc:  # a failed retry must not stop the scene
                logger.warning("FreeFuseStrategy: zone %s regeneration failed (%s)",
                               zone_short_id(_zid), exc)
                continue
            _new = list((_imgs or {}).get(_zid) or [])
            if not _new:
                logger.warning("FreeFuseStrategy: zone %s regeneration gave no image",
                               zone_short_id(_zid))
                continue
            try:
                _lift1 = self._run_ovmono3d_zones([{**_lz, "image_paths": [Path(p) for p in _new[:1]]}])
            except Exception as exc:
                logger.warning("FreeFuseStrategy: zone %s lift of the regenerated image failed (%s)",
                               zone_short_id(_zid), exc)
                continue
            zone_images[_zid] = _new
            batch_lift[_zid] = _lift1.get(_zid, {})
            # Losing an out-of-band anchor is worse than keeping it: restore the original.
            if (_st["out_of_band"] and _orig_imgs
                    and _anchor_bookkeeping([_lz], batch_lift)[_zid]["missing_after"]):
                zone_images[_zid] = _orig_imgs
                batch_lift[_zid] = _orig_lift

        # The compiler reads the zone's final proposal image from proposal.json.
        for zone_id, image_paths in zone_images.items():
            if zone_id in _by_id and image_paths:
                (Path(_by_id[zone_id]["run_dir"]) / "proposal.json").write_text(
                    json.dumps({"image": Path(image_paths[0]).name}, indent=2), encoding="utf-8")

        all_placed: list[Any] = []
        for zone_id, image_paths in zone_images.items():
            if zone_id not in zone_metadata or not image_paths:
                continue
            zone, _ = zone_metadata[zone_id]
            try:
                zone_placed = self._lift_and_place_zone(
                    [Path(p) for p in image_paths[:1]],
                    zone,
                    batch_lift.get(zone_id, {}),
                    zone_candidates.get(zone_id, []),
                )
                all_placed.extend(zone_placed)
            except Exception as exc:
                logger.warning(
                    "FreeFuseStrategy: zone %s is not placed: lifting raised %s",
                    zone_short_id(zone_id), exc, exc_info=True,
                )

        if not all_placed:
            raise FreeFuseTotalFailure(
                f"{len(zone_images)} zone(s) generated images but no object survived the lift"
            )
        return SceneLayout(
            layout_id=layout_id,
            zoning_plan=plan,
            placed_objects=all_placed,
        )

    def _zone_subjects(self, candidates: list[AssetRecord]) -> list[str]:
        """Up to ``_MAX_SUBJECTS_PER_ZONE`` unique FreeFuse subjects from one zone's candidates."""
        seen: list[str] = []
        capped: list[str] = []
        for asset in candidates:
            subj = self._category_map.get(asset.category)
            if subj and subj in _VALID_SUBJECTS and subj not in seen and subj not in capped:
                if len(seen) < _MAX_SUBJECTS_PER_ZONE:
                    seen.append(subj)
                else:
                    capped.append(subj)
        if capped:
            logger.warning("FreeFuseStrategy: zone subject cap %d reached; dropped %s",
                           _MAX_SUBJECTS_PER_ZONE, capped)
        return seen

    def _subjects_named_in_prompt(
        self,
        zone: Any,
        zone_short: str,
        subjects: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Keep subjects the zone's ``diffusion_prompt`` names; returns ``(subjects, aliases)``."""
        prompt = getattr(zone, "diffusion_prompt", None)
        aliases = alias_replacements(subjects, _LIBRARY_NOUN, self._category_map)
        if not (prompt or "").strip():
            logger.warning(
                "FreeFuseStrategy: zone %s has no diffusion_prompt; using all "
                "%d subjects", zone_short, len(subjects),
            )
            return subjects, aliases
        named, dropped = split_subjects_by_prompt(
            prompt, subjects, _LIBRARY_NOUN, aliases
        )
        for s in dropped:
            logger.warning(
                "FreeFuseStrategy: zone %s: %s allowed but not named in prompt "
                "-> dropped", zone_short, s,
            )
        if not named:
            logger.warning(
                "FreeFuseStrategy: zone %s: prompt names none of %s; using all of them",
                zone_short, subjects,
            )
            return subjects, aliases
        return named, aliases

    def _run_freefuse(
        self, zone_specs: list[dict[str, Any]]
    ) -> dict[str, list[Path]]:
        """Generate the zone images with FLUX in a cli_generate subprocess."""
        spec = {
            "lora_root": str(self._lora_root),
            "seeds": [int(self._config.random_seed)],
            "zones": list(zone_specs),
        }
        # cli_generate exits with 2 when no zone produced an image; the caller handles that.
        payload = _run_module("scenebloom.sampling.proposals.cli_generate", spec,
                              ok_codes=(0, 2))
        results: dict[str, list[Path]] = {
            zid: [Path(p) for p in paths]
            for zid, paths in payload.get("results", {}).items()
        }
        logger.info("FreeFuseStrategy: FLUX produced %d/%d zones", len(results), len(zone_specs))
        return results

    def _lift_and_place_zone(
        self,
        image_paths: list[Path],
        zone: Any,
        per_image: dict[str, list[dict[str, Any]]],
        zone_candidates: list[Any] | None = None,
    ) -> list[Any]:
        """Place one zone's lifted objects (image -> objects_3d) in the room frame."""
        from scenebloom.compilation.lifting.adapter import lift3d_to_placed_objects
        from scenebloom.compilation.lifting.adapter import resolve_mount_table

        xs = [pt[0] for pt in zone.polygon_2d]
        zs = [pt[1] for pt in zone.polygon_2d]
        zone_origin_x = min(xs)
        zone_origin_z = min(zs)
        zone_x_m = max(xs) - zone_origin_x
        zone_z_m = max(zs) - zone_origin_z

        # DB-wide mount table first: the image can show categories that retrieval did not return.
        overrides: dict[str, str] = {}
        for rec in zone_candidates or []:
            mt = getattr(rec.mount_type, "value", rec.mount_type)
            overrides[rec.category] = str(mt)
        category_to_mount_type = resolve_mount_table(overrides)

        # Packing, de-collision and clamping run later, once per room, in the compiler.
        all_placed: list[Any] = []
        for img_path in image_paths:
            img_key = str(Path(img_path).resolve())
            objects_3d = per_image.get(img_key, [])
            all_placed.extend(
                lift3d_to_placed_objects(
                    objects_3d,
                    zone.zone_id,
                    room_x_m=zone_x_m,
                    room_z_m=zone_z_m,
                    zone_origin_x_m=zone_origin_x,
                    zone_origin_z_m=zone_origin_z,
                    category_to_mount_type=category_to_mount_type,
                    wall_orientation=getattr(zone, "wall_orientation", None),
                )
            )

        # Rotate the zone so its largest object is wall-parallel (undoes camera obliqueness).
        from scenebloom.compilation.lifting.wall_anchor import snap_zone_parallel

        return snap_zone_parallel(all_placed)

    def _ovmono3d_zone_entry(
        self,
        image_paths: list[Path],
        subjects: list[str],
        zone_id: str,
        category_counts: dict[str, int] | None,
        candidates: list[AssetRecord] | None = None,
        anchor_category: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """One zone's ovmono3d spec entry and its detector-noun -> category map."""
        labels = [_LIBRARY_NOUN.get(s, s.replace("_", " ")) for s in subjects]
        count_by_label: dict[str, int] = {}
        for cat, n in (category_counts or {}).items():
            subj = self._category_map.get(cat, cat)
            if subj not in subjects:
                continue
            label = _LIBRARY_NOUN.get(subj, subj.replace("_", " "))
            count_by_label[label] = max(count_by_label.get(label, 1), max(1, int(n)))
        footprint_by_label: dict[str, list[float]] = {}
        size_by_label: dict[str, list[float]] = {}
        for rec in candidates or []:
            subj = self._category_map.get(rec.category, rec.category)
            if subj not in subjects:
                continue
            label = _LIBRARY_NOUN.get(subj, subj.replace("_", " "))
            if label in footprint_by_label:
                continue  # first candidate per subject = retrieval's top pick
            fx, fz = rec.footprint_size
            footprint_by_label[label] = list(sorted_footprint(fx, fz))
            # Height too: a drawer has a plausible footprint but an impossible height.
            size_by_label[label] = list(sorted_footprint(fx, fz)) + [
                float(rec.bbox_size[1])
            ]
        group_by_label = {
            labels[i]: group_of(subjects[i])
            for i in range(len(subjects))
            if group_of(subjects[i]) is not None
        }
        # Anchor as a detector noun; None when it is not among this zone's subjects.
        _asubj = self._category_map.get(anchor_category, anchor_category) if anchor_category else None
        anchor_label = (_LIBRARY_NOUN.get(_asubj, _asubj.replace("_", " "))
                        if _asubj and _asubj in subjects else None)
        entry = {
            "zone_id": zone_id,
            "images": [str(Path(p).resolve()) for p in image_paths],
            "labels": labels,
            "anchor_label": anchor_label,
            "count_by_label": count_by_label,
            "footprint_by_label": footprint_by_label,
            "size_by_label": size_by_label,
            "group_by_label": group_by_label,
        }
        noun_to_cat = {labels[i]: subjects[i] for i in range(len(subjects))}
        return entry, noun_to_cat

    @staticmethod
    def _remap_ovmono3d_payload(
        payload: dict[str, Any], remaps: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Rename detector nouns back to canonical categories, per zone, in place."""
        out: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for zid, noun_to_cat in remaps.items():
            per_image = payload.get(zid, {})
            n_relabelled = 0
            for objs in per_image.values():
                for o in objs:
                    o["name"] = noun_to_cat.get(o.get("name"), o.get("name"))
                    if "relabelled_from" in o:
                        o["relabelled_from"] = noun_to_cat.get(
                            o["relabelled_from"], o["relabelled_from"]
                        )
                        n_relabelled += 1
            if n_relabelled:
                logger.info(
                    "FreeFuseStrategy: zone %s slot-fill relabelled %d box(es): %s",
                    zid[:8], n_relabelled,
                    ", ".join(
                        f"{o['relabelled_from']}->{o['name']}"
                        for objs in per_image.values() for o in objs
                        if "relabelled_from" in o
                    ),
                )
            out[zid] = per_image
        return out

    def _run_ovmono3d_zones(
        self, zone_lift_specs: list[dict[str, Any]]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Lift all zones in one OVMono3D subprocess; returns ``{zone_id: {image: objects_3d}}``."""
        ckpt = CHECKPOINTS / "ovmono3d"

        zone_entries: list[dict[str, Any]] = []
        remaps: dict[str, dict[str, str]] = {}
        for zs in zone_lift_specs:
            entry, noun_to_cat = self._ovmono3d_zone_entry(
                zs["image_paths"], zs["subjects"], zs["zone_id"],
                zs.get("category_counts"), zs.get("candidates"),
                anchor_category=zs.get("anchor_category"),
            )
            zone_entries.append(entry)
            remaps[zs["zone_id"]] = noun_to_cat

        spec = {
            "config_file": "configs/OVMono3D_dinov2_SFP.yaml",
            "weights": str(ckpt / "ovmono3d_lift.pth"),
            "depthpro_ckpt": str(ckpt / "depth_pro.pt"),
            "threshold": 0.30,
            "zones": zone_entries,
        }

        payload = _run_module("scenebloom.compilation.lifting.ovmono3d_cli", spec,
                              no_user_site=True)
        return self._remap_ovmono3d_payload(payload, remaps)


def _run_module(module: str, spec: dict[str, Any], ok_codes: tuple[int, ...] = (0,),
                no_user_site: bool = False) -> Any:
    """Run ``python -m <module> --spec ... --output ...``; JSON files keep library log noise out."""
    env = os.environ.copy()
    if no_user_site:
        # A package in the user site (~/.local) would shadow the env's own CUDA build.
        env["PYTHONNOUSERSITE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{existing_pp}" if existing_pp else str(_REPO_ROOT)
    with tempfile.TemporaryDirectory(prefix="scenebloom_sub_") as td:
        spec_path, out_path = Path(td) / "spec.json", Path(td) / "out.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        logger.info("FreeFuseStrategy: running %s in a subprocess", module)
        proc = subprocess.run(
            [sys.executable, "-m", module, "--spec", str(spec_path), "--output", str(out_path)],
            env=env, check=False,
        )
        if proc.returncode not in ok_codes or not out_path.exists():
            raise RuntimeError(f"{module} failed with rc={proc.returncode}; see stderr above")
        return json.loads(out_path.read_text(encoding="utf-8"))
