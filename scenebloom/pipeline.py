"""Scene generation pipeline: prompt -> zoning plan -> compiled layout -> Genesis renders."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing import TYPE_CHECKING

from scenebloom.utils.config import PipelineConfig

if TYPE_CHECKING:
    from scenebloom.utils.schemas import SceneLayout, SceneRecord, SceneSpec
    from scenebloom.render.renderer import GenesisRenderer

logger = logging.getLogger(__name__)


def _place_wall_objects(config, layout, asset_db):
    """Append fixed wall objects after furniture and before manipulands."""
    from scenebloom.compilation.completion.wall_placer import (
        VLMWallCategoryPlanner,
        VLMWallPoseProposer,
        WallObjectConfig,
        plan_wall_objects,
    )
    from scenebloom.utils.vlm_openrouter import call_openrouter

    # Takes a message list so the wall stages keep their system turn.
    def llm_fn(messages: list[dict[str, str]]) -> str:
        return call_openrouter(messages, model=config.llm_model)

    # The planner picks what goes on a wall, the proposer where; the placer owns the geometry.
    placed = plan_wall_objects(
        layout.model_dump(mode="json"),
        asset_db,
        proposer=VLMWallPoseProposer(llm_fn),
        planner=VLMWallCategoryPlanner(llm_fn=llm_fn),
        cfg=WallObjectConfig(max_per_zone=config.wall_object_max_per_zone),
    )
    logger.info("wall objects: added %d", len(placed))
    return layout.model_copy(
        update={
            "layout_id": f"{layout.layout_id}__wallobj",
            "placed_objects": [*layout.placed_objects, *placed],
        }
    )


#: Max ratio between a variant's aspect and the base mesh's; rejects a different object
#: under the same category name (a floor-standing vanity under ``sink``).
_VARIANT_ASPECT_TOL = 2.0

#: Max anisotropy (max/min of the per-axis size/extent factors) of an acceptable variant;
#: the render scales any mesh into the box, so only uneven stretch shows.
_VARIANT_ANISO_MAX = 1.5


def _anisotropy(size, extents) -> float:
    w = [max(s, 1e-6) / max(e, 1e-6) for s, e in zip(size, extents)]
    return max(w) / min(w)


def _mesh_extents(key: str) -> tuple[float, float, float] | None:
    """(x, y, z) extents of the mesh the render loads for ``key``, or None."""
    try:
        from scenebloom.utils.assets import (
            CATEGORY_TO_GLB_ID,
            get_object_metadata,
        )

        glb_id = CATEGORY_TO_GLB_ID.get(key)
        if glb_id is None:
            return None
        extents = get_object_metadata(glb_id).get("bounding_box_extents")
    except Exception:  # missing metadata must not break phase 2
        return None
    if not extents or len(extents) != 3 or min(extents) <= 0:
        return None
    return (float(extents[0]), float(extents[1]), float(extents[2]))


def _aspect_compatible(obj, variant_id: str) -> bool:
    """Whether ``variant_id`` keeps the base mesh's proportions; True without metadata."""
    base = _mesh_extents(obj.category)
    cand = _mesh_extents(variant_id)
    if base is None or cand is None:
        return True
    for axis in (1, 2):  # height/width and depth/width
        b = base[axis] / base[0]
        c = cand[axis] / cand[0]
        if b <= 0 or c <= 0:
            return True
        if max(b / c, c / b) > _VARIANT_ASPECT_TOL:
            logger.info(
                "asset variants: dropped %s for %s (aspect %.2fx over tolerance)",
                variant_id, obj.category, max(b / c, c / b),
            )
            return False
    return True


def _assign_asset_variants(config, layout):
    """Rewrite floor ``asset_id``s to a seeded mesh variant; a mesh FreeFuse recorded wins."""
    import hashlib

    from scenebloom.utils.assets import variant_keys

    seed = config.random_seed
    # A recorded mesh is kept even against the aspect guard: the box was lifted from its image.
    recorded = _recorded_subject_meshes(config)

    changed = 0
    objects = []
    for obj in layout.placed_objects:
        rec_mesh = recorded.get((obj.zone_id, obj.category))
        if rec_mesh and obj.asset_id == obj.category:
            chosen = rec_mesh
        else:
            pool = (
                [obj.category, *variant_keys(obj.category)]
                if obj.asset_id == obj.category
                else []
            )
            if len(pool) > 1:
                pool = [pool[0],
                        *(v for v in pool[1:] if _aspect_compatible(obj, v))]
            if len(pool) <= 1:
                objects.append(obj)
                continue
            digest = hashlib.sha1(
                f"{seed}:{obj.instance_id}:{obj.category}".encode()
            ).hexdigest()
            chosen = pool[int(digest, 16) % len(pool)]
            scored = [(v, _mesh_extents(v)) for v in pool]
            scored = [(v, e) for v, e in scored if e is not None]
            if scored:
                ok = [v for v, e in scored if _anisotropy(obj.pose.size, e) <= _VARIANT_ANISO_MAX]
                if ok:
                    chosen = ok[int(digest, 16) % len(ok)]
                else:
                    chosen = min(scored, key=lambda ve: _anisotropy(obj.pose.size, ve[1]))[0]
        if chosen != obj.category:
            obj = obj.model_copy(update={"asset_id": chosen})
            changed += 1
        objects.append(obj)

    logger.info("asset variants: reassigned %d/%d objects", changed, len(objects))
    return layout.model_copy(update={"placed_objects": objects})


def _recorded_subject_meshes(config) -> dict[tuple[str, str], str]:
    """``(zone_id, category) -> mesh`` from the ``variant_choice.json`` records FreeFuse wrote."""
    import glob as _glob

    out: dict[tuple[str, str], str] = {}
    root = Path(config.output_dir) / "proposals"
    for f in _glob.glob(str(root / "**" / "variant_choice.json"), recursive=True):
        try:
            rec = json.loads(Path(f).read_text())
        except Exception:  # a malformed record must not stop the run
            logger.warning("asset variants: unreadable record %s", f)
            continue
        zid = rec.get("zone_id")
        for cat, mesh in (rec.get("subject_meshes") or {}).items():
            if zid:
                out[(zid, cat)] = mesh
    return out


def _place_small_objects(config, layout, asset_db):
    """Place small objects on furniture surfaces; errors propagate."""
    from scenebloom.utils.vlm_openrouter import call_openrouter
    from scenebloom.compilation.completion.small_objects import place_small_objects
    from scenebloom.compilation.completion.surface_placer import LLMProposer

    surface_assets = asset_db.query_surface_mounted()
    if not surface_assets:
        raise ValueError(f"no SURFACE assets found in {config.assets_path}")
    room_type = layout.zoning_plan.rooms[0].room_type
    supports = tuple(
        item.strip() for item in config.small_object_supports.split(",") if item.strip()
    )

    # The VLM picks assets and rough positions per surface; ``validate_and_settle`` checks them.
    def _llm_fn(prompt: str) -> str:
        raw = call_openrouter(
            [{"role": "user", "content": prompt}], model=config.llm_model,
        )
        # Strip prose and code fences around the JSON array.
        lo, hi = raw.find("["), raw.rfind("]")
        return raw[lo : hi + 1] if lo != -1 and hi != -1 else raw

    final_layout = place_small_objects(
        layout,
        surface_assets,
        room_type=room_type,
        supports=supports,
        max_candidates=config.small_object_max_candidates,
        proposer_factory=lambda n: LLMProposer(llm_fn=_llm_fn, max_candidates=n),
        retry_rounds=config.small_object_retry_rounds,
    )
    logger.info(
        "small objects: added %d object(s)",
        len(final_layout.placed_objects) - len(layout.placed_objects),
    )
    return final_layout


class PipelineError(RuntimeError):
    """Failure of one phase; ``phase`` names it and ``cause`` holds the original exception."""

    def __init__(
        self,
        phase: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(f"[{phase}] {message}")
        self.phase = phase
        self.cause = cause


class PipelineOrchestrator:
    """Generate one scene from a text prompt; ``strategy`` defaults to ``FreeFuseStrategy``."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        strategy=None,
    ) -> None:
        self._config = config or PipelineConfig()
        if strategy is None:
            from scenebloom.sampling.proposals.strategy import FreeFuseStrategy

            strategy = FreeFuseStrategy(config=self._config)
        self._strategy = strategy
        self._material_assignment: Path | None = None

    def run(self, prompt: str) -> SceneLayout:
        """Run all phases into ``output_dir``; return the layout or raise ``PipelineError``."""
        out = self._config.output_dir
        logger.info("Pipeline run starting: %r", prompt[:80])
        out.mkdir(parents=True, exist_ok=True)

        # Phase 1: prompt -> zoning plan.
        try:
            from scenebloom.render.materials import ROOM_QUERIES, supported_room_type
            from scenebloom.sampling.zoning.prompt_parser import PromptParser
            from scenebloom.sampling.zoning.llm_zone_decomposer import (
                LLMZoneDecomposer,
            )

            spec = PromptParser().parse(prompt)
            room_type = spec.room_constraints[0].room_type
            if not supported_room_type(room_type):
                raise ValueError(
                    f"the prompt describes a {room_type!r} room; SceneBloom generates "
                    f"{', '.join(sorted(ROOM_QUERIES))} rooms"
                )
            retrieved = self._retrieve(spec)
            decomposer = LLMZoneDecomposer(
                m3dlayout_root=self._config.m3dlayout_root,
                model=self._config.llm_model,
            )
            plan = decomposer.decompose(spec, retrieved)
            logger.info("Phase 1 done: %d room(s) planned", len(plan.rooms))
        except Exception as exc:
            raise PipelineError("phase1", str(exc), exc) from exc

        # Phase 2: zoning plan -> layout.
        try:
            from scenebloom.compilation.compiler import compile_scene_from_pipeline
            from scenebloom.utils.assets import AssetDB
            from scenebloom.utils.assets import AssetSelector

            asset_db = AssetDB.from_path(self._config.assets_path)
            candidates = AssetSelector(asset_db).select_for_plan(plan)

            layout = self._strategy.generate(plan, candidates)
            layout = compile_scene_from_pipeline(self._config, layout)
            # Chosen in the layout so the render and the Habitat export load the same meshes.
            layout = _assign_asset_variants(self._config, layout)
            layout = _place_wall_objects(self._config, layout, asset_db)
            layout = _place_small_objects(self._config, layout, asset_db)
            (out / "scene_layout.json").write_text(
                layout.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Phase 2 done: %d object(s) placed", len(layout.placed_objects))
        except Exception as exc:
            raise PipelineError("phase2", str(exc), exc) from exc

        # Phase 3: layout -> renders.
        try:
            render_dir = out / "renders"
            self._material_assignment = self._texture_assignment(layout)
            renderer = self._build_renderer()
            renderer.load_scene(layout)
            renderer.render(render_dir)
            logger.info("Phase 3 render: %s", render_dir)
        except Exception as exc:
            raise PipelineError("phase3", str(exc), exc) from exc
        return layout

    def _retrieve(self, spec: SceneSpec) -> list[SceneRecord]:
        """Exemplars for the zoner from the embedding index."""
        from scenebloom.sampling.zoning.retrieval import (
            EmbeddingIndex,
            EmbeddingSceneRetriever,
            Qwen3Encoder,
            select_exemplars,
        )

        index_dirs = [self._config.embed_index_dir, *self._config.embed_index_extra_dirs]
        if not any((d / "embeddings.npy").exists() for d in index_dirs):
            raise FileNotFoundError(
                f"no embedding index in any of {[str(d) for d in index_dirs]}; "
                "build it with scripts/build_index/build_retrieval_index.sh"
            )
        index = EmbeddingIndex.from_dirs(index_dirs)
        retriever = EmbeddingSceneRetriever(index, Qwen3Encoder())
        candidate_k = max(
            self._config.retrieval_candidate_k,
            self._config.retrieval_context_n,
        )
        ranked = retriever.retrieve(spec, k=candidate_k)
        return select_exemplars(
            ranked,
            spec,
            context_n=self._config.retrieval_context_n,
            candidate_k=candidate_k,
            global_seed=self._config.random_seed,
            temperature=self._config.retrieval_temperature,
            max_score_drop=self._config.retrieval_max_score_drop,
        )

    def _texture_assignment(self, layout) -> Path:
        """Resolve the room's floor/wall materials and write ``material_assignment.json``."""
        from scenebloom.render.materials import resolve_room_assignment

        config = self._config
        room_type = layout.zoning_plan.rooms[0].room_type
        assignment = resolve_room_assignment(
            room_type,
            materials_dir=config.texture_dir,
            bank_path=config.texture_bank_path,
            seed=config.random_seed,
            top_k=config.texture_top_k,
            sample_top_k=config.texture_sample_top_k,
        )
        out = config.output_dir / "material_assignment.json"
        out.write_text(json.dumps(assignment, indent=2), encoding="utf-8")
        logger.info(
            "texture[%s]: floor=%s wall=%s",
            room_type,
            assignment["floor"]["base_color"],
            assignment["wall"]["base_color"],
        )
        return out.resolve()

    def _build_renderer(self) -> "GenesisRenderer":
        """Build the Genesis renderer (requires the 'genesis' package)."""
        from scenebloom.render.renderer import GenesisRenderer

        return GenesisRenderer(material_assignment=self._material_assignment)
