"""FreeFuse FLUX.1-dev generation of the per-zone layout proposals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from scenebloom.sampling.proposals.prompts import (
    LORA_DIR_BY_SUBJECT,
    build_concept_map,
    build_prompts,
    find_concept_positions,
    get_prompt_token_texts,
    instance_records,
    prepare_prompt_embeds,
    weights_path,
)
from scenebloom.utils.paths import CHECKPOINTS

logger = logging.getLogger(__name__)


def _ensure_subject_loras(
    pipe: Any,
    subjects: list[str],
    lora_root: str,
    meshes: dict[str, str] | None = None,
) -> None:
    """Load each subject's LoRA once per (subject, mesh), under the subject's adapter name."""
    loaded: dict = getattr(pipe, "_freefuse_loaded_adapters", None)
    if not isinstance(loaded, dict):
        loaded = {}
    # Attach first, so loads before a failure stay recorded: diffusers refuses to re-load a name.
    pipe._freefuse_loaded_adapters = loaded
    for subj in subjects:
        mesh = (meshes or {}).get(subj, subj)
        lora_path = _resolve_lora_path(Path(lora_root), subj, meshes)
        if not lora_path.exists():
            raise FileNotFoundError(
                f"LoRA weights not found for '{subj}' (mesh {mesh}): {lora_path}"
            )
        if loaded.get(subj) == mesh:
            logger.info("generator: LoRA %s (%s) already loaded — reusing", subj, mesh)
            continue
        if subj in loaded:
            if not hasattr(pipe, "delete_adapters"):
                raise RuntimeError(
                    f"subject {subj!r} needs mesh {mesh} but adapter holds "
                    f"{loaded[subj]} and this pipeline cannot delete adapters"
                )
            logger.info("generator: swapping LoRA %s: %s -> %s", subj, loaded[subj], mesh)
            pipe.delete_adapters(subj)
            loaded.pop(subj, None)
        logger.info("generator: loading LoRA %s (%s) <- %s", subj, mesh, lora_path)
        # diffusers needs an explicit weight_name under HF_HUB_OFFLINE, even for a file path.
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name,
                               adapter_name=subj)
        loaded[subj] = mesh


def _resolve_lora_path(
    lora_root: Path, subject: str, meshes: dict[str, str] | None = None
) -> Path:
    """LoRA weights for ``subject``: its zone mesh's LoRA if given, else the category's."""
    mesh = (meshes or {}).get(subject)
    if mesh and mesh != subject:
        return weights_path(instance_records()[mesh])
    return lora_root / LORA_DIR_BY_SUBJECT[subject] / "pytorch_lora_weights.safetensors"


def load_flux_pipe(torch_dtype: Any = None) -> Any:
    """Load the FreeFuse FLUX.1-dev pipeline onto CUDA; reuse it across zones."""
    import torch
    from scenebloom.sampling.proposals.freefuse.pipeline.freefuse_flux_direct_extract_attn_bais_background_except_pipeline import (
        FreeFuseFluxDirectExtractAttnBaisBGExceptPipeline,
    )

    dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
    logger.info("generator: loading FLUX.1-dev pipeline …")
    pipe = FreeFuseFluxDirectExtractAttnBaisBGExceptPipeline.from_pretrained(
        str(CHECKPOINTS / "FLUX.1-dev"), torch_dtype=dtype
    )
    pipe.to("cuda")
    return pipe


def generate_freefuse_images_for_zone(
    subjects: list[str],
    run_dir: Path,
    lora_root: Path,
    seeds: list[int],
    pipe: Any | None = None,
    zone_prompt: str | None = None,
    prompt_aliases: dict[str, str] | None = None,
    subject_meshes: dict[str, str] | None = None,
) -> tuple[list[Path], Any]:
    """Generate one FLUX multi-LoRA PNG per seed for one zone; returns ``(png_paths, pipe)``."""
    import torch
    from diffusers.models.transformers.transformer_flux import (
        FluxAttention,
        FluxTransformerBlock,
    )
    from peft.tuners.lora.layer import Linear, LoraLayer

    from scenebloom.sampling.proposals.freefuse.attn_processor.freefuse_attn_processor import FreeFuseFluxAttnProcessor
    from scenebloom.sampling.proposals.freefuse.models.freefuse_transformer_flux import (
        FluxTransformer2DModel,
        FreeFuseFluxAttention,
        FreeFuseFluxTransformerBlock,
    )
    from scenebloom.sampling.proposals.freefuse.tuner.freefuse_lora_layer import FreeFuseLinear  # noqa: F401

    if pipe is None:
        pipe = load_flux_pipe()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _ensure_subject_loras(pipe, subjects, lora_root, subject_meshes)

    pipe.set_adapters(subjects, [1.0] * len(subjects))
    pipe.transformer.__class__ = FluxTransformer2DModel

    current = pipe.transformer.attn_processors
    proc_dict: dict = {}
    for name in current:
        if "single_transformer_blocks" in name:
            proc_dict[name] = current[name]
        elif "transformer_blocks.18.attn" in name and "single" not in name:
            p = FreeFuseFluxAttnProcessor()
            p.cal_concept_sim_map = True
            proc_dict[name] = p
        else:
            proc_dict[name] = FreeFuseFluxAttnProcessor()
    pipe.transformer.set_attn_processor(proc_dict)

    for parent in pipe.transformer.modules():
        for _, module in parent.named_modules():
            if isinstance(module, LoraLayer) and isinstance(module, Linear):
                FreeFuseLinear.init_from_lora_linear(module)
    for parent in pipe.transformer.modules():
        for _, module in parent.named_modules():
            if isinstance(module, FluxTransformerBlock):
                module.__class__ = FreeFuseFluxTransformerBlock
            if isinstance(module, FluxAttention):
                module.__class__ = FreeFuseFluxAttention

    clip_prompt, t5_prompt = build_prompts(
        subjects, zone_prompt=zone_prompt, aliases=prompt_aliases,
        meshes=subject_meshes,
    )
    concept_map = build_concept_map(
        subjects, zone_prompt=zone_prompt, aliases=prompt_aliases,
        meshes=subject_meshes,
    )
    # No true CFG: FLUX.1-dev is guidance-distilled.
    true_cfg_scale = 0.0
    neg_text = None
    pooled, embeds, neg_pooled, neg_embeds = prepare_prompt_embeds(
        pipe, clip_prompt, t5_prompt, negative_prompt=neg_text
    )
    token_pos_maps = find_concept_positions(pipe, t5_prompt, concept_map)
    pipe.transformer.set_freefuse_token_pos_maps(token_pos_maps)
    token_texts = {s: get_prompt_token_texts(pipe, t5_prompt) for s in concept_map}

    saved: list[Path] = []
    for seed in seeds:
        gen = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt_embeds=embeds,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            concept_embeds_map=None,
            height=1024,
            width=1024,
            generator=gen,
            joint_attention_kwargs={
                "freefuse_token_pos_maps": token_pos_maps,
                "concept_token_texts": token_texts,
            },
            guidance_scale=7,
            true_cfg_scale=true_cfg_scale,
            aggreate_lora_score_step=5,
            use_attention_bias=True,
        ).images[0]
        out = run_dir / f"flux_seed{seed}.png"
        image.save(str(out))
        logger.info("generator: saved %s", out)
        saved.append(out)
    return saved, pipe


def generate_freefuse_batch(
    zone_specs: list[dict[str, Any]],
    lora_root: Path,
    seeds: list[int],
) -> dict[str, list[Path]]:
    """Run every zone spec on one FLUX pipe; a failed zone is logged and left out."""
    results: dict[str, list[Path]] = {}
    pipe = None
    for spec in zone_specs:
        zone_id = spec["zone_id"]
        subjects = list(spec["subjects"])
        run_dir = Path(spec["run_dir"])
        # Per-zone seeds from the strategy override the batch seeds.
        zone_seeds = [int(x) for x in (spec.get("seeds") or seeds)]
        try:
            saved, pipe = generate_freefuse_images_for_zone(
                subjects=subjects,
                run_dir=run_dir,
                lora_root=Path(lora_root),
                seeds=zone_seeds,
                pipe=pipe,
                zone_prompt=spec.get("diffusion_prompt"),
                prompt_aliases=spec.get("prompt_aliases"),
                subject_meshes=spec.get("subject_meshes"),
            )
            results[zone_id] = saved
        except Exception as exc:
            logger.warning(
                "generator: zone %s produced no image (%s)", zone_id, exc,
                exc_info=True,
            )
    return results


def apply_sdpa_backend_policy() -> None:
    """Disable cuDNN SDPA, which has no execution plan for CLIP's attention on H100/H200/B200."""
    try:
        import torch
    except Exception:
        return
    fn = getattr(getattr(torch.backends, "cuda", None), "enable_cudnn_sdp", None)
    if fn is None:
        return
    fn(False)
    enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if callable(enabled) and enabled():
        raise RuntimeError("torch still reports cudnn_sdp_enabled -- the disable did not take")
