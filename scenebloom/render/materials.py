"""Floor and wall materials chosen by CLIP retrieval over AmbientCG (as in SceneSmith)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np

from scenebloom.utils.paths import CHECKPOINTS

# Per-room (floor, wall) text queries.
ROOM_QUERIES: dict[str, tuple[str, str]] = {
    "bathroom": (
        "light porcelain bathroom tile floor",
        "clean light plaster bathroom wall",
    ),
    "bedroom": (
        "warm light oak bedroom floor",
        "subtle warm plaster bedroom wall",
    ),
    "dining": (
        "warm natural oak dining room floor",
        "soft neutral plaster dining room wall",
    ),
    "living": (
        "warm natural oak living room floor",
        "subtle warm plaster living room wall",
    ),
    "office": (
        "neutral low pile carpet office floor",
        "clean light painted plaster office wall, vertical wall surface, not ceiling",
    ),
}


# Families CLIP can rank top for a shell query that are never a shell: ceilings look
# like plain walls, and Fabric081C's greyscale JPG crashes the Genesis pyrender path.
_NON_SHELL_TOKENS = ("ceiling", "fabric")


def material_is_non_shell(material_id: str) -> bool:
    """True for material families that are never a valid floor/wall shell."""
    material_id = material_id.lower()
    return any(token in material_id for token in _NON_SHELL_TOKENS)


def base_color_map(materials_dir: Path, material_id: str) -> Path:
    """The 2K base-colour map of an AmbientCG material."""
    return Path(materials_dir) / material_id / f"{material_id}_2K-JPG_Color.jpg"


def retrieve(
    *,
    query_embedding: np.ndarray,
    material_ids: np.ndarray,
    material_embeddings: np.ndarray,
    top_k: int = 5,
    allow: Callable[[str], bool] | None = None,
) -> list[tuple[str, float]]:
    """``(material_id, score)`` of the CLIP cosine top-k, ties broken by id."""
    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    embs = np.asarray(material_embeddings, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[1] != q.shape[0]:
        raise ValueError("query/material embedding dimensions do not match")
    q /= max(float(np.linalg.norm(q)), 1e-12)
    denom = np.maximum(np.linalg.norm(embs, axis=1), 1e-12)
    scores = (embs @ q) / denom
    ranked = sorted(
        ((str(material_ids[i]), float(scores[i])) for i in range(len(scores))
         if allow is None or allow(str(material_ids[i]))),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:top_k]


def _surface_record(
    query: str,
    *,
    query_embedding: np.ndarray,
    material_ids: np.ndarray,
    material_embeddings: np.ndarray,
    materials_dir: Path,
    top_k: int,
    sample_top_k: int,
    seed: int,
    salt: int,
) -> dict[str, str]:
    """Retrieve one surface material, formatted as one entry of an assignment JSON."""
    ranked = retrieve(
        query_embedding=query_embedding,
        material_ids=material_ids,
        material_embeddings=material_embeddings,
        top_k=top_k,
        allow=lambda material_id: not material_is_non_shell(material_id),
    )
    if not ranked:
        raise ValueError(f"no material retrieved for query {query!r}")
    width = max(1, min(sample_top_k, len(ranked)))
    material_id, _ = ranked[random.Random(seed + salt).randrange(width)]
    base_color = base_color_map(materials_dir, material_id).resolve()
    if not base_color.is_file():
        raise FileNotFoundError(f"base_color missing for {material_id}: {base_color}")
    return {"base_color": str(base_color)}


# Long room names the zoner emits, mapped to the ROOM_QUERIES keys.
ROOM_TYPE_ALIASES: dict[str, str] = {
    "living_room": "living",
    "dining_room": "dining",
}


def supported_room_type(room_type: str) -> bool:
    """Whether ``room_type``, or its long name, has floor and wall queries in ROOM_QUERIES."""
    return ROOM_TYPE_ALIASES.get(room_type, room_type) in ROOM_QUERIES


# Local weights for the bank's (model, open_clip tag), used instead of a Hugging Face download.
CLIP_WEIGHTS: dict[tuple[str, str], Path] = {
    ("ViT-H-14-378-quickgelu", "dfn5b"):
        CHECKPOINTS / "DFN5B-CLIP-ViT-H-14-378" / "open_clip_pytorch_model.bin",
}


def resolve_room_assignment(
    room_type: str,
    *,
    materials_dir: Path,
    bank_path: Path,
    seed: int = 42,
    top_k: int = 5,
    sample_top_k: int = 3,
    device: str = "cpu",
) -> dict[str, dict[str, str]]:
    """Seeded floor and wall materials for ``room_type``, as an assignment dict."""
    room_type = ROOM_TYPE_ALIASES.get(room_type, room_type)
    if room_type not in ROOM_QUERIES:
        raise KeyError(
            f"no texture query policy for room type {room_type!r}; "
            f"known: {sorted(ROOM_QUERIES)}"
        )
    materials_dir, bank_path = Path(materials_dir), Path(bank_path)
    if not materials_dir.is_dir():
        raise FileNotFoundError(f"texture directory not found: {materials_dir}")
    if not bank_path.is_file():
        raise FileNotFoundError(f"texture CLIP bank not found: {bank_path}")

    import open_clip
    import torch

    bank = np.load(bank_path, allow_pickle=False)
    model_name, pretrained = str(bank["model"]), str(bank["pretrained"])
    weights = CLIP_WEIGHTS.get((model_name, pretrained))
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=str(weights) if weights else pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    floor_query, wall_query = ROOM_QUERIES[room_type]
    with torch.no_grad():
        vectors = model.encode_text(tokenizer([floor_query, wall_query]).to(device))
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    floor_vec = vectors[0].float().cpu().numpy()
    wall_vec = vectors[1].float().cpu().numpy()

    material_ids = bank["material_ids"]
    material_embeddings = bank["embeddings"]

    floor = _surface_record(
        floor_query, query_embedding=floor_vec,
        material_ids=material_ids, material_embeddings=material_embeddings,
        materials_dir=materials_dir, top_k=top_k, sample_top_k=sample_top_k,
        seed=seed, salt=0,
    )
    wall = _surface_record(
        wall_query, query_embedding=wall_vec,
        material_ids=material_ids, material_embeddings=material_embeddings,
        materials_dir=materials_dir, top_k=top_k, sample_top_k=sample_top_k,
        seed=seed, salt=1,
    )
    return {"floor": floor, "wall": wall}


def shell_textures(material_assignment: Path) -> tuple[Path, Path]:
    """``(floor, wall)`` base-colour maps named by a ``material_assignment.json``."""
    try:
        assignment = json.loads(Path(material_assignment).read_text(encoding="utf-8"))
        floor = Path(assignment["floor"]["base_color"])
        wall = Path(assignment["wall"]["base_color"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid material assignment {material_assignment}: {exc}") from exc
    if not floor.is_file() or not wall.is_file():
        raise FileNotFoundError(f"shell texture missing: floor={floor}, wall={wall}")
    return floor, wall
