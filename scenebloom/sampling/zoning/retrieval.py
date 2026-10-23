"""Exemplar retrieval for the zoner: scene records, Qwen3 embeddings, seeded choice."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scenebloom.utils.paths import CHECKPOINTS
from scenebloom.utils.schemas import SceneRecord, SceneSpec

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Substring -> room type, tried in order (specific first) on the scene_id's room token.
_ROOM_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("masterbedroom",   "bedroom"),
    ("secondbedroom",   "bedroom"),
    ("kidsbedroom",     "bedroom"),
    ("bedroom",         "bedroom"),
    ("livingdining",    "living_room"),
    ("livingroom",      "living_room"),
    ("dining",          "dining_room"),
    ("kitchen",         "kitchen"),
    ("bathroom",        "bathroom"),
    ("study",           "study"),
    ("library",         "library"),
    ("office",          "office"),
    ("hallway",         "hallway"),
    ("corridor",        "hallway"),
    ("storage",         "storage"),
    ("balcony",         "balcony"),
]

# 3D-FRONT scene_id: "<uuid>_<RoomType>-<n>", e.g. "2e82690f-..._SecondBedroom-2658".
_SCENE_ID_ROOM_RE = re.compile(r"_([A-Za-z]+)-\d+$")


def _infer_room_type(scene_id: str) -> str:
    """Extract and normalise the room type embedded in a scene_id."""
    m = _SCENE_ID_ROOM_RE.search(scene_id)
    if not m:
        return "unknown"
    token = m.group(1).lower()
    for pattern, room_type in _ROOM_TYPE_PATTERNS:
        if pattern in token:
            return room_type
    return "unknown"


def load_scene_records(path: Path) -> list[SceneRecord]:
    """Parse one M3DLayout JSON file into SceneRecords."""
    if not path.exists():
        raise FileNotFoundError(f"Scene DB not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    source = raw.get("source", "unknown")
    records = [
        SceneRecord(
            scene_id=s.get("scene_id", ""),
            source_dataset=source,
            room_type=_infer_room_type(s.get("scene_id", "")),
            objects=s.get("objects", []),
            style_tags=[],  # 3D-Front source has no style tags
            similarity_score=0.0,
        )
        for s in raw.get("scenes", [])
    ]
    logger.info("loaded %d scene records from %s", len(records), path)
    return records

_PREVIEW_SUBDIR = Path("extracted") / "3dfront" / "3dfront_postprocessed_5173"
_PREVIEW_FILENAME = "rendered_scene_256.png"


def to_minimal_string(scene: SceneRecord) -> str:
    """Embedding string, e.g. "bedroom | nightstand(2) bed(1) wardrobe(1)" (count desc, then name)."""
    counts: Counter[str] = Counter(
        obj["category"]
        for obj in scene.objects
        if "category" in obj
    )
    cat_str = " ".join(
        f"{cat}({cnt})"
        for cat, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    )
    return f"{scene.room_type} | {cat_str}" if cat_str else scene.room_type


def to_rich_json(scene: SceneRecord) -> dict[str, Any]:
    """Spatial summary for the zoner prompt, largest footprint (the anchor) first."""
    # M3DLayout: location = [centre_x, height, centre_z] (y repeats size[1]), size = [w, h, d].
    objects = []
    for obj in scene.objects:
        cat = obj.get("category", "unknown")
        loc = obj.get("location", [0.0, 0.0, 0.0])
        sz = obj.get("size", [0.0, 0.0, 0.0])
        rot = obj.get("rotation", 0.0)
        objects.append(
            {
                "category": cat,
                "centre_xz": [round(loc[0], 3), round(loc[2], 3)],
                "size_xz": [round(sz[0], 3), round(sz[2], 3)],
                "height": round(sz[1], 3),
                "rotation_y": round(rot, 3) if isinstance(rot, (int, float)) else 0.0,
            }
        )

    objects.sort(key=lambda o: -(o["size_xz"][0] * o["size_xz"][1]))

    return {
        "scene_id": scene.scene_id,
        "room_type": scene.room_type,
        "object_count": len(objects),
        "objects": objects,
    }


def image_path_for_scene(scene_id: str, m3dlayout_root: Path) -> Path | None:
    """Top-down 3D-FRONT render of a scene, or None if absent."""
    candidate = m3dlayout_root / _PREVIEW_SUBDIR / scene_id / _PREVIEW_FILENAME
    return candidate if candidate.exists() else None

# Qwen/Qwen3-Embedding-0.6B at the revision scripts/download/download_models.sh fetches.
_DEFAULT_MODEL = str(CHECKPOINTS / "Qwen3-Embedding-0.6B")
_DEFAULT_BATCH = 64


def _resolve_device() -> str:
    """Encoder device: CUDA if available, else CPU (sentence-transformers given None may stay on CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Qwen3Encoder:
    """Qwen3-Embedding text encoder (L2-normalised float32); the model loads on first encode()."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if self._device is None:
            self._device = _resolve_device()
        logger.info("Loading encoder: %s (device=%s)", self.model_name, self._device)
        self._model = SentenceTransformer(self.model_name, device=self._device)
        if self._device == "cpu":
            logger.info("Encoder on CPU — batch encoding will be slow for large corpora.")

    def encode(
        self,
        texts: list[str],
        batch_size: int = _DEFAULT_BATCH,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (len(texts), D)."""
        self._load()
        assert self._model is not None
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return result.astype(np.float32)

class EmbeddingIndexIntegrityError(ValueError):
    """The persisted index violates a structural or identity invariant."""


class EmbeddingIndex:
    """Cosine index over (N, D) L2-normalised scene embeddings and their N metadata dicts."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise EmbeddingIndexIntegrityError(
                f"embeddings rows ({embeddings.shape[0]}) != metadata length ({len(metadata)})"
            )

        scene_rows: dict[str, list[int]] = {}
        for row, item in enumerate(metadata):
            scene_id = str(item.get("scene_id", ""))
            if not scene_id:
                raise EmbeddingIndexIntegrityError(
                    f"embedding metadata row {row} has an empty scene_id"
                )
            scene_rows.setdefault(scene_id, []).append(row)
        duplicates = {
            scene_id: rows for scene_id, rows in scene_rows.items() if len(rows) > 1
        }
        if duplicates:
            sample = ", ".join(
                f"{scene_id!r}: rows={rows}"
                for scene_id, rows in list(duplicates.items())[:5]
            )
            raise EmbeddingIndexIntegrityError(
                f"Duplicate scene_id in embedding index ({len(duplicates)} IDs): {sample}"
            )
        self._embeddings = embeddings.astype(np.float32)
        self._metadata = metadata

        self._room_index: dict[str, list[int]] = {}
        for i, m in enumerate(metadata):
            rt = m.get("room_type", "unknown")
            self._room_index.setdefault(rt, []).append(i)

        logger.info(
            "EmbeddingIndex loaded: %d scenes, dim=%d", len(metadata), self._embeddings.shape[1]
        )

    @classmethod
    def from_dir(cls, out_dir: Path) -> "EmbeddingIndex":
        """Load the index written to a directory by build_embeddings.py."""
        emb_path = out_dir / "embeddings.npy"
        meta_path = out_dir / "metadata.json"

        for p in (emb_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Embedding artefact not found: {p}. "
                    "Run scripts/build_index/build_retrieval_index.sh first."
                )

        embeddings = np.load(str(emb_path))
        metadata: list[dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(embeddings, metadata)

    @classmethod
    def from_dirs(cls, out_dirs: list[Path]) -> "EmbeddingIndex":
        """Load and merge the index dirs that contain embeddings.npy."""
        existing = [d for d in out_dirs if (d / "embeddings.npy").exists()]
        if not existing:
            raise FileNotFoundError(
                f"No embedding artefacts found in any of: {[str(d) for d in out_dirs]}"
            )
        parts = [cls.from_dir(d) for d in existing]
        if len(parts) == 1:
            return parts[0]

        dims = {p.dim for p in parts}
        if len(dims) != 1:
            raise EmbeddingIndexIntegrityError(
                f"Cannot merge indices with differing dims: {dims}"
            )

        embeddings = np.concatenate([p._embeddings for p in parts], axis=0)
        metadata: list[dict[str, Any]] = []
        for p in parts:
            metadata.extend(p._metadata)
        return cls(embeddings, metadata)

    @property
    def dim(self) -> int:
        return int(self._embeddings.shape[1])

    @property
    def n_scenes(self) -> int:
        return int(self._embeddings.shape[0])

    @property
    def room_types(self) -> list[str]:
        return list(self._room_index.keys())

    def query(
        self,
        query_vec: np.ndarray,
        room_type: str | None,
        k: int,
    ) -> list[tuple[int, float]]:
        """Top-k (row, cosine score) pairs, best first, within ``room_type`` if it has scenes."""
        if room_type is not None:
            candidate_indices = self._room_index.get(room_type, [])
            if not candidate_indices:
                logger.warning(
                    "room_type='%s' has no indexed scenes; searching the full index.",
                    room_type,
                )
                candidate_indices = list(range(self.n_scenes))
        else:
            candidate_indices = list(range(self.n_scenes))

        if not candidate_indices:
            return []

        pool = self._embeddings[candidate_indices]
        q = query_vec.astype(np.float32).ravel()
        scores = pool @ q  # cosine, as rows are L2-normalised

        actual_k = min(k, len(candidate_indices))
        if actual_k < len(candidate_indices):
            top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        else:
            top_idx = np.arange(len(candidate_indices))

        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            (candidate_indices[int(i)], float(scores[i]))
            for i in top_idx
        ]

    def get_metadata(self, row_index: int) -> dict[str, Any]:
        return self._metadata[row_index]

def _rich_obj_to_raw(obj: dict) -> dict:
    """Invert to_rich_json for one object, back to M3DLayout location/size/rotation."""
    cx, cz = obj.get("centre_xz", [0.0, 0.0])
    wx, wz = obj.get("size_xz", [0.0, 0.0])
    hy = obj.get("height", 0.0)
    return {
        "category": obj.get("category", "unknown"),
        "location": [cx, hy, cz],
        "size": [wx, hy, wz],
        "rotation": obj.get("rotation_y", 0.0),
    }


class EmbeddingSceneRetriever:
    """SceneSpec -> top-K SceneRecords: room-type filter, then Qwen3 cosine similarity."""

    def __init__(
        self,
        index: "EmbeddingIndex",
        encoder: "Qwen3Encoder",
        top_k: int = 3,
    ) -> None:
        self._index = index
        self._encoder = encoder
        self._top_k = top_k

    def retrieve(
        self,
        spec: SceneSpec,
        k: int | None = None,
    ) -> list[SceneRecord]:
        """Top-k SceneRecords for ``spec``, best first; ``k`` defaults to top_k."""
        effective_k = k if k is not None else self._top_k
        if effective_k <= 0:
            raise ValueError(f"k must be positive, got {effective_k}")

        query_str = self._spec_to_query_string(spec)
        room_type = self._primary_room_type(spec)

        logger.debug(
            "EmbeddingSceneRetriever: query=%r room_type=%s k=%d",
            query_str,
            room_type,
            effective_k,
        )

        query_vec = self._encoder.encode([query_str])[0]
        hits = self._index.query(query_vec, room_type, effective_k)

        records: list[SceneRecord] = []
        for row_idx, score in hits:
            meta = self._index.get_metadata(row_idx)
            records.append(self._meta_to_record(meta, score))

        return records

    @property
    def top_k(self) -> int:
        return self._top_k

    @staticmethod
    def _primary_room_type(spec: SceneSpec) -> str | None:
        if spec.room_constraints:
            return spec.room_constraints[0].room_type
        return None

    @staticmethod
    def _spec_to_query_string(spec: SceneSpec) -> str:
        """Query in the to_minimal_string format, so it shares the index's embedding space."""
        if not spec.room_constraints:
            return spec.scene_type or "room"
        rc = spec.room_constraints[0]
        room_type = rc.room_type
        cats = rc.required_categories
        if cats:
            return f"{room_type} | {' '.join(cats)}"
        return room_type

    @staticmethod
    def _meta_to_record(meta: dict, score: float) -> SceneRecord:
        """Reconstruct a SceneRecord from embedding index metadata."""
        rich = meta.get("rich_json", {})
        raw_objects = [
            _rich_obj_to_raw(obj) for obj in rich.get("objects", [])
        ]
        return SceneRecord(
            scene_id=meta["scene_id"],
            source_dataset="3D-Front",
            room_type=meta["room_type"],
            objects=raw_objects,
            style_tags=[],
            similarity_score=score,
        )

_MP3D_REGION_RE = re.compile(r"^(?P<building>.+)_region\d+$")


def source_building_id(scene_id: str) -> str:
    """Return a stable source-scene/building key for diversity preference."""
    match = _MP3D_REGION_RE.match(scene_id)
    if match:
        return match.group("building")
    # 3D-FRONT: the uuid of "<uuid>_<RoomType>-<n>" names the source house.
    return scene_id.split("_", 1)[0]


def derive_retrieval_seed(global_seed: int, spec: SceneSpec) -> int:
    """Derive a stable 64-bit retrieval seed."""
    room_type = spec.room_constraints[0].room_type if spec.room_constraints else ""
    categories = (
        sorted(spec.room_constraints[0].required_categories)
        if spec.room_constraints
        else []
    )
    payload = "|".join(
        [
            str(global_seed),
            room_type,
            ",".join(categories),
            spec.prompt_raw.strip(),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _deduplicate_ranked(records: list[SceneRecord]) -> list[SceneRecord]:
    unique: list[SceneRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.scene_id in seen:
            continue
        seen.add(record.scene_id)
        unique.append(record)
    return unique


def _weighted_choice(
    candidates: list[tuple[int, SceneRecord]],
    *,
    best_score: float,
    temperature: float,
    rng: random.Random,
) -> tuple[int, SceneRecord]:
    weights = [
        math.exp((float(record.similarity_score) - best_score) / temperature)
        for _, record in candidates
    ]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if threshold <= cumulative:
            return candidate
    return candidates[-1]


def select_exemplars(
    ranked_records: list[SceneRecord],
    spec: SceneSpec,
    *,
    context_n: int,
    candidate_k: int,
    global_seed: int,
    temperature: float = 0.04,
    max_score_drop: float = 0.08,
) -> list[SceneRecord]:
    """Select the exemplar context: rank 1 plus seeded draws, in similarity order."""
    if context_n <= 0:
        raise ValueError(f"context_n must be positive, got {context_n}")
    if candidate_k < context_n:
        raise ValueError(
            f"candidate_k ({candidate_k}) must be >= context_n ({context_n})"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if max_score_drop < 0:
        raise ValueError(f"max_score_drop must be non-negative, got {max_score_drop}")

    candidates = _deduplicate_ranked(ranked_records[:candidate_k])
    if not candidates:
        return []

    actual_n = min(context_n, len(candidates))
    best_score = float(candidates[0].similarity_score)
    cutoff = best_score - max_score_drop
    eligible = [record for record in candidates if record.similarity_score >= cutoff]
    if len(eligible) < actual_n:
        eligible = candidates[:actual_n]

    ranked_eligible = list(enumerate(eligible, start=1))
    retrieval_seed = derive_retrieval_seed(global_seed, spec)

    if actual_n == 1:
        selected_ranked = ranked_eligible[:actual_n]
    else:
        rng = random.Random(retrieval_seed)
        selected_ranked = [ranked_eligible[0]]
        remaining = ranked_eligible[1:]
        selected_buildings = {source_building_id(ranked_eligible[0][1].scene_id)}

        while len(selected_ranked) < actual_n and remaining:
            diverse_pool = [
                item
                for item in remaining
                if source_building_id(item[1].scene_id) not in selected_buildings
            ]
            choice_pool = diverse_pool or remaining
            chosen = _weighted_choice(
                choice_pool,
                best_score=best_score,
                temperature=temperature,
                rng=rng,
            )
            selected_ranked.append(chosen)
            remaining.remove(chosen)
            selected_buildings.add(source_building_id(chosen[1].scene_id))

        # Only the set is random; the prompt keeps similarity order.
        selected_ranked.sort(key=lambda item: item[0])

    return [record for _, record in selected_ranked]
