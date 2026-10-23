"""Pipeline configuration and its defaults."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from scenebloom.utils.paths import DATA


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_default=True)

    # Relative paths resolve against the working directory.
    assets_path: Path = Path("data/assets.json")
    output_dir: Path = Path("outputs")

    # The top exemplar is kept; the other contexts are sampled from the candidates by seed.
    retrieval_candidate_k: int = 12
    retrieval_context_n: int = 3
    retrieval_temperature: float = 0.04
    retrieval_max_score_drop: float = 0.08
    embed_index_dir: Path = Path("data/embeddings")
    # Matterport3D rooms, merged into the index; a missing directory is skipped.
    embed_index_extra_dirs: tuple[Path, ...] = (Path("data/embeddings_mp3d"),)
    m3dlayout_root: Path = DATA / "m3dlayout"

    llm_model: str = "qwen/qwen3-vl-8b-instruct"  # OpenRouter model id

    small_object_supports: str = (
        "desk,coffee_table,nightstand,side_table,dining_table,"
        "cabinet,tv_stand,chest_of_drawers,"
        "dressing_table"
    )
    small_object_max_candidates: int = 20
    # A retry re-queries only the unplaced assets, told the accepted (u, v) positions.
    small_object_retry_rounds: int = 2

    wall_object_max_per_zone: int = 4

    # AmbientCG floor/wall materials, retrieved per room type with CLIP as in SceneSmith.
    texture_dir: Path = Path("data/materials/ambientcg_full")
    texture_bank_path: Path = Path("data/materials/ambientcg_full_clip.npz")
    texture_top_k: int = 5
    texture_sample_top_k: int = 3

    random_seed: int = 42

    @field_validator("retrieval_candidate_k", "retrieval_context_n")
    @classmethod
    def _retrieval_counts_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("retrieval counts must be positive")
        return v

    @field_validator("retrieval_temperature")
    @classmethod
    def _retrieval_temperature_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("retrieval_temperature must be positive")
        return v

    @field_validator("retrieval_max_score_drop")
    @classmethod
    def _retrieval_score_drop_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("retrieval_max_score_drop must be non-negative")
        return v

    @field_validator("small_object_max_candidates")
    @classmethod
    def _small_object_candidates_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("small_object_max_candidates must be positive")
        return v

    @field_validator("wall_object_max_per_zone")
    @classmethod
    def _wall_object_cap_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("wall_object_max_per_zone must be positive")
        return v
