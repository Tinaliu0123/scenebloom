"""Genesis renderer of a scene layout; ``genesis`` is optional at import time."""

from __future__ import annotations

from pathlib import Path

from scenebloom.utils.schemas import SceneLayout

try:
    import genesis as _genesis  # type: ignore[import]
    _GENESIS_AVAILABLE = True
except ImportError:
    _genesis = None
    _GENESIS_AVAILABLE = False


class GenesisRenderer:
    """Holds one Genesis scene and writes its top-down and oblique renders."""

    def __init__(self, *, material_assignment: Path, headless: bool = True) -> None:
        if not _GENESIS_AVAILABLE:
            raise ImportError(
                "GenesisRenderer requires the 'genesis' package. "
                "Install it with envs/install_scenebloom.sh."
            )
        self._headless = headless
        self._material_assignment = material_assignment
        self._scene: object | None = None
        self._room_extents: tuple[float, float, float, float, float] | None = None
        self._loaded_layout_id: str | None = None

    def load_scene(self, layout: SceneLayout) -> None:
        """Build the Genesis scene for ``layout``; a no-op if that layout is already loaded."""
        if layout.layout_id == self._loaded_layout_id:
            return

        from scenebloom.render.scene_builder import (
            build_genesis_scene,
        )

        self._scene, self._room_extents = build_genesis_scene(
            layout,
            headless=self._headless,
            material_assignment=self._material_assignment,
        )
        self._loaded_layout_id = layout.layout_id

    def render(self, out_dir: Path) -> list[Path]:
        """Render the loaded scene into ``out_dir``; returns the written paths."""
        if self._scene is None or self._room_extents is None:
            raise RuntimeError(
                "GenesisRenderer.render() called before load_scene(). "
                "Call load_scene(layout) first."
            )
        from scenebloom.render.scene_builder import render_scene

        return render_scene(self._scene, self._room_extents, out_dir)
