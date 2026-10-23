"""Genesis scene of a ``SceneLayout`` and its top-down and oblique renders."""

# scenebloom is y-up (floor on x-z); Genesis is z-up (floor on x-y), so a Genesis
# position is (center[0], center[2], height).

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenebloom.utils.schemas import PlacedObject, SceneLayout
from scenebloom.render.materials import shell_textures
from scenebloom.utils.assets import object_mesh_pose

logger = logging.getLogger(__name__)


RENDER_RES: tuple[int, int] = (1024, 1024)
TOP_DOWN_FOV_DEG: float = 84.0

OBLIQUE_FOV_DEG: float = 65.0
TOP_DOWN_HEIGHT_M: float = 3.8
WALL_INSET_M: float = 0.1   # camera offset from wall to avoid clipping

# Shell surfaces are grid quads whose UVs run 0 -> 1 -> 0 ..., so the texture
# repeats SHELL_TEXTURE_TILES times per side, mirrored, without seams.
SHELL_GRID_VERTS: int = 64
SHELL_TEXTURE_TILES: int = 4


def _cache_dir() -> Path:
    """Floor/wall OBJ cache under the system temp dir; safe to delete."""
    root = Path(tempfile.gettempdir()) / "scenebloom_genesis_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


_GS_INITIALIZED: bool = False


def _init_genesis_once(seed: int = 0) -> None:
    """Call ``gs.init`` once per process; a second init can crash Genesis."""
    global _GS_INITIALIZED
    if _GS_INITIALIZED:
        return
    import genesis as gs

    _patch_genesis_igl_signed_distance()
    try:
        gs.init(seed=seed, precision="32", logging_level="warning", backend=gs.gpu)
    except RuntimeError as exc:
        logger.debug("genesis already initialised: %s", exc)
    _GS_INITIALIZED = True


_IGL_PATCHED = False


def _patch_genesis_igl_signed_distance() -> None:
    """Make Genesis 0.2.1 accept libigl's 4-value ``signed_distance`` return.

    Genesis star-unpacks 3 values, which fails on newer libigl; the patch indexes instead.
    """
    global _IGL_PATCHED
    if _IGL_PATCHED:
        return
    try:
        import igl
        import numpy as _np
        from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom

        def _compute_sd(self, query_points):
            return igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)[0]

        def _compute_closest_verts(self, query_points):
            closest_faces = igl.signed_distance(
                query_points, self._init_verts, self._init_faces
            )[1]
            verts_ids = self._init_faces[closest_faces]
            verts_ids = verts_ids[
                _np.arange(len(query_points)).astype(int),
                _np.argmin(
                    _np.linalg.norm(
                        self._init_verts[verts_ids] - query_points[:, None, :], axis=-1
                    ),
                    axis=-1,
                ),
            ]
            return verts_ids

        RigidGeom._compute_sd = _compute_sd
        RigidGeom._compute_closest_verts = _compute_closest_verts
        logger.debug("patched genesis RigidGeom signed_distance unpacking")
    except Exception as exc:
        logger.warning("genesis RigidGeom signed_distance left as installed: %s", exc)
    _IGL_PATCHED = True


# Column order that embeds a quad's local (u, v, 0) grid in the Genesis frame.
_QUAD_AXES: dict[str, list[int]] = {
    "floor": [0, 1, 2],    # (u, v, 0): horizontal; u along x, v along y
    "y_wall": [0, 2, 1],   # (u, 0, v): constant-y wall; u along x, v up
    "x_wall": [2, 1, 0],   # (0, v, u): constant-x wall; u up, v along y
}


def _mirrored_ramp(n: int, tiles: int) -> np.ndarray:
    """``n`` texture coordinates sweeping 0 -> 1 -> 0 ... over ``tiles`` copies."""
    half = n // tiles
    phase = np.arange(n) % (2 * half)
    return np.where(phase < half, phase, 2 * half - phase) / half


def _grid_faces(n: int, reverse: bool) -> np.ndarray:
    """Triangles of an ``n`` x ``n`` vertex grid, CCW seen from local +z unless ``reverse``."""
    k = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    lower = [k, k + 1, k + n + 1]
    upper = [k, k + n + 1, k + n]
    if reverse:
        lower = [lower[0], lower[2], lower[1]]
        upper = [upper[0], upper[2], upper[1]]
    pairs = np.stack([np.stack(lower, axis=1), np.stack(upper, axis=1)], axis=1)
    return pairs.reshape(-1, 3)


def _write_shell_quad(
    path: Path,
    plane: str,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    reverse: bool = False,
) -> None:
    """Write one shell surface as a UV-mapped OBJ grid quad in the Genesis frame."""
    import trimesh

    n = SHELL_GRID_VERTS
    u = np.linspace(u_span[0], u_span[1], n)
    v = np.linspace(v_span[0], v_span[1], n)
    local = np.column_stack([np.tile(u, n), np.repeat(v, n), np.zeros(n * n)])
    ramp = _mirrored_ramp(n, SHELL_TEXTURE_TILES)
    mesh = trimesh.Trimesh(
        vertices=local[:, _QUAD_AXES[plane]], faces=_grid_faces(n, reverse)
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.column_stack([np.tile(ramp, n), np.repeat(ramp, n)])
    )
    mesh.export(str(path))


def _shell_quad_path(
    kind: str,
    edges: tuple[float, float, float, float],
    *,
    height: float | None = None,
    reverse: bool = False,
) -> Path:
    """Cache path of a shell quad, keyed by its room edges (1 cm) and winding."""
    key = "_".join(f"{edge:.2f}" for edge in edges)
    if height is not None:
        key += f"_h{height:.2f}"
    suffix = "_rev" if reverse else ""
    return _cache_dir() / f"shell_{kind}_{key.replace('-', 'n')}{suffix}.obj"


def _ensure_shell_quad(
    path: Path,
    plane: str,
    u_span: tuple[float, float],
    v_span: tuple[float, float],
    reverse: bool = False,
) -> Path:
    """Write the quad unless a cached file already exists; return its path."""
    if not path.exists():
        _write_shell_quad(path, plane, u_span, v_span, reverse)
    return path


def _add_shell_entity(
    scene: Any,
    quad: Path,
    origin: tuple[float, float, float],
    texture: Path,
    *,
    finish: str,
    collision: bool,
    double_sided: bool,
) -> Any:
    """Add one fixed, textured shell surface ("plastic" or "rough" finish)."""
    import genesis as gs

    material = gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16)
    morph = gs.morphs.Mesh(
        collision=collision,
        file=str(quad),
        pos=origin,
        euler=(0, 0, 0),
        scale=1.0,
        fixed=True,
    )
    image = gs.textures.ImageTexture(image_path=str(texture))
    if finish == "plastic":
        surface = gs.surfaces.Plastic(
            diffuse_texture=image, roughness=10, double_sided=double_sided
        )
    else:
        surface = gs.surfaces.Rough(diffuse_texture=image, double_sided=double_sided)
    return scene.add_entity(material=material, morph=morph, surface=surface)


def _add_floor_and_ceiling(
    scene: Any,
    x_l: float,
    x_r: float,
    y_l: float,
    y_r: float,
    floor_texture: Path,
    ceiling_texture: Path,
    ceiling_z: float,
) -> Any:
    """Visual-only floor at z=0 and ceiling at ``ceiling_z``; returns the floor.

    The ceiling is single-sided and faces down, so the top-down camera sees through it.
    """
    edges = (x_l, x_r, y_l, y_r)
    floor_quad = _ensure_shell_quad(
        _shell_quad_path("floor", edges), "floor", (x_l, x_r), (y_l, y_r)
    )
    ceiling_quad = _ensure_shell_quad(
        _shell_quad_path("floor", edges, reverse=True),
        "floor", (x_l, x_r), (y_l, y_r), reverse=True,
    )
    floor = _add_shell_entity(
        scene, floor_quad, (0, 0, 0), floor_texture,
        finish="plastic", collision=False, double_sided=True,
    )
    _add_shell_entity(
        scene, ceiling_quad, (0, 0, ceiling_z), ceiling_texture,
        finish="plastic", collision=False, double_sided=False,
    )
    return floor


@dataclass(frozen=True)
class _ShellWall:
    """One wall quad of the room rectangle, in Genesis coordinates."""

    side: str
    plane: str                                  # "x_wall" or "y_wall"
    edges: tuple[float, float, float, float]    # cache key: x0, x1, y0, y1
    u_span: tuple[float, float]
    v_span: tuple[float, float]
    origin: tuple[float, float, float]


def _shell_walls(
    x_l: float, x_r: float, y_l: float, y_r: float, height: float
) -> list[_ShellWall]:
    """West, south, east and north walls, in that order."""
    return [
        _ShellWall("west", "x_wall", (x_l, x_l, y_l, y_r),
                   (0, height), (y_l, y_r), (x_l, 0, 0)),
        _ShellWall("south", "y_wall", (x_l, x_r, y_l, y_l),
                   (x_l, x_r), (0, height), (0, y_l, 0)),
        _ShellWall("east", "x_wall", (x_r, x_r, y_l, y_r),
                   (0, height), (y_l, y_r), (x_r, 0, 0)),
        _ShellWall("north", "y_wall", (x_l, x_r, y_r, y_r),
                   (x_l, x_r), (0, height), (0, y_r, 0)),
    ]


def _add_shell_walls(
    scene: Any,
    x_l: float,
    x_r: float,
    y_l: float,
    y_r: float,
    height: float,
    texture: Path,
) -> list[Any]:
    """Add the four double-sided wall entities; returned in ``_shell_walls`` order."""
    entities = []
    for wall in _shell_walls(x_l, x_r, y_l, y_r, height):
        quad = _ensure_shell_quad(
            _shell_quad_path("wall", wall.edges, height=height),
            wall.plane, wall.u_span, wall.v_span,
        )
        entities.append(_add_shell_entity(
            scene, quad, wall.origin, texture,
            finish="rough", collision=False, double_sided=True,
        ))
    return entities


def _add_placed_object(
    scene: Any,
    placed: PlacedObject,
) -> Any | None:
    """Add one PlacedObject as a Genesis mesh entity; None when its category has no GLB."""
    import genesis as gs

    mesh_pose = object_mesh_pose(placed)
    if mesh_pose is None:
        return None
    try:
        entity = scene.add_entity(
            material=gs.materials.Rigid(sdf_min_res=16, sdf_max_res=16),
            morph=gs.morphs.Mesh(
                file=str(mesh_pose.glb_path),
                pos=mesh_pose.pos,
                euler=mesh_pose.euler,
                scale=mesh_pose.scale,
                collision=False,
                fixed=placed.placement_method.startswith("wall_proposer"),
                convexify=True,
                coacd_options=None,
            ),
        )
    except Exception:
        logger.exception(
            "_add_placed_object: failed for %s (asset_id=%s)",
            placed.instance_id, mesh_pose.asset_id,
        )
        raise
    return entity


def _oblique_pose(
    x_l: float, x_r: float, y_l: float, y_r: float, ceiling_h: float
) -> tuple[list[float], list[float]]:
    """(eye, target): below the ceiling in the (-x, -y) corner, looking at the room centre."""
    h = max(0.5, ceiling_h * 0.97)
    eye = [x_l + WALL_INSET_M, y_l + WALL_INSET_M, h]
    return eye, [(x_l + x_r) / 2, (y_l + y_r) / 2, 0.4]


def _add_scene_cameras(
    scene: Any,
    x_l: float,
    x_r: float,
    y_l: float,
    y_r: float,
    ceiling_h: float,
) -> tuple[Any, Any, tuple[list[float], list[float]]]:
    """Add the top-down and oblique cameras; returns ``(top, oblique, oblique_pose)``."""
    def camera(pos: Any, lookat: Any, fov: float) -> Any:
        return scene.add_camera(
            pos=tuple(pos), lookat=tuple(lookat), res=RENDER_RES, fov=fov, GUI=False,
        )

    centre = ((x_l + x_r) / 2, (y_l + y_r) / 2)
    top_cam = camera(
        (*centre, _top_down_height(x_l, x_r, y_l, y_r)), (*centre, 0.0), TOP_DOWN_FOV_DEG,
    )
    oblique_pose = _oblique_pose(x_l, x_r, y_l, y_r, ceiling_h)
    oblique_cam = camera(*oblique_pose, OBLIQUE_FOV_DEG)
    return top_cam, oblique_cam, oblique_pose


# Four diagonal lights from above; pyrender renders at most 4 lights.
STUDIO_LIGHT_DIRS: tuple[tuple[int, int, int], ...] = (
    (-1, -1, -1), (1, 1, -1), (1, -1, -1), (-1, 1, -1),
)


STUDIO_AMBIENT: float = 0.5
STUDIO_LIGHT_INTENSITY: float = 3.0


def studio_vis_options(gs: Any) -> Any:
    """Shadow-free, even studio lighting shared by every Genesis render."""
    return gs.options.VisOptions(
        shadow=False,
        ambient_light=(STUDIO_AMBIENT,) * 3,
        lights=[
            {"type": "directional", "dir": direction,
             "color": (1.0, 1.0, 1.0), "intensity": STUDIO_LIGHT_INTENSITY}
            for direction in STUDIO_LIGHT_DIRS
        ],
    )


def _scene_options(gs: Any, *, headless: bool) -> dict[str, Any]:
    """Keyword arguments for ``gs.Scene``: a static render, no gravity or collisions."""
    vis = studio_vis_options(gs)
    return {
        "show_viewer": not headless,
        "renderer": gs.renderers.Rasterizer(),
        "sim_options": gs.options.SimOptions(),
        "rigid_options": gs.options.RigidOptions(gravity=(0, 0, 0), enable_collision=False),
        "vis_options": vis,
    }


def _top_down_height(x_l: float, x_r: float, y_l: float, y_r: float) -> float:
    """Top-down camera height that fits the whole floor in the FOV, never below the default."""
    half_extent = max(x_r - x_l, y_r - y_l) / 2.0
    fit = 1.05 * half_extent / np.tan(np.radians(TOP_DOWN_FOV_DEG / 2.0))
    return max(TOP_DOWN_HEIGHT_M, fit)


def _room_extents_from_layout(
    layout: SceneLayout,
) -> tuple[float, float, float, float, float]:
    """(x_min, x_max, z_min, z_max, ceiling_h) of the first room."""
    bbox = layout.zoning_plan.rooms[0].bbox
    return (
        bbox.min_x,
        bbox.max_x,
        bbox.min_z,
        bbox.max_z,
        layout.zoning_plan.ceiling_h,
    )


def build_genesis_scene(
    layout: SceneLayout,
    *,
    headless: bool = True,
    seed: int = 0,
    material_assignment: Path,
) -> tuple[Any, tuple[float, float, float, float, float]]:
    """Build the Genesis scene of ``layout``: textured room shell, object meshes, cameras.

    ``material_assignment`` is the JSON whose floor and wall ``base_color`` texture the
    shell; ``seed`` is used only by the first init in a process. Returns
    ``(scene, (x_min, x_max, z_min, z_max, ceiling_h))``.
    """
    import genesis as gs

    floor_texture, wall_texture = shell_textures(material_assignment)

    _init_genesis_once(seed)

    extents = _room_extents_from_layout(layout)
    x_l, x_r, z_l, z_r, ceiling_h = extents

    scene = gs.Scene(**_scene_options(gs, headless=headless))

    _add_floor_and_ceiling(
        scene,
        x_l, x_r, z_l, z_r,
        floor_texture=floor_texture,
        ceiling_texture=wall_texture,
        ceiling_z=ceiling_h,
    )
    _add_shell_walls(scene, x_l, x_r, z_l, z_r, ceiling_h, wall_texture)

    n_added = sum(
        _add_placed_object(scene, placed) is not None for placed in layout.placed_objects
    )

    cam_top, cam_oblique, oblique_pose = _add_scene_cameras(
        scene, x_l, x_r, z_l, z_r, ceiling_h
    )
    # render_scene reads the cameras and the oblique pose from the scene.
    scene._cam_top = cam_top
    scene._cam_oblique = cam_oblique
    scene._oblique_pose = oblique_pose

    scene.build(n_envs=0)

    logger.info(
        "build_genesis_scene: %d objects placed, room=(%.2fx%.2f m), ceiling=%.2f m",
        n_added, x_r - x_l, z_r - z_l, ceiling_h,
    )
    return scene, extents


def _refresh_viewer(scene: Any) -> None:
    """Push pending camera poses to the renderer (no-op without a visualizer)."""
    viewer = getattr(scene, "visualizer", None)
    if viewer is not None:
        viewer.update()


# Some processes read back an all-zero frame every other render; a lit room is
# never all zero, so such a frame is shot again.
RENDER_ATTEMPTS: int = 4


def _render_lit(cam: Any, **outputs: bool) -> tuple[Any, ...]:
    """``cam.render(rgb=True, **outputs)``, repeated while the RGB is all zero."""
    for attempt in range(1, RENDER_ATTEMPTS + 1):
        frame = cam.render(rgb=True, **outputs)
        if np.any(frame[0]):
            return frame
        logger.warning("camera %s: all-zero frame on attempt %d", cam.idx, attempt)
    raise RuntimeError(
        f"camera {cam.idx}: {RENDER_ATTEMPTS} renders in a row read back all zero"
    )


def render_scene(
    scene: Any,
    room_extents: tuple[float, float, float, float, float],
    out_dir: Path,
) -> list[Path]:
    """Write ``top_down.png`` and ``oblique.png`` into ``out_dir``; returns their paths."""
    import imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_l, x_r, z_l, z_r, _ = room_extents

    # Re-pose each camera right before its shot.
    centre = ((x_l + x_r) / 2, (z_l + z_r) / 2)
    shots = (
        ("top_down", scene._cam_top,
         ((*centre, _top_down_height(x_l, x_r, z_l, z_r)), (*centre, 0.0))),
        ("oblique", scene._cam_oblique, scene._oblique_pose),
    )
    written: list[Path] = []
    for name, cam, (pos, lookat) in shots:
        cam.set_pose(pos=tuple(pos), lookat=tuple(lookat))
        _refresh_viewer(scene)
        path = out_dir / f"{name}.png"
        imageio.imwrite(path, _render_lit(cam)[0])
        written.append(path)

    logger.info("render_scene: wrote %d files to %s", len(written), out_dir)
    return written
