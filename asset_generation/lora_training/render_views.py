"""Render a library mesh from eight azimuths on white, the training views of its LoRA adapter.

usage: python asset_generation/lora_training/render_views.py <mesh.glb> <out_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh

from scenebloom.render.scene_builder import _init_genesis_once, _render_lit

AZIMUTHS_DEG = tuple(range(0, 360, 45))
RES = (1024, 1024)
FOV_DEG = 45.0
#: Camera distance and height, in units of the mesh's largest extent.
DISTANCE = 1.5
HEIGHT = 0.7
AMBIENT = 0.45
LIGHT_INTENSITY = 1.5
LIGHT_DIRS = ((-1, -1, -1), (1, 1, -1), (1, -1, -1), (-1, 1, -1))


def mesh_frame(glb: Path) -> tuple[np.ndarray, float]:
    """Centre of the mesh after the y-up to z-up turn, and its largest extent."""
    bounds = trimesh.load(glb, force="scene").bounds
    cx, cy, cz = bounds.mean(axis=0)
    # euler (90, 0, 0) maps the GLB's (x, y, z) to (x, -z, y).
    return np.array([cx, -cz, cy]), float((bounds[1] - bounds[0]).max())


def render_views(glb: Path, out_dir: Path) -> list[Path]:
    import genesis as gs

    _init_genesis_once()
    centre, extent = mesh_frame(glb)
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        rigid_options=gs.options.RigidOptions(gravity=(0, 0, 0), enable_collision=False),
        vis_options=gs.options.VisOptions(
            shadow=False,
            background_color=(1.0, 1.0, 1.0),
            ambient_light=(AMBIENT,) * 3,
            lights=[{"type": "directional", "dir": d, "color": (1.0, 1.0, 1.0),
                     "intensity": LIGHT_INTENSITY} for d in LIGHT_DIRS],
        ),
    )
    scene.add_entity(gs.morphs.Mesh(file=str(glb), euler=(90, 0, 0), fixed=True,
                                    collision=False))
    cam = scene.add_camera(res=RES, pos=tuple(centre + (DISTANCE * extent, 0, 0)),
                           lookat=tuple(centre), fov=FOV_DEG, GUI=False)
    scene.build()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for az in AZIMUTHS_DEG:
        a = np.deg2rad(az)
        offset = extent * np.array([DISTANCE * np.cos(a), DISTANCE * np.sin(a), HEIGHT])
        cam.set_pose(pos=tuple(centre + offset), lookat=tuple(centre))
        rgb = _render_lit(cam)[0]
        # A one-colour frame means the mesh is out of view.
        if np.asarray(rgb).std() == 0:
            raise RuntimeError(f"{glb.name}: blank view at azimuth {az}")
        path = out_dir / f"{glb.stem}_{az:03d}.png"
        imageio.imwrite(path, rgb)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("glb", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args(argv)
    for path in render_views(args.glb, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
