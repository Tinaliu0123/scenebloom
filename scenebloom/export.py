"""Export a generated scene as one textured ``scene.glb`` that Habitat loads as ``scene_id``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from scenebloom.render.materials import shell_textures
from scenebloom.utils.assets import effective_lookup_key, resolve_glb
from scenebloom.utils.schemas import BBox3D, PlacedObject, SceneLayout

FLOOR_THICKNESS = 0.05  # metres; the floor's top face is at y = 0
WALL_HEIGHT = 2.5
WALL_THICKNESS = 0.05
TEXTURE_TILE = 1.5  # metres per repeat of the floor and wall textures


def _textured(mesh: trimesh.Trimesh, image: Image.Image, u_axis: int, v_axis: int) -> trimesh.Trimesh:
    """Planar UVs from two world axes. glTF's default sampler repeats the texture."""
    uv = (mesh.vertices[:, [u_axis, v_axis]] / TEXTURE_TILE).astype(np.float32)
    material = PBRMaterial(baseColorTexture=image, baseColorFactor=[255, 255, 255, 255],
                           metallicFactor=0.0, roughnessFactor=1.0)
    mesh.visual = TextureVisuals(uv=uv, material=material)
    return mesh


def room_shell(room: BBox3D, floor_image: Image.Image,
               wall_image: Image.Image) -> list[tuple[str, trimesh.Trimesh]]:
    """The floor slab under the room box and one wall slab centred on each of its edges."""
    (cx, _, cz), (sx, _, sz) = room.center, room.size
    floor = trimesh.creation.box(extents=(sx, FLOOR_THICKNESS, sz))
    floor.apply_translation([cx, -FLOOR_THICKNESS / 2.0, cz])
    shell = [("floor", _textured(floor, floor_image, 0, 2))]

    hx, hz = sx / 2.0, sz / 2.0
    corners = [(cx - hx, cz - hz), (cx + hx, cz - hz), (cx + hx, cz + hz), (cx - hx, cz + hz)]
    for i, ((x0, z0), (x1, z1)) in enumerate(zip(corners, corners[1:] + corners[:1])):
        dx, dz = x1 - x0, z1 - z0
        wall = trimesh.creation.box(extents=(math.sqrt(dx * dx + dz * dz), WALL_HEIGHT, WALL_THICKNESS))
        wall.apply_transform(trimesh.transformations.rotation_matrix(math.atan2(-dz, dx), [0.0, 1.0, 0.0]))
        wall.apply_translation([(x0 + x1) / 2.0, WALL_HEIGHT / 2.0, (z0 + z1) / 2.0])
        lo, hi = wall.bounds
        along = 0 if hi[0] - lo[0] >= hi[2] - lo[2] else 2
        shell.append((f"wall_{i:02d}", _textured(wall, wall_image, along, 1)))
    return shell


def object_mesh(obj: PlacedObject) -> trimesh.Trimesh:
    """Library mesh scaled to the object's box, base at ``pose.center[1]``, yawed about +Y."""
    mesh = trimesh.load(str(resolve_glb(effective_lookup_key(obj.asset_id, obj.category))),
                        force="mesh")
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)
    size = np.asarray(obj.pose.size, dtype=float)
    extent = hi - lo
    scale = np.where(extent > 1e-6, size / extent, 1.0)
    mesh.apply_transform(np.diag([scale[0], scale[1], scale[2], 1.0]))
    mesh.apply_translation([0.0, size[1] / 2.0, 0.0])
    c, s = math.cos(obj.pose.rotation_y), math.sin(obj.pose.rotation_y)
    x, y, z = obj.pose.center
    mesh.apply_transform(np.array([[c, 0.0, s, x], [0.0, 1.0, 0.0, y],
                                   [-s, 0.0, c, z], [0.0, 0.0, 0.0, 1.0]]))
    return mesh


def export_scene(layout: SceneLayout, material_assignment: Path, out_glb: Path) -> None:
    floor_texture, wall_texture = shell_textures(material_assignment)
    scene = trimesh.Scene()
    for name, mesh in room_shell(layout.zoning_plan.rooms[0].bbox,
                                 Image.open(floor_texture).convert("RGB"),
                                 Image.open(wall_texture).convert("RGB")):
        scene.add_geometry(mesh, geom_name=name)
    for obj in layout.placed_objects:
        scene.add_geometry(object_mesh(obj), geom_name=f"object_{obj.instance_id}")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(file_obj=str(out_glb), file_type="glb")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scenebloom.export", description=__doc__)
    p.add_argument("scene_dir", type=Path, help="output directory of scenebloom.cli")
    p.add_argument("export_dir", type=Path, help="receives scene.glb")
    args = p.parse_args(argv)

    layout = SceneLayout.model_validate_json(
        (args.scene_dir / "scene_layout.json").read_text(encoding="utf-8"))
    out_glb = args.export_dir / "scene.glb"
    export_scene(layout, args.scene_dir / "material_assignment.json", out_glb)
    print(f"exported: {out_glb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
