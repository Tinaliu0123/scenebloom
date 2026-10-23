"""Text-to-mesh with TRELLIS-text-xlarge: a textured, y-up, unit-cube GLB (TRELLIS env).

usage: python asset_generation/mesh_generation/trellis_text_to_glb.py <name> "<prompt>"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_SOURCE = REPO_ROOT / "third_party" / "TRELLIS"

OUT_DIR = REPO_ROOT / "outputs" / "meshes"
MODEL_ID = "microsoft/TRELLIS-text-xlarge"
SIMPLIFY = 0.95  # fraction of triangles removed
TEXTURE_SIZE = 1024


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="mesh name; the output is outputs/meshes/<name>.glb")
    p.add_argument("prompt", help="text description of the object")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.environ.setdefault("HF_HUB_CACHE", str(REPO_ROOT / "checkpoints" / "huggingface"))
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(TRELLIS_SOURCE))
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisTextTo3DPipeline.from_pretrained(MODEL_ID)
    pipeline.cuda()
    outputs = pipeline.run(args.prompt, seed=args.seed)
    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0],
                                      simplify=SIMPLIFY, texture_size=TEXTURE_SIZE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.name}.glb"
    glb.export(str(path))
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
