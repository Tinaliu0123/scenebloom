#!/usr/bin/env bash
# Download the data the pipeline reads at run time and verify it:
#   1. Object meshes and FLUX LoRA adapters -> data/lora_assets/, from the SceneBloom
#      Hugging Face repositories
#   2. AmbientCG floor and wall materials (CC0, ambientcg.com) listed in
#      data/materials/ambientcg_room_materials.json -> data/materials/ambientcg_full/
#   3. M3DLayout (CC BY-NC 4.0) -> data/m3dlayout/: the layouts the retrieval indexes are built
#      from (scripts/build_index/build_retrieval_index.sh) and the 3D-FRONT previews for zoning
#
# usage: bash scripts/download/download_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
# SceneBloom Hugging Face repositories, read at a fixed tag:
#   <namespace>/SceneBloom-Assets (dataset):     glb/*.glb
#   <namespace>/SceneBloom-AssetLoRAs (model):   adapters/<name>/pytorch_lora_weights.safetensors
SCENEBLOOM_HF_NAMESPACE="tina390"
SCENEBLOOM_HF_REVISION=v1.0
M3D=data/m3dlayout
M3DLAYOUT_REVISION=72a2a80c3b96788635b9192b64853e1428d59cb4

case "$SCENEBLOOM_HF_NAMESPACE" in
    "<"*) echo "SCENEBLOOM_HF_NAMESPACE in $0 is not filled in yet" >&2; exit 1 ;;
esac

# 1. Meshes and LoRA adapters -> data/lora_assets/
python - "$SCENEBLOOM_HF_NAMESPACE" "$SCENEBLOOM_HF_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

namespace, revision = sys.argv[1], sys.argv[2]
snapshot_download(f"{namespace}/SceneBloom-Assets", repo_type="dataset", revision=revision,
                  local_dir="data/lora_assets", allow_patterns=["glb/*"])
snapshot_download(f"{namespace}/SceneBloom-AssetLoRAs", revision=revision,
                  local_dir="data/lora_assets", allow_patterns=["adapters/*"])
PY
# The variant pool is built from the LoRA files present on disk, so a missing
# file changes which variants scenes can use; every listed file must exist.
python - <<'PY'
import json
import sys
from pathlib import Path

lib = json.loads(Path("data/lora_assets/instances.json").read_text())
loras = {r["output_name"] for r in lib["categories"] + lib["variants"] if "output_name" in r}
missing = [n for n in sorted(loras)
           if not Path("data/lora_assets/adapters", n, "pytorch_lora_weights.safetensors").is_file()]
meshes = [f"glb/{k}.glb" for k in json.loads(Path("data/meshes.json").read_text())]
missing += [m for m in meshes if not Path("data/lora_assets", m).is_file()]
if missing:
    sys.exit(f"{len(missing)} files missing, e.g. {missing[:5]}; re-run the download")
print(f"LoRA adapters ({len(loras)}) and meshes ({len(meshes)}) present")
PY

# 2. AmbientCG materials -> data/materials/ambientcg_full/ (about 0.6 GB at 2K JPG): the CLIP
#    top-5 floor and wall materials of each room type, from which a scene samples its own
python scripts/download/download_ambientcg.py --output data/materials/ambientcg_full -r 2K -f JPG \
    --ids data/materials/ambientcg_room_materials.json
# Check that every listed material has its colour map on disk.
python - <<'PY'
import json
import sys
from pathlib import Path

from scenebloom.render.materials import base_color_map

rooms = json.loads(Path("data/materials/ambientcg_room_materials.json").read_text())
ids = sorted({m for surfaces in rooms.values() for ms in surfaces.values() for m in ms})
missing = [m for m in ids if not base_color_map("data/materials/ambientcg_full", m).is_file()]
if missing:
    sys.exit(f"{len(missing)} of {len(ids)} materials are missing, "
             f"e.g. {missing[:5]}; re-run the download")
print(f"AmbientCG: all {len(ids)} listed materials present")
PY

# 3. M3DLayout -> data/m3dlayout/extracted/ (about 0.34 GB download)
mkdir -p "$M3D"
python - "$M3D" "$M3DLAYOUT_REVISION" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

root, revision = Path(sys.argv[1]), sys.argv[2]
# (file in the dataset repository, expected sha256, unpack into)
FILES = [
    ("layout_dataset/M3DLayout_json.zip",
     "10c0b11d43f980cd03f214cbd6bf377192257c7da58b56c94125d20375ef39e2", "layout"),
    ("rendering_dataset/3dfront/3dfront_postprocessed_5173.zip",
     "1d41914b14cad59f2360968d34ac01deb0555b0362b0659f40ad2f662d63a9c6", "3dfront"),
]
for name, sha, sub in FILES:
    path = Path(hf_hub_download("Metaverse-AI-Lab/M3DLayout", name, repo_type="dataset",
                                revision=revision, local_dir=root))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        sys.exit(f"{name}: sha256 {h.hexdigest()} differs from the expected {sha}")
    dest = root / "extracted" / sub
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)
    print(f"M3DLayout: {name} -> {dest}")
PY
echo "done. Next: bash scripts/build_index/build_retrieval_index.sh"
