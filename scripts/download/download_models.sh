#!/usr/bin/env bash
# Download the models the pipeline loads, at pinned revisions, into
# checkpoints/, and the OVMono3D source into third_party/ovmono3d. Run once on a machine
# with network access; generation then needs network only for the OpenRouter API.
#
# usage: bash scripts/download/download_models.sh
# HF_TOKEN (environment or .env) is required for black-forest-labs/FLUX.1-dev, a gated
# model: accept its license on huggingface.co with the same account first.
#
# checkpoints/
#   FLUX.1-dev/  Qwen3-Embedding-0.6B/  DFN5B-CLIP-ViT-H-14-378/   one folder per model
#   ovmono3d/     ovmono3d_lift.pth, depth_pro.pt, groundingdino_swinb_cogcoor.pth
#   huggingface/  bert-base-uncased, which GroundingDINO loads by its hub id
#   torch_hub/    DINOv2, which OVMono3D loads through torch.hub
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }

CKPT=checkpoints
OVDIR=third_party/ovmono3d
OVMONO3D_REPO=https://github.com/UVA-Computer-Vision-Lab/ovmono3d.git
OVMONO3D_COMMIT=25b65771797902d8d64034600078a7b56b8aed89
GDINO_URL=https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
DEPTHPRO_URL=https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

# 1. OVMono3D source at a pinned commit. The pipeline runs it unmodified.
if [ ! -d "$OVDIR/.git" ]; then
    git clone "$OVMONO3D_REPO" "$OVDIR"
fi
git -C "$OVDIR" fetch --quiet origin
git -C "$OVDIR" checkout --quiet "$OVMONO3D_COMMIT"
echo "ovmono3d: $(git -C "$OVDIR" rev-parse HEAD)"

# 2. OVMono3D checkpoints -> checkpoints/ovmono3d. The upstream code opens
# ./checkpoints/groundingdino_swinb_cogcoor.pth inside the checkout, so that folder
# is a link to checkpoints/ovmono3d.
mkdir -p "$CKPT/ovmono3d"
if [ ! -L "$OVDIR/checkpoints" ]; then
    [ -e "$OVDIR/checkpoints" ] && { echo "$OVDIR/checkpoints exists and is not a link to $CKPT/ovmono3d" >&2; exit 1; }
    ln -s ../../checkpoints/ovmono3d "$OVDIR/checkpoints"
fi
fetch() {  # fetch URL DEST
    if [ -s "$2" ]; then echo "present: $2"; return; fi
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
}
fetch "$GDINO_URL" "$CKPT/ovmono3d/groundingdino_swinb_cogcoor.pth"
fetch "$DEPTHPRO_URL" "$CKPT/ovmono3d/depth_pro.pt"

# 3. Hugging Face models at pinned revisions.
python - "$CKPT" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ckpt = Path(sys.argv[1])
# (repo id, pinned revision, files to fetch; None = whole repository)
MODELS = [
    ("black-forest-labs/FLUX.1-dev", "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
     ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
      "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*"]),
    ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", None),
    ("apple/DFN5B-CLIP-ViT-H-14-378", "01b771ed0d1395ca5ffdd279897d665ebe00dfd2",
     ["open_clip_config.json", "open_clip_pytorch_model.bin"]),
]
for repo, revision, patterns in MODELS:
    dest = ckpt / repo.split("/")[1]
    snapshot_download(repo, revision=revision, allow_patterns=patterns, local_dir=dest)
    print(f"{repo}@{revision[:7]} -> {dest}")

lift = hf_hub_download("uva-cv-lab/ovmono3d_lift", "ovmono3d_lift.pth",
                       revision="e9933f7726e532eb65d19989663839c39ba20b3b",
                       local_dir=ckpt / "ovmono3d")
print(f"uva-cv-lab/ovmono3d_lift@e9933f7 -> {lift}")

# GroundingDINO loads BERT by the hub id "bert-base-uncased". OVMono3D reads it offline
# from this cache (compilation/lifting/ovmono3d_infer.py), where refs/main names the
# pinned revision.
bert_revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
cache = ckpt / "huggingface"
snapshot_download("bert-base-uncased", revision=bert_revision, cache_dir=cache,
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "vocab.txt"])
ref = cache / "models--bert-base-uncased" / "refs" / "main"
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(bert_revision)
print(f"bert-base-uncased@{bert_revision[:7]} -> {cache}")
PY

# 4. DINOv2 backbone, loaded by OVMono3D through torch.hub.
python -c "import sys, torch; torch.hub.set_dir(sys.argv[1]); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')" \
    "$CKPT/torch_hub" > /dev/null
echo "dinov2_vitb14 -> $CKPT/torch_hub"
echo "done. Next: bash scripts/download/download_assets.sh"
