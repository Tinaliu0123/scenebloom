#!/usr/bin/env bash
# Build the two layout-retrieval indexes from the downloaded M3DLayout rooms:
#   data/embeddings       3D-FRONT rooms: 5173, one per scene id
#   data/embeddings_mp3d  Matterport3D rooms of the six types 3D-FRONT lacks: 627
#
# usage: bash scripts/build_index/build_retrieval_index.sh   (needs a GPU; encoding
# 5.8k short texts takes minutes on one)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
J=data/m3dlayout/extracted/layout/M3DLayout_json
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$J/3dfront_train.json" ] || { echo "missing $J; run scripts/download/download_assets.sh first" >&2; exit 1; }
[ -f checkpoints/Qwen3-Embedding-0.6B/config.json ] || {
    echo "missing checkpoints/Qwen3-Embedding-0.6B; run scripts/download/download_models.sh first" >&2; exit 1; }

python scripts/build_index/build_embeddings.py data/embeddings \
    "$J/3dfront_train.json" "$J/3dfront_val.json" "$J/3dfront_test.json"
python scripts/build_index/build_embeddings.py data/embeddings_mp3d \
    "$J/mp3d_train.json" "$J/mp3d_val.json" "$J/mp3d_test.json" \
    --room-types office bathroom kitchen lounge rec_game meeting
