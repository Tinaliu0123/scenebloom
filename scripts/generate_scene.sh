#!/usr/bin/env bash
# Generate one scene from a text prompt and export it for Habitat. Needs no scheduler.
#
# usage: scripts/generate_scene.sh "<prompt>" <out_dir> [--seed N] [--no-export] [cli options]
#   e.g. scripts/generate_scene.sh "a living room" outputs/living
# Without --seed every run draws a random seed and prints it. --no-export stops after
# generation; further options go to scenebloom.cli.
# Reads OPENROUTER_API_KEY and HF_TOKEN from the environment or from .env.
#
# Output: <out_dir>/scene_layout.json (layout), <out_dir>/renders/ (Genesis renders),
#         <out_dir>/habitat/scene.glb (Habitat scene).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# data/ paths in the code are relative to the repository root.
cd "$ROOT"

[ $# -ge 2 ] || { echo "usage: $0 \"<prompt>\" <out_dir> [--seed N] [--no-export] [cli options]" >&2; exit 2; }
PROMPT=$1
OUT=$2
shift 2
EXPORT=1
SEED=
CLI_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-export) EXPORT=0 ;;
        --seed) SEED=${2:?--seed needs a value}; shift ;;
        *) CLI_ARGS+=("$1") ;;
    esac
    shift
done
SEED=${SEED:-$(( RANDOM * 32768 + RANDOM ))}
case "$SEED" in ''|*[!0-9]*) echo "--seed takes a non-negative integer" >&2; exit 2 ;; esac

[ -f .env ] && { set -a; . ./.env; set +a; }
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (see .env.example)}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -e "$OUT" ]; then
    echo "$OUT already exists; choose a new output directory" >&2
    exit 2
fi

# Check every download and index before the first stage starts.
missing=()
for f in data/embeddings/embeddings.npy data/embeddings_mp3d/embeddings.npy \
         checkpoints/ovmono3d/ovmono3d_lift.pth checkpoints/ovmono3d/depth_pro.pt \
         third_party/ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth \
         checkpoints/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin \
         checkpoints/FLUX.1-dev/model_index.json \
         checkpoints/Qwen3-Embedding-0.6B/config.json; do
    [ -s "$f" ] || missing+=("$f")
done
for d in data/lora_assets/glb \
         data/materials/ambientcg_full data/lora_assets/adapters data/m3dlayout/extracted/3dfront \
         checkpoints/huggingface checkpoints/torch_hub; do
    [ -n "$(ls -A "$d" 2>/dev/null)" ] || missing+=("$d/")
done
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing input: %s\n' "${missing[@]}" >&2
    echo "run scripts/download/download_models.sh, scripts/download/download_assets.sh and" \
         "scripts/build_index/build_retrieval_index.sh (README, Download Models and Assets)" >&2
    exit 2
fi

echo "RUN prompt=\"$PROMPT\" seed=$SEED"
python -m scenebloom.cli "$PROMPT" --seed "$SEED" --output-dir "$OUT" "${CLI_ARGS[@]}"
echo "generated: $OUT"

if [ "$EXPORT" = 1 ]; then
    python -m scenebloom.export "$OUT" "$OUT/habitat"
fi
