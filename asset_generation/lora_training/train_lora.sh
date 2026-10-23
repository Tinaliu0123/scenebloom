#!/usr/bin/env bash
# Train the LoRA adapters of library meshes with the settings of the released adapters:
# render each mesh from eight views, then fine-tune FLUX.1 [dev] on them with DreamBooth,
# prior preservation and orthogonal adaptation.
#
# usage: asset_generation/lora_training/train_lora.sh <mesh>... [-- trainer options]
#        asset_generation/lora_training/train_lora.sh all [-- trainer options]
#   e.g. asset_generation/lora_training/train_lora.sh bed sofa
# A mesh is a mesh of data/lora_assets/glb whose adapter is listed in
# data/lora_assets/instances.json; "all" trains every listed adapter, one after another.
# Each adapter is written to $OUT/<adapter>/pytorch_lora_weights.safetensors and skipped
# when it exists; copy its folder to data/lora_assets/adapters/ to use it in the pipeline.
# Options after "--" go to train_dreambooth_lora_flux.py and override the settings below.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Settings of the released adapters
OUT=outputs/lora
RESOLUTION=512
BATCH_SIZE=1
GRAD_ACCUM=2
LEARNING_RATE=1.0  # Prodigy with a constant schedule
STEPS=1000
RANK=4
CLASS_IMAGES=100
PRIOR_LOSS_WEIGHT=0.3

HERE=asset_generation/lora_training
MESHES=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do MESHES+=("$1"); shift; done
[ $# -gt 0 ] && shift
[ ${#MESHES[@]} -gt 0 ] || { echo "usage: $0 <mesh>...|all [-- trainer options]" >&2; exit 2; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

[ -s checkpoints/FLUX.1-dev/model_index.json ] || {
    echo "checkpoints/FLUX.1-dev is missing; run scripts/download/download_models.sh" >&2; exit 2; }

# One line per adapter: mesh, adapter, instance prompt, class prompt, class folder and an
# orthogonal-adaptation seed of its own.
PLAN="$(python - "${MESHES[@]}" <<'PY'
import hashlib
import json
import sys

lib = json.load(open("data/lora_assets/instances.json"))
nouns = {r["category"]: r["noun"] for r in lib["categories"]}
adapters = {}
for r in lib["categories"] + lib["variants"]:
    mesh = r.get("mesh", r["category"])
    if r.get("output_name", "").split("__")[0] == mesh:
        adapters.setdefault(mesh, r)
wanted = sorted(adapters) if sys.argv[1:] == ["all"] else sys.argv[1:]
unknown = [m for m in wanted if m not in adapters]
if unknown:
    sys.exit(f"no adapter in data/lora_assets/instances.json for: {' '.join(unknown)}")
for mesh in wanted:
    rec = adapters[mesh]
    noun = nouns[rec["category"]]
    article = "an" if noun[0] in "aeiou" else "a"
    seed = int(hashlib.sha256(rec["output_name"].encode()).hexdigest()[:8], 16)
    print("\t".join([mesh, rec["output_name"], f"a photo of {rec['trigger']} {noun}",
                     f"a photo of {article} {noun}", noun.replace(" ", "_"), str(seed)]))
PY
)"

mapfile -t ROWS <<< "$PLAN"
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r MESH ADAPTER INSTANCE_PROMPT CLASS_PROMPT CLASS_NAME ORTHOGONAL_SEED <<< "$row"
    if [ -s "$OUT/$ADAPTER/pytorch_lora_weights.safetensors" ]; then
        echo "SKIP $ADAPTER (exists)"
        continue
    fi
    GLB=data/lora_assets/glb/$MESH.glb
    [ -s "$GLB" ] || { echo "$GLB is missing; run scripts/download/download_assets.sh" >&2; exit 2; }
    VIEWS=$OUT/views/$MESH
    if [ -z "$(ls -A "$VIEWS" 2>/dev/null)" ]; then
        python "$HERE/render_views.py" "$GLB" "$VIEWS"
    fi
    # The trainer opens every entry of the instance folder as an image.
    if find "$VIEWS" -mindepth 1 ! -name '*.png' | grep -q .; then
        echo "$VIEWS holds entries other than the rendered PNGs" >&2; exit 1
    fi

    echo "TRAIN $ADAPTER: \"$INSTANCE_PROMPT\" (prior: \"$CLASS_PROMPT\")"
    accelerate launch --num_processes=1 --num_machines=1 --dynamo_backend=no \
        --mixed_precision=bf16 "$HERE/train_dreambooth_lora_flux.py" \
        --pretrained_model_name_or_path=checkpoints/FLUX.1-dev \
        --instance_data_dir="$VIEWS" \
        --instance_prompt="$INSTANCE_PROMPT" \
        --with_prior_preservation \
        --class_data_dir="$OUT/class/$CLASS_NAME" \
        --class_prompt="$CLASS_PROMPT" \
        --num_class_images="$CLASS_IMAGES" \
        --prior_loss_weight="$PRIOR_LOSS_WEIGHT" \
        --use_orthogonal_adaptation \
        --orthogonal_seed="$ORTHOGONAL_SEED" \
        --rank="$RANK" \
        --lora_alpha="$RANK" \
        --resolution="$RESOLUTION" \
        --train_batch_size="$BATCH_SIZE" \
        --gradient_accumulation_steps="$GRAD_ACCUM" \
        --guidance_scale=1 \
        --optimizer=prodigy \
        --learning_rate="$LEARNING_RATE" \
        --lr_scheduler=constant \
        --lr_warmup_steps=0 \
        --max_train_steps="$STEPS" \
        --mixed_precision=bf16 \
        --output_dir="$OUT/$ADAPTER" \
        "$@"
    echo "adapter: $OUT/$ADAPTER/pytorch_lora_weights.safetensors"
done
