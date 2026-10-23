#!/usr/bin/env bash
# The CUDA compiler and gcc come from conda; no GPU is needed to build.
# Each CUDA extension is compiled against the torch installed in step 2, so the
# order matters. Usage (from the repository root):
#   bash envs/install_scenebloom.sh
set -euo pipefail

ENV_NAME=scenebloom
export PYTHONNOUSERSITE=1
# Compile for A100 (8.0), A10/RTX30 (8.6), L40S/RTX40 (8.9) and H100/H200 (9.0).
# An extension built for the wrong architecture can return empty detections
# without raising an error; add any other GPU you plan to run on here.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export FORCE_CUDA=1

# 1. base environment, with the CUDA 12.1 compiler and gcc 11 that steps 5-8 use.
# The GL and EGL libraries let Genesis render on nodes without a system libGL.
conda create -y -n "$ENV_NAME" -c nvidia/label/cuda-12.1.1 -c conda-forge \
    python=3.10 pip libgl libgl-devel libglu libegl libegl-devel mesalib \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev gxx_linux-64=11
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
PREFIX="$(dirname "$(dirname "$PY")")"
PIP=("$PY" -m pip)
export CUDA_HOME="$PREFIX"
export CC="$PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

# 2. torch first, from the CUDA 12.1 index
"${PIP[@]}" install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. pin numpy 1.x and torch for every later step
CONSTRAINTS="$(mktemp)"
printf 'numpy==1.26.4\ntorch==2.5.1+cu121\n' > "$CONSTRAINTS"

# 4. pure-wheel stack (same list as envs/scenebloom.yml)
"${PIP[@]}" install -c "$CONSTRAINTS" \
    numpy==1.26.4 diffusers==0.37.0 transformers==4.57.6 "huggingface-hub>=0.34,<1.0" \
    accelerate==1.13.0 peft==0.18.1 safetensors==0.7.0 sentencepiece protobuf \
    prodigyopt==1.1.2 tensorboard==2.21.0 \
    sentence-transformers==5.1.1 open_clip_torch==3.3.0 ftfy "timm>=1.0,<1.1" \
    taichi==1.7.3 tetgen==0.6.4 libigl==2.6.1 mujoco==3.2.5 \
    scipy==1.13.1 opencv-python-headless==4.11.0.86 open3d==0.18.0 trimesh==4.8.3 \
    shapely==2.0.7 python-fcl imageio matplotlib pillow tqdm \
    "pydantic>=2.12,<2.13" openai==1.109.1 requests \
    fvcore==0.1.5.post20221221 iopath==0.1.9 einops==0.8.1
# genesis-world pins pydantic, moviepy and opencv-python versions that conflict
# with the stack above, so it is installed without its dependencies.
"${PIP[@]}" install --no-deps genesis-world==0.2.1
"${PIP[@]}" install -c "$CONSTRAINTS" \
    coacd==1.0.7 freetype-py==2.5.1 lxml==6.0.2 moviepy==1.0.3 numba==0.60.0 OpenEXR==3.4.0 \
    pycollada==0.9.2 PyGEL3D==0.6.1 pyglet==2.1.9 pygltflib==1.16.0 pymeshlab==2025.7 \
    scikit-image==0.24.0 screeninfo==0.8.1

# 5-8. CUDA extensions, built against the torch from step 2
"${PIP[@]}" install -c "$CONSTRAINTS" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@055ab3a2e3e611dff66fa82f632e62a315f3b5e7"
"${PIP[@]}" install -c "$CONSTRAINTS" --no-build-isolation \
    "git+https://github.com/yaojin17/detectron2.git@8ece38992c987b0877a45deeb662ec1d1dd069a4"
"${PIP[@]}" install -c "$CONSTRAINTS" --no-build-isolation \
    "git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44"

# 9. Depth Pro and segment-anything (pure Python)
"${PIP[@]}" install -c "$CONSTRAINTS" \
    "git+https://github.com/apple/ml-depth-pro.git@b2cd0d51daa95e49277a9f642f7fd736b7f9e91d" \
    "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"

rm -f "$CONSTRAINTS"
echo "done: $PY"
