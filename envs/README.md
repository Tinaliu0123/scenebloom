# Environments

The code uses two conda environments.

- **`scenebloom`** (Python 3.10) runs the whole pipeline, from zoning to the Habitat export, and the LoRA training in `asset_generation/`. Build it with `bash envs/install_scenebloom.sh`. The script installs the CUDA compiler from conda, so the build needs neither a system CUDA nor a GPU.
- **`habitat`** (Python 3.9) is optional and only loads the exported scenes in habitat-sim 0.3.3, which is published for Python 3.9 only. Build it with `conda env create -f envs/habitat.yml`.

The install script compiles the CUDA extensions for compute capability 8.0, 8.6, 8.9 and 9.0. To run on another GPU, add its compute capability to `TORCH_CUDA_ARCH_LIST` at the top of the script before building.
