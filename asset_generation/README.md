# Asset Generation

The asset library can be extended with new objects. Each object needs a mesh, which we generate with TRELLIS, and a LoRA adapter, which teaches FLUX.1 [dev] to draw that mesh in the layout proposals. A new object is added as a variant of an existing category, named `<category>_<NN>`, in three steps.

## 1. Generate the Mesh

Install [TRELLIS](https://github.com/microsoft/TRELLIS) and its environment in `third_party/TRELLIS` following its README, then generate the mesh from a text prompt:

```bash
python asset_generation/mesh_generation/trellis_text_to_glb.py armchair_03 "a green velvet armchair with short wooden legs"
```

The mesh is written to `outputs/meshes/armchair_03.glb` as a textured GLB, y-up and inside a unit cube. The renderer scales it to the box of the object it stands for.

## 2. Register the Mesh

1. Copy the GLB to `data/lora_assets/glb/`.
2. Add an entry for it to `data/meshes.json`: the `bounding_box_extents` and `center` of the GLB's bounds (`trimesh.load(path).bounds`), and the `footprint`, its width and depth in metres.
3. Add a record to the `variants` list of `data/lora_assets/instances.json` with the mesh name, its category, a trigger word and the adapter name `<mesh>__<trigger>`.

## 3. Train the LoRA Adapter

[`train_lora.sh`](lora_training/train_lora.sh) renders eight views of the mesh on a white background ([`render_views.py`](lora_training/render_views.py)) and fine-tunes FLUX.1 [dev] on them with [`train_dreambooth_lora_flux.py`](lora_training/train_dreambooth_lora_flux.py), the diffusers DreamBooth LoRA script with orthogonal adaptation added, so that the adapters of one zone can be loaded together. It runs in the `scenebloom` environment:

```bash
bash asset_generation/lora_training/train_lora.sh armchair_03
```

The adapter is written to `outputs/lora/armchair_03__<trigger>/`. Copy this folder to `data/lora_assets/adapters/`, and the pipeline can then draw and place the new mesh. The training settings are at the top of `train_lora.sh`; options after `--` go to the training script.
