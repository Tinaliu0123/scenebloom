# Contract-Guided 3D Compilation

`lifting/` lifts each proposal into 3D and extracts the relation contract, `fitting/` fits the room layout, `refinement/` refines it against the contract, and `completion/` adds wall objects and small objects.

## Models

| Name | Description |
|---|---|
| [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) with the [LIFT checkpoint](https://huggingface.co/uva-cv-lab/ovmono3d_lift) | 3D lifting of the proposals |
| [Depth Pro](https://github.com/apple/ml-depth-pro) | metric depth for lifting |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) Swin-B with its BERT text encoder | 2D detection for lifting |
| [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 | OVMono3D backbone |

Relation extraction and scene completion call [Qwen3-VL-8B-Instruct](https://openrouter.ai/qwen/qwen3-vl-8b-instruct) through OpenRouter.
