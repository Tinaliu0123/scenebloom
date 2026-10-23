# Hierarchical Visual Sampling

`zoning/` divides the room into functional zones, guided by exemplar rooms retrieved from M3DLayout, and `proposals/` draws a visual layout proposal for each zone with FLUX and our LoRA adapters.

## Models and Data

| Name | Description |
|---|---|
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | text embedding for exemplar retrieval |
| [M3DLayout](https://huggingface.co/datasets/Metaverse-AI-Lab/M3DLayout) | exemplar rooms |
| [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev) | text-to-image model for the layout proposals |
| SceneBloom LoRA adapters | one FLUX LoRA adapter per library mesh |

Zoning calls [Qwen3-VL-8B-Instruct](https://openrouter.ai/qwen/qwen3-vl-8b-instruct) through OpenRouter.
