# Render

Builds the compiled scene in Genesis, renders it, and chooses floor and wall materials.

| Name | Description |
|---|---|
| [DFN5B-CLIP-ViT-H-14-378](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378) | ranks the materials for the room type |
| [AmbientCG](https://ambientcg.com) materials | floor and wall textures |

`download_assets.sh` downloads only the materials listed in `data/materials/ambientcg_room_materials.json`: the CLIP top-5 floor and wall materials of each room type. After changing `ROOM_QUERIES` in `materials.py`, run `python scripts/download/download_ambientcg.py` to download the full library (about 45 GB).
