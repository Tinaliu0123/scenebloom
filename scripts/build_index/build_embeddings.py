"""Embed M3DLayout rooms with Qwen3 for exemplar retrieval (embeddings.npy, metadata.json).

usage: python scripts/build_index/build_embeddings.py <out_dir> <M3DLayout json>...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scenebloom.sampling.zoning.retrieval import (  # noqa: E402
    Qwen3Encoder,
    load_scene_records,
    to_minimal_string,
    to_rich_json,
)

MODEL = "checkpoints/Qwen3-Embedding-0.6B"
BATCH_SIZE = 128

# Matterport3D room type from its sentence; the first matching rule wins, specific ones first.
SENTENCE_RULES = [
    ("entryway",        [r"entryway", r"foyer", r"lobby"]),
    ("laundry_mudroom", [r"laundryroom", r"mudroom", r"laundry"]),
    ("gym",             [r"workout", r"\bgym\b", r"exercise"]),
    ("meeting",         [r"meeting room", r"conference", r"boardroom"]),
    ("rec_game",        [r"home theater", r"game room", r"rec room", r"recreation",
                         r"media room", r"\btheater\b", r"playroom", r"billiard", r"arcade"]),
    ("spa_sauna",       [r"spa-sauna", r"sauna", r"\bspa\b"]),
    ("porch_deck",      [r"porch", r"terrace", r"patio", r"\bdeck\b", r"veranda"]),
    ("utility",         [r"utility", r"toolroom", r"tool room", r"boiler",
                         r"furnace", r"mechanical", r"\bpantry\b"]),
    ("classroom",       [r"classroom"]),
    ("library",         [r"library"]),
    ("bar",             [r"\bbar\b", r"wet bar", r"\bpub\b"]),
    ("garage",          [r"garage"]),
    ("balcony",         [r"balcony"]),
    ("closet",          [r"closet", r"walk-in", r"wardrobe room", r"dressing room"]),
    ("bathroom",        [r"bathroom", r"powder room", r"restroom", r"washroom"]),
    ("toilet",          [r"\btoilet\b", r"water closet", r"\bwc\b"]),
    ("bedroom",         [r"bedroom", r"\bnursery\b"]),
    ("kitchen",         [r"kitchen", r"kitchenette"]),
    ("dining",          [r"dining"]),
    ("living",          [r"living"]),
    ("family",          [r"family room", r"\bfamily\b", r"\bden\b"]),
    ("lounge",          [r"lounge", r"sitting room"]),
    ("office",          [r"office", r"\bstudy\b", r"workspace", r"home office"]),
    ("hallway",         [r"hallway", r"corridor", r"passage"]),
    ("stairs",          [r"stair"]),
    ("church_hall",     [r"church", r"cathedral", r"ecclesiastical", r"chapel", r"\bhall\b"]),
    ("wine_cellar",     [r"wine cellar", r"\bcellar\b"]),
    ("outdoor",         [r"backyard", r"courtyard", r"\bgarden\b", r"\byard\b", r"driveway"]),
]
# Without a room word, the first type whose signature objects are present.
OBJECT_RULES = [
    ("bathroom", lambda s: s & {"toilet", "bathtub", "shower", "standing_sink"}),
    ("kitchen", lambda s: s & {"oven", "microwave", "kitchen_cabinet", "kitchen_island",
                               "kitchen_space", "dishwasher", "stove", "beverage_fridge"}
                          or {"counter", "sink"} <= s),
    ("bedroom", lambda s: s & {"bed", "kids_bed"}),
    ("dining", lambda s: "dining_table" in s and s & {"dining_chair", "chair"}),
    ("living", lambda s: s & {"sofa", "l_shaped_sofa", "multi_seat_sofa", "loveseat_sofa",
                              "lazy_sofa"}),
    ("office", lambda s: "desk" in s and not s & {"bed", "kids_bed"}),
]


def sentence_room_type(scene: dict) -> str | None:
    """Room type of a Matterport3D room from its sentence, else from its objects."""
    sentence = scene.get("description", {}).get("global_description", {}).get("room_type", [])
    text = (" ".join(sentence) if isinstance(sentence, list) else sentence or "").lower()
    for room_type, patterns in SENTENCE_RULES:
        if any(re.search(p, text) for p in patterns):
            return room_type
    cats = {o.get("category") for o in scene.get("objects", [])}
    return next((room_type for room_type, rule in OBJECT_RULES if rule(cats)), None)


def load_rooms(paths: list[Path], room_types: set[str] | None) -> list:
    """One record per scene_id; with room_types, only non-empty rooms of those types."""
    rooms, seen = [], set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        labels = ({} if raw["source"] == "3DFront" else
                  {s["scene_id"]: sentence_room_type(s) for s in raw["scenes"]})
        for rec in load_scene_records(path):
            if rec.scene_id in seen:
                continue
            if labels:
                rec = rec.model_copy(update={"room_type": labels[rec.scene_id]})
            if room_types is not None and (rec.room_type not in room_types or not rec.objects):
                continue
            seen.add(rec.scene_id)
            rooms.append(rec)
    return rooms


def room_metadata(rooms: list) -> list[dict]:
    return [{"scene_id": rec.scene_id, "room_type": rec.room_type,
             "minimal_str": to_minimal_string(rec), "rich_json": to_rich_json(rec)}
            for rec in rooms]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("scene_db", nargs="+", type=Path, help="M3DLayout JSON files")
    p.add_argument("--room-types", nargs="+", help="keep only these room types")
    args = p.parse_args()

    rooms = load_rooms(args.scene_db, set(args.room_types) if args.room_types else None)
    if not rooms:
        sys.exit("no rooms to index")
    metadata = room_metadata(rooms)
    embeddings = Qwen3Encoder(MODEL).encode([m["minimal_str"] for m in metadata],
                                            batch_size=BATCH_SIZE, show_progress=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "embeddings.npy", embeddings)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(metadata)} rooms -> {args.out_dir}")


if __name__ == "__main__":
    main()
