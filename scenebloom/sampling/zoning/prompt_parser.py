"""PromptParser: rule-based text prompt -> SceneSpec."""

from __future__ import annotations

from typing import NamedTuple

from scenebloom.utils.schemas import RoomConstraint, SceneSpec

_ROOM_TYPE_KEYWORDS: dict[str, str] = {
    "bedroom":          "bedroom",
    "bed room":         "bedroom",
    "master bedroom":   "bedroom",
    "guest bedroom":    "bedroom",
    "living room":      "living_room",
    "living":           "living_room",
    "lounge":           "lounge",
    "sitting room":     "lounge",
    "dining room":      "dining_room",
    "dining":           "dining_room",
    "kitchen":          "kitchen",
    "bathroom":         "bathroom",
    "bath room":        "bathroom",
    "office":           "office",
    "study":            "study",
    "library":          "library",
    "game room":        "rec_game",
    "games room":       "rec_game",
    "rec room":         "rec_game",
    "recreation room":  "rec_game",
    "meeting room":     "meeting",
    "conference room":  "meeting",
    "boardroom":        "meeting",
    "hallway":          "hallway",
    "corridor":         "hallway",
    "nursery":          "bedroom",
}

_OBJECT_KEYWORDS: dict[str, str] = {
    "sofa":           "sofa",
    "couch":          "sofa",
    "sectional":      "l_shaped_sofa",
    "bed":            "bed",
    "desk":           "desk",
    "table":          "dining_table",
    "dining table":   "dining_table",
    "coffee table":   "coffee_table",
    "bookshelf":      "bookshelf",
    "bookcase":       "bookshelf",
    "wardrobe":       "wardrobe",
    "closet":         "wardrobe",
    "chair":          "chair",
    "armchair":       "armchair",
    "nightstand":     "nightstand",
    "tv stand":       "tv_stand",
    "tv cabinet":     "tv_stand",
    "television":     "tv_stand",
    "shelf":          "shelf",
    "shelves":        "shelf",
    "cabinet":        "cabinet",
    "plant":          "plant",
}

_STYLE_KEYWORDS: dict[str, str] = {
    "modern":        "modern",
    "contemporary":  "contemporary",
    "cozy":          "cozy",
    "warm":          "cozy",
    "minimalist":    "minimalist",
    "minimal":       "minimalist",
    "clean":         "minimalist",
    "simple":        "minimalist",
    "rustic":        "rustic",
    "industrial":    "industrial",
    "scandinavian":  "scandinavian",
    "nordic":        "scandinavian",
    "luxurious":     "luxurious",
    "luxury":        "luxurious",
    "traditional":   "traditional",
    "vintage":       "vintage",
    "retro":         "retro",
    "bohemian":      "bohemian",
    "boho":          "bohemian",
}


class _ParseResult(NamedTuple):
    scene_type: str
    room_constraints: list[RoomConstraint]
    style_tags: list[str]


class PromptParser:
    """Convert a free-text prompt into a SceneSpec, with defaults for what it does not say."""

    def parse(self, prompt: str) -> SceneSpec:
        """Parse a prompt such as "a modern bedroom with a desk" into a SceneSpec."""
        result = self._parse_internal(prompt.strip())
        return SceneSpec(
            prompt_raw=prompt,
            scene_type=result.scene_type,
            room_constraints=result.room_constraints,
            style_tags=result.style_tags,
        )

    def _parse_internal(self, prompt: str) -> _ParseResult:
        lower = prompt.lower()
        # Keywords are tried longest first, and a matched span is not matched again.
        consumed_spans: list[tuple[int, int]] = []

        scene_type = "indoor_office" if "office" in lower else "indoor_residential"

        room_constraints: list[RoomConstraint] = []
        for kw, room_type in sorted(_ROOM_TYPE_KEYWORDS.items(), key=lambda x: -len(x[0])):
            pos = lower.find(kw)
            if pos >= 0 and not _span_consumed(pos, pos + len(kw), consumed_spans):
                room_constraints.append(RoomConstraint(room_type=room_type))
                consumed_spans.append((pos, pos + len(kw)))

        if not room_constraints:
            room_constraints = [RoomConstraint(room_type="living_room")]

        object_categories: list[str] = []
        for kw, cat in sorted(_OBJECT_KEYWORDS.items(), key=lambda x: -len(x[0])):
            pos = lower.find(kw)
            if pos >= 0 and not _span_consumed(pos, pos + len(kw), consumed_spans):
                object_categories.append(cat)
                consumed_spans.append((pos, pos + len(kw)))

        if object_categories and room_constraints:
            first = room_constraints[0]
            room_constraints[0] = RoomConstraint(
                room_type=first.room_type,
                required_categories=list(dict.fromkeys(
                    first.required_categories + object_categories
                )),
            )

        style_tags: list[str] = []
        seen_styles: set[str] = set()
        for kw, tag in sorted(_STYLE_KEYWORDS.items(), key=lambda x: -len(x[0])):
            pos = lower.find(kw)
            if pos >= 0 and tag not in seen_styles and not _span_consumed(
                pos, pos + len(kw), consumed_spans
            ):
                style_tags.append(tag)
                seen_styles.add(tag)
                consumed_spans.append((pos, pos + len(kw)))

        return _ParseResult(
            scene_type=scene_type,
            room_constraints=room_constraints,
            style_tags=style_tags,
        )


def _span_consumed(start: int, end: int, consumed: list[tuple[int, int]]) -> bool:
    """Return True if the span [start, end) overlaps any consumed span."""
    for cs, ce in consumed:
        if start < ce and end > cs:
            return True
    return False
