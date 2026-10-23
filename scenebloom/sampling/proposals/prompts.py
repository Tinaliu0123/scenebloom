"""FreeFuse subjects, their LoRAs, and the trigger-injected FLUX prompts."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from scenebloom.utils.assets import CATEGORY_TO_GLB_ID, variant_keys
from scenebloom.utils.paths import DATA

_log = logging.getLogger(__name__)

_LORA_ASSETS = DATA / "lora_assets" / "instances.json"
LORA_ROOT = DATA / "lora_assets" / "adapters"


def weights_path(record: dict[str, Any]) -> Path:
    return LORA_ROOT / record["output_name"] / "pytorch_lora_weights.safetensors"


@lru_cache(maxsize=1)
def instance_records() -> dict[str, dict[str, Any]]:
    """``mesh_key -> LoRA record`` for every mesh whose LoRA weights exist on disk."""
    lib = json.loads(_LORA_ASSETS.read_text(encoding="utf-8"))
    records = {r["category"]: r for r in lib["categories"] if "output_name" in r}
    records.update((r["mesh"], r) for r in lib["variants"])
    return {k: r for k, r in records.items() if weights_path(r).is_file()}


def rendered_mesh(mesh_key: str) -> str:
    """The library mesh the renderer places for ``mesh_key`` (aliases resolved)."""
    glb_id = CATEGORY_TO_GLB_ID.get(mesh_key)
    return glb_id.partition(":")[2] if glb_id else mesh_key


def variant_pool(category: str) -> list[str]:
    """Sorted renderable meshes for ``category`` that have a trained LoRA."""
    recs = instance_records()
    pool = {rendered_mesh(k) for k in (category, *variant_keys(category))}
    return sorted(m for m in pool if m in recs)


def choose_mesh(category: str, zone_id: str, seed: int) -> str:
    """Deterministic mesh per (seed, zone, category); FLUX repeats one design per zone."""
    pool = variant_pool(category)
    if not pool:
        raise FileNotFoundError(f"no LoRA adapter for any {category} mesh under {LORA_ROOT}")
    h = hashlib.sha1(f"{seed}:{zone_id}:{category}".encode()).hexdigest()
    return pool[int(h, 16) % len(pool)]


def _find_free(text: str, phrase: str, taken: list) -> "re.Match | None":
    """First word-boundary, plural-tolerant match of ``phrase`` outside the ``taken`` spans."""
    for m in re.finditer(rf"\b{re.escape(phrase)}s?\b", text, flags=re.IGNORECASE):
        if all(m.end() <= a or m.start() >= b for a, b in taken):
            return m
    return None


def match_subjects(
    text: str,
    subjects: Sequence[str],
    nouns: Mapping[str, str],
    *,
    head_noun: bool = True,
) -> dict:
    """``{subject: match or None}``; full phrases claim text first ("coffee table" != side_table)."""
    out = {s: None for s in subjects}
    taken: list = []
    full = []
    for s in subjects:
        noun = nouns.get(s) or s.replace("_", " ")
        for cand in subject_candidates(s, noun, head_noun=False):
            full.append((-len(cand), s, cand))
    for _, s, cand in sorted(full):
        if out[s] is not None:
            continue
        m = _find_free(text, cand, taken)
        if m:
            out[s] = m; taken.append((m.start(), m.end()))
    if head_noun:
        for s in subjects:
            if out[s] is not None:
                continue
            noun = nouns.get(s) or s.replace("_", " ")
            head = noun.split()[-1]
            m = _find_free(text, head, taken)
            if m:
                out[s] = m; taken.append((m.start(), m.end()))
    return out


def inject_triggers_text(
    text: str,
    subjects: Sequence[str],
    nouns: Mapping[str, str],
    triggers: Mapping[str, str],
) -> tuple:
    """Insert triggers before named subjects; returns ``(text, concept_map, unnamed)``."""
    concept_map: dict = {}
    unnamed: list = []
    found = match_subjects(text, subjects, nouns)
    edits = []
    for s in subjects:
        m = found[s]
        if m is None:
            unnamed.append(s)
            continue
        phrase = f"{triggers[s]} {m.group(0)}"
        concept_map[s] = phrase
        edits.append((m.start(), m.end(), phrase))
    for start, end, phrase in sorted(edits, reverse=True):
        text = text[:start] + phrase + text[end:]
    return text, concept_map, unnamed


#: Asset-db category -> library subject; shared by the subject filter, trigger injector and zoner.
DEFAULT_CATEGORY_ALIASES: dict[str, str] = {
    "officechair": "chair",
    "lounge_chair": "armchair",
    "bedsidetable": "nightstand",
    "kids_bed": "bed",
    "couch": "sofa",
    "l_shaped_sofa": "sofa",
    "lamp": "floor_lamp",
    "dresser": "chest_of_drawers",
    "shelf": "bookshelf",
}


def normalise_category_names(
    text: str,
    subjects: Iterable[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Rewrite underscored category names and ``aliases`` to spaced, matchable text."""
    for s in subjects:
        spaced = s.replace("_", " ")
        if spaced != s:
            text = re.sub(
                rf"\b{re.escape(s)}(s?)\b", spaced + r"\1", text, flags=re.IGNORECASE
            )
    for alias, replacement in (aliases or {}).items():
        text = re.sub(
            rf"\b{re.escape(alias)}(s?)\b", replacement + r"\1", text,
            flags=re.IGNORECASE,
        )
    return text


def alias_replacements(
    subjects: Sequence[str],
    nouns: Mapping[str, str],
    category_map: Mapping[str, str],
) -> dict[str, str]:
    """``{alias: phrase}``: the spaced alias if it names its subject, else the subject noun."""
    out: dict[str, str] = {}
    for alias, subject in category_map.items():
        if subject not in subjects or alias == subject:
            continue
        noun = nouns.get(subject) or subject.replace("_", " ")
        spaced = alias.replace("_", " ")
        out[alias] = spaced if match_subject(spaced, subject, noun) else noun
    return out


def subject_candidates(
    subject: str, noun: str, *, head_noun: bool = True
) -> list[str]:
    """Phrases that name ``subject``, most specific first: noun, category name, head noun."""
    cat_name = subject.replace("_", " ")
    candidates = [noun]
    if cat_name not in candidates:
        candidates.append(cat_name)
    head = noun.split()[-1]
    if head_noun and head not in candidates:
        candidates.append(head)
    return candidates


def match_subject(
    text: str, subject: str, noun: str, *, head_noun: bool = True
) -> re.Match[str] | None:
    """First word-boundary, plural-tolerant match of any candidate phrase in ``text``, or None."""
    for cand in subject_candidates(subject, noun, head_noun=head_noun):
        match = re.search(rf"\b{re.escape(cand)}s?\b", text, flags=re.IGNORECASE)
        if match:
            return match
    return None


def split_subjects_by_prompt(
    prompt: str | None,
    subjects: Sequence[str],
    nouns: Mapping[str, str],
    aliases: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split ``subjects`` into ``(named, unnamed)`` by ``prompt``; an empty prompt names all."""
    text = (prompt or "").strip()
    if not text:
        return list(subjects), []
    text = normalise_category_names(text.rstrip("."), subjects, aliases)
    named: list[str] = []
    unnamed: list[str] = []
    found = match_subjects(text, subjects, nouns)
    for s in subjects:
        (named if found[s] is not None else unnamed).append(s)
    return named, unnamed


def prompt_named_subjects(
    prompt: str | None,
    vocabulary: Iterable[str],
    nouns: Mapping[str, str],
) -> list[str]:
    """Subjects in ``vocabulary`` that ``prompt`` names; used by the zoner's consistency check."""
    text = (prompt or "").strip()
    if not text:
        return []
    vocab = list(vocabulary)
    text = normalise_category_names(text.rstrip("."), vocab)
    # Strict (no head noun): "dining table" must not count as naming side_table.
    return [
        s
        for s in vocab
        if match_subject(text, s, nouns.get(s) or s.replace("_", " "), head_noun=False)
    ]


def _load_library_instances() -> Dict[str, Dict]:
    with open(_LORA_ASSETS, encoding="utf-8") as f:
        data = json.load(f)
    return {r["category"]: r for r in data["categories"]}


_LIBRARY_INSTANCES: Dict[str, Dict] = _load_library_instances()

LORA_DIR_BY_SUBJECT: Dict[str, str] = {
    cat: rec["output_name"] for cat, rec in _LIBRARY_INSTANCES.items() if "output_name" in rec
}

# A category without a trigger of its own is drawn only through its variant meshes.
SUBJECT_META: Dict[str, Dict] = {
    cat: {
        **({"trigger": rec["trigger"]} if "trigger" in rec else {}),
        "noun": rec.get("noun") or cat.replace("_", " "),
    }
    for cat, rec in _LIBRARY_INSTANCES.items()
}


def subject_meta(subject: str, meshes: Dict[str, str] | None = None) -> Dict:
    """Trigger and noun for ``subject``; ``meshes`` swaps in the zone mesh's trigger."""
    mesh = (meshes or {}).get(subject)
    if mesh and mesh != subject:
        # Keep the category noun: subject filtering and trigger placement match on it.
        return {"trigger": instance_records()[mesh]["trigger"], "noun": SUBJECT_META[subject]["noun"]}
    return dict(SUBJECT_META[subject])


# Wrapped around every zone prompt. Keep the prefix short: CLIP truncates at 77 tokens
# and a long prefix pushes the subject phrases out of its window.
FLOOR_PREFIX = (
    "A photo of a furnished room interior, "
    "every piece of furniture standing directly on the floor."
)
CONSTRAINT_SUFFIX = (
    "all of the furniture pieces present and fully visible together, "
    "each shown in full, complete and uncropped, "
    "the whole object inside the frame, nothing cut off at the edges, "
    # The LoRAs are trained on white-background renders; without this FLUX drifts to product shots.
    "the room's walls and floor visible as the background, "
    "wide-angle interior shot"
)


def _wrap_constraints(prompt: str) -> str:
    """Wrap a built prompt with the floor prefix and the coverage/no-crop suffix."""
    body = prompt.strip().lstrip(".").strip().rstrip(".")
    if not body:
        return f"{FLOOR_PREFIX} {CONSTRAINT_SUFFIX}."
    return f"{FLOOR_PREFIX} {body}, {CONSTRAINT_SUFFIX}."


def _subject_phrase(subject: str, meshes: Dict[str, str] | None = None) -> str:
    """'<trigger> <noun>', e.g. 'yte bed'."""
    meta = subject_meta(subject, meshes)
    return f"{meta['trigger']} {meta['noun']}"


def _inject_triggers(
    zone_prompt: str,
    subjects: List[str],
    aliases: Dict[str, str] | None = None,
    meshes: Dict[str, str] | None = None,
) -> tuple[str, Dict[str, str]]:
    """Return ``(t5_body, concept_map)``; each concept phrase is verbatim in the body."""
    text = normalise_category_names(
        zone_prompt.strip().rstrip("."), subjects, aliases
    )
    text, concept_map, unnamed = inject_triggers_text(
        text, subjects,
        {s: subject_meta(s, meshes)["noun"] for s in subjects},
        {s: subject_meta(s, meshes)["trigger"] for s in subjects},
    )
    fallback: List[str] = []
    for s in unnamed:
        phrase = _subject_phrase(s, meshes)
        fallback.append(phrase)
        concept_map[s] = phrase
    if fallback:
        _log.warning(
            "_inject_triggers: %d subject(s) not named in prompt, appended: %s",
            len(fallback), fallback,
        )
        text = text.rstrip(" .,") + ", with " + ", ".join(f"a {p}" for p in fallback)
    return text, concept_map


def build_concept_map(
    subjects: List[str],
    zone_prompt: str | None = None,
    aliases: Dict[str, str] | None = None,
    meshes: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Map each subject to its verbatim phrase inside the t5 prompt."""
    if zone_prompt and zone_prompt.strip():
        _, concept_map = _inject_triggers(
            zone_prompt, subjects, aliases=aliases, meshes=meshes
        )
        return concept_map
    return {s: f"a {_subject_phrase(s, meshes)}" for s in subjects}


def build_prompts(
    subjects: List[str],
    zone_prompt: str | None = None,
    aliases: Dict[str, str] | None = None,
    meshes: Dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(clip_prompt, t5_prompt)``; only the t5 prompt carries the LoRA triggers."""
    base = (zone_prompt or "").strip().rstrip(".")
    if base:
        clip_prompt = base
        t5_prompt, _ = _inject_triggers(
            base, subjects, aliases=aliases, meshes=meshes
        )
        t5_prompt = t5_prompt.rstrip(".") + "."
    else:
        items = ", ".join(
            f"a {_subject_phrase(s, meshes)}" for s in subjects
        )
        t5_prompt = f"A photo of a room containing {items}."
        clip_prompt = "A photo of a furnished indoor room"
    return _wrap_constraints(clip_prompt), _wrap_constraints(t5_prompt)


def prepare_prompt_embeds(pipe, prompt, prompt_2=None, negative_prompt=None,
                          num_images_per_prompt=1, max_sequence_length=512):
    import torch

    with torch.no_grad():
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        pooled_prompt_embeds = pipe._get_clip_prompt_embeds(
            prompt=prompt,
            device=pipe.text_encoder_2.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = pipe._get_t5_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=pipe.text_encoder_2.device,
        )

        pooled_negative_prompt_embeds = None
        negative_prompt_embeds = None
        if negative_prompt is not None:
            negative_prompt_2 = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            pooled_negative_prompt_embeds = pipe._get_clip_prompt_embeds(
                prompt=negative_prompt,
                device=pipe.text_encoder_2.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative_prompt_embeds = pipe._get_t5_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=pipe.text_encoder_2.device,
            )

        return pooled_prompt_embeds, prompt_embeds, pooled_negative_prompt_embeds, negative_prompt_embeds


def find_concept_positions(pipe, prompts, concepts, filter_meaningless=True, filter_single_char=True):
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'nor', 'so', 'yet',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'over', 'it', 'its', 'this', 'that', 'these',
        'those', 'their', 'his', 'her', 'my', 'your', 'our', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had', 'having',
        'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
    }
    PUNCTUATION = {',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']',
                   '{', '}', '-', '–', '—', '/', '\\', '...', '..'}
    MEANINGLESS_TOKENS = STOPWORDS | PUNCTUATION

    def is_meaningless_token(token_text, check_single_char=True):
        cleaned = token_text.replace('▁', '').replace('_', '').strip().lower()
        if not cleaned:
            return True
        if check_single_char and len(cleaned) == 1:
            return True
        return cleaned in MEANINGLESS_TOKENS

    if isinstance(prompts, str):
        prompts = [prompts]

    prompt_data_list = []
    for prompt in prompts:
        prompt_inputs = pipe.tokenizer_2(
            prompt,
            padding=False,
            truncation=False,
            return_length=False,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs.input_ids[0].tolist()
        offset_mapping = prompt_inputs.offset_mapping[0].tolist()
        prompt_data_list.append({'text': prompt, 'ids': prompt_ids, 'offset_mapping': offset_mapping})

    concept_pos_map = {}
    for concept_name, concept_text in concepts.items():
        concept_pos_map[concept_name] = []

        for prompt_data in prompt_data_list:
            positions = []
            positions_with_text = []
            prompt_text = prompt_data['text']
            prompt_ids = prompt_data['ids']
            offset_mapping = prompt_data['offset_mapping']

            search_start = 0
            while True:
                char_start = prompt_text.find(concept_text, search_start)
                if char_start == -1:
                    break
                char_end = char_start + len(concept_text)

                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_end > char_start and token_start < char_end:
                        if token_idx not in positions:
                            positions.append(token_idx)
                            token_text = pipe.tokenizer_2.decode(
                                [prompt_ids[token_idx]], skip_special_tokens=False
                            )
                            positions_with_text.append((token_idx, token_text))
                search_start = char_start + 1

            if filter_meaningless and positions_with_text:
                filtered_positions = [
                    pos for pos, text in positions_with_text
                    if not is_meaningless_token(text, check_single_char=filter_single_char)
                ]
                if not filtered_positions:
                    non_punct_positions = [
                        pos for pos, text in positions_with_text
                        if text.replace('▁', '').replace('_', '').strip() not in PUNCTUATION
                    ]
                    if non_punct_positions:
                        filtered_positions = [non_punct_positions[0]]
                    elif positions_with_text:
                        filtered_positions = [positions_with_text[0][0]]
                positions = filtered_positions

            positions.sort()
            concept_pos_map[concept_name].append(positions)

    return concept_pos_map


def get_prompt_token_texts(pipe, prompt, max_sequence_length=512):
    prompt_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_inputs.input_ids[0].tolist()
    return [pipe.tokenizer_2.decode([tid], skip_special_tokens=False) for tid in prompt_ids]
