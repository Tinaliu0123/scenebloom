"""Subprocess entry point: generate FreeFuse zone images from a JSON spec file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("cli_generate")


def _read_spec(spec_path: Path) -> dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"spec file not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if "zones" not in spec or not isinstance(spec["zones"], list):
        raise ValueError("spec must contain a 'zones' list")
    if "lora_root" not in spec:
        raise ValueError("spec must contain 'lora_root'")
    if "seeds" not in spec:
        raise ValueError("spec must contain 'seeds'")
    return spec


def _generate(spec: dict[str, Any]) -> dict[str, list[str]]:
    """Run one spec and return ``{zone_id: [png path]}`` for the zones with an image."""
    from scenebloom.sampling.proposals.generator import generate_freefuse_batch

    zone_specs = spec["zones"]
    results = generate_freefuse_batch(
        zone_specs=zone_specs,
        lora_root=Path(spec["lora_root"]),
        seeds=[int(s) for s in spec["seeds"]],
    )
    logger.info("cli_generate: generated %d/%d zones", len(results), len(zone_specs))
    return {zid: [str(p) for p in paths] for zid, paths in results.items()}


def main(argv: list[str] | None = None) -> int:
    from scenebloom.sampling.proposals.generator import apply_sdpa_backend_policy
    apply_sdpa_backend_policy()
    parser = argparse.ArgumentParser(
        description="Run FreeFuse multi-LoRA FLUX generation for one or more zones.",
    )
    parser.add_argument("--spec", required=True,
                        help="Path to JSON spec file with lora_root/seeds/zones.")
    parser.add_argument("--output", required=True, help="Path to write the JSON result map.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    spec = _read_spec(Path(args.spec))
    logger.info(
        "cli_generate: %d zones, lora_root=%s, seeds=%s",
        len(spec["zones"]), spec["lora_root"], spec["seeds"],
    )
    results = _generate(spec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
