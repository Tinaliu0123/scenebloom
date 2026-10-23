"""Command line of the pipeline: ``python -m scenebloom.cli "a living room"``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scenebloom.cli",
        description="Text prompt → simulation-ready 3-D scene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "prompt",
        help="Natural-language description of a bedroom, living room, dining room, office or bathroom",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override default outputs/ directory",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed (overrides config.random_seed)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    overrides: dict = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["random_seed"] = args.seed

    from scenebloom.utils.config import PipelineConfig
    from scenebloom.pipeline import PipelineError, PipelineOrchestrator

    config = PipelineConfig(**overrides)
    orchestrator = PipelineOrchestrator(config=config)

    try:
        layout = orchestrator.run(args.prompt)
    except PipelineError as exc:
        print(f"ERROR [{exc.phase}]: {exc}", file=sys.stderr)
        if args.verbose and exc.cause is not None:
            import traceback
            traceback.print_exception(type(exc.cause), exc.cause, exc.cause.__traceback__,
                                      file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1

    print(f"{len(layout.placed_objects)} objects placed: {config.output_dir / 'scene_layout.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
