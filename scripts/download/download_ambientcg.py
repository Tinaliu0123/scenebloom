#!/usr/bin/env python3
# Adapted from scripts/download_ambientcg.py of SceneSmith
# (https://github.com/nepfaff/scenesmith), under the MIT License:
#
# Copyright (c) 2026 Nicholas Pfaff
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Download AmbientCG PBR material packs (adapted from SceneSmith); extracted packs are skipped."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://ambientcg.com/api/v2/full_json"


def list_downloads(resolution: str, fmt: str) -> list[tuple[str, str, str]]:
    session, offset, found = requests.Session(), 0, []
    target = f"{resolution}-{fmt}"
    while True:
        response = session.get(
            API,
            params={"type": "Material", "include": "downloadData",
                    "limit": 250, "offset": offset},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        assets = data.get("foundAssets", [])
        for asset in assets:
            cats = (asset.get("downloadFolders", {}).get("default", {})
                    .get("downloadFiletypeCategories", {}))
            zip_cat = cats.get("zip", {}) if isinstance(cats, dict) else {}
            for download in zip_cat.get("downloads", []):
                if download.get("attribute") == target:
                    found.append((asset["assetId"], download["downloadLink"],
                                  download.get("fileName", asset["assetId"] + ".zip")))
                    break
        offset += len(assets)
        if not assets or offset >= int(data.get("numberOfResults", offset)):
            return found
        time.sleep(0.1)


def _stream_to_file(session: requests.Session, url: str, dest: Path, timeout: int) -> None:
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)


def download_one(item: tuple[str, str, str], output: Path) -> str:
    asset_id, url, filename = item
    target = output / asset_id
    if target.exists() and any(target.iterdir()):
        return "skip"
    archive = output / filename
    _stream_to_file(requests.Session(), url, archive, timeout=300)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink()
    return "ok"


def listed_ids(path: Path) -> set[str]:
    """Material ids of a ``{room: {surface: [material_id, ...]}}`` JSON file."""
    rooms = json.loads(Path(path).read_text(encoding="utf-8"))
    return {material_id for surfaces in rooms.values()
            for material_ids in surfaces.values() for material_id in material_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data/materials/ambientcg_full"))
    ap.add_argument("-r", "--resolution", default="2K")
    ap.add_argument("-f", "--format", default="JPG", choices=["PNG", "JPG"])
    ap.add_argument("--ids", type=Path, default=None, metavar="JSON",
                    help="download only the materials listed in this file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    items = list_downloads(args.resolution, args.format)
    if args.ids is not None:
        wanted = listed_ids(args.ids)
        items = [item for item in items if item[0] in wanted]
        unavailable = sorted(wanted - {item[0] for item in items})
        if unavailable:
            sys.exit(f"{len(unavailable)} listed materials have no "
                     f"{args.resolution}-{args.format} download: {unavailable}")
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} AmbientCG material packs match {args.resolution}-{args.format}")
    if args.dry_run:
        for item in items[:20]:
            print(item[0])
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        results = list(pool.map(lambda item: download_one(item, args.output), items))
    print(f"downloaded={results.count('ok')} skipped={results.count('skip')}")


if __name__ == "__main__":
    main()
