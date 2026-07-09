#!/usr/bin/env python3
"""Download official COCO val2014 and extract the exact image IDs used by coco_30k.csv."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# The official host does not serve TLS reliably from every RunPod region.
# Integrity is enforced with the published archive checksum below.
COCO_VAL2014_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_VAL2014_MD5 = "a3d79f5ed8d289b7a7554ce06a5782b3"


class DownloadProgress(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_progress(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--progress_path", required=True)
    parser.add_argument("--keep_archive", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    cache = Path(args.cache_dir)
    progress = Path(args.progress_path)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    ids = [int(value) for value in pd.read_csv(args.prompts_path)["image_id"].tolist()]
    if len(ids) != 30000 or len(set(ids)) != 30000:
        raise RuntimeError(f"Expected 30,000 unique COCO image IDs, found {len(ids)} rows/{len(set(ids))} unique")
    complete = sum((output / f"{image_id}.jpg").exists() for image_id in ids)
    if complete == len(ids):
        print(f"Official COCO reference already complete: {complete:,} images")
        return

    archive = cache / "val2014.zip"
    if not archive.exists() or md5sum(archive) != COCO_VAL2014_MD5:
        archive.unlink(missing_ok=True)
        append_progress(progress, {"phase": "coco_reference_download", "status": "started"})
        with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc="val2014.zip") as bar:
            urllib.request.urlretrieve(COCO_VAL2014_URL, archive, reporthook=bar.update_to)
    actual_md5 = md5sum(archive)
    if actual_md5 != COCO_VAL2014_MD5:
        raise RuntimeError(f"COCO archive checksum mismatch: expected {COCO_VAL2014_MD5}, got {actual_md5}")

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for index, image_id in enumerate(tqdm(ids, desc="Extracting COCO-30k"), start=1):
            destination = output / f"{image_id}.jpg"
            if destination.exists():
                continue
            member = f"val2014/COCO_val2014_{image_id:012d}.jpg"
            if member not in names:
                raise RuntimeError(f"COCO image ID {image_id} is absent from official val2014 archive")
            with zf.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            if index % 1000 == 0:
                append_progress(
                    progress,
                    {"phase": "coco_reference_extract", "completed_images": index, "total_images": len(ids)},
                )

    complete = sum((output / f"{image_id}.jpg").exists() for image_id in ids)
    if complete != len(ids):
        raise RuntimeError(f"COCO reference has {complete:,}/{len(ids):,} images after extraction")
    append_progress(progress, {"phase": "coco_reference", "status": "completed", "total_images": complete})
    if not args.keep_archive:
        archive.unlink()
    print(f"Official COCO reference ready: {complete:,} images in {output}")


if __name__ == "__main__":
    main()
