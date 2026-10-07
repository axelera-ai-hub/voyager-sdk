# Copyright Axelera AI, 2026
"""Getting the two files a build needs: the weight, and the calibration images.

Both are plain HTTPS fetches with a checksum, cached under one directory and
reused. Nothing here knows anything about a model, which is why it is here and
not in the recipe.

What stays in the recipe is the part you change to build your own model: the
URL and checksum it declares, `load_source()`, and `CALIBRATION_DIR`. This
module is what those declarations are handed to.

Imports no zoo: a recipe builds, runs and benchmarks with `axelera-zoo`
uninstalled, and this is on that path.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def md5(path: Path) -> str:
    """Hex md5 of a file, read in chunks so a weight does not land in memory."""
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str = '') -> Path:
    """Download `url` to `dest` once, verifying `expected_md5`.

    A later run reuses the file. The download lands in a `.partial` sibling and
    is renamed only after the checksum passes, so an interrupted transfer
    cannot leave a truncated file that every later run then trusts. `url` is
    `s3://` for an in-development (QA-only) weight, AXZOO_ENDPOINT-driven --
    never seen by a customer build, which only ever gets an `https://` CDN url.
    """
    if dest.is_file() and (not expected_md5 or md5(dest) == expected_md5):
        return dest
    if os.environ.get('AXZOO_OFFLINE', '0') == '1':
        raise FileNotFoundError(
            f'Asset not in cache and offline mode is enabled (AXZOO_OFFLINE=1). '
            f'Expected path: {dest}'
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'[fetch] {url}')
    partial = dest.with_suffix(dest.suffix + '.partial')
    try:
        if url.startswith('s3://'):
            _fetch_s3(url, partial, dest)
        else:
            _fetch_http(url, partial, dest)
        if expected_md5 and md5(partial) != expected_md5:
            raise SystemExit(f'checksum mismatch for {url}; expected md5 {expected_md5}')
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def _fetch_http(url: str, partial: Path, dest: Path) -> None:
    try:
        with urllib.request.urlopen(url) as response, open(partial, 'wb') as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            raise FileNotFoundError(
                'Asset not available at '
                f'{url} (HTTP {error.code}); it may not be published yet. '
                f'Expected cache path: {dest}. Export the model locally or '
                'prepopulate this cache.'
            ) from None
        raise


def _fetch_s3(url: str, partial: Path, dest: Path) -> None:
    """Download an `s3://bucket/key` asset with boto3 -- the internal SoT bucket.

    Mirrors the zoo's own `_download.py` S3 downloader: same bucket/key parse,
    same boto3-missing guidance, same not-published framing (a pending/validated
    model whose weight was never pushed to SoT is a publish gap, not a
    customer-facing 404).
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError(
            f'boto3 is required for S3 downloads (AXZOO_ENDPOINT={url}). '
            'Install it with: pip install boto3, or switch AXZOO_ENDPOINT to an HTTPS URL.'
        ) from None

    bucket, _, key = url[len('s3://') :].partition('/')
    try:
        boto3.client('s3').download_file(bucket, key, str(partial))
    except Exception as error:
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            raise
        code = ''
        if isinstance(error, ClientError):
            code = str(error.response.get('Error', {}).get('Code', ''))
        if code in ('403', '404', 'NoSuchKey', 'NoSuchBucket', 'AccessDenied'):
            raise FileNotFoundError(
                f'Weight not found in the SoT bucket at {url}; it may not be published yet.\n'
                f'  A pending/validated model must have its source weight published to SoT.\n'
                f'  Publish it once (engineer): axzoo export <config> --publish\n'
                f'  Expected cache path: {dest}.'
            ) from None
        raise


def unpack_once(url: str, expected_md5: str, root: Path) -> Path:
    """Download and unzip `url` into `root` once, and return `root`.

    Unpacked into a staging directory and renamed on success, for the same
    reason `fetch` uses a `.partial`: a half-populated cache directory is
    indistinguishable from a complete one on the next run.
    """
    if root.is_dir():
        return root
    staging = root.with_name(root.name + '.partial')
    shutil.rmtree(staging, ignore_errors=True)
    archive = fetch(url, staging / Path(url).name, expected_md5)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(staging)
    archive.unlink()
    staging.rename(root)
    return root


def calibration_images(
    *, directory: str, url: str, expected_md5: str, cache_dir: Path, limit: int
) -> list[Path]:
    """The images to calibrate on: `directory` when set, else the zoo's subset.

    `directory` is the recipe's `CALIBRATION_DIR`, and it wins so that pointing
    a recipe at your own scene needs no download and no argument.
    """
    if directory:
        root = Path(directory).expanduser()
    else:
        try:
            root = unpack_once(url, expected_md5, cache_dir / 'calibration' / Path(url).stem)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Calibration images not available at {url}. Point the recipe\'s '
                'CALIBRATION_DIR at your own images instead, which needs no download.'
            ) from None
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f'no {"/".join(IMAGE_SUFFIXES)} calibration images under {root}')
    return files[:limit]


def preprocessed(path: Path, preprocess):
    """One image through `preprocess`, exactly as inference will see it.

    The runtime's Image wrapper carries the RGB source format, without which a
    preprocess opening on `op.color_convert` cannot tell what it converts from.
    """
    import numpy as np
    from axelera.runtime import Image
    from PIL import Image as PilImage

    array = np.asarray(PilImage.open(path).convert('RGB'), dtype=np.uint8)
    return preprocess(Image.from_any(array, fallback_color_format='RGB'))


def preprocessed_samples(paths, preprocess):
    """Yield every image in `paths` preprocessed, one at a time."""
    return (preprocessed(path, preprocess) for path in paths)


def torch_loader(paths, preprocess, batch_size: int):
    """A torch DataLoader over `paths`, opening one image at a time.

    Torch is imported here rather than at module scope: only the axgraph build
    path calls this, and the axrelay path must not need torch installed.
    """
    import torch

    def collate(batch_paths):
        return torch.stack([torch.as_tensor(preprocessed(p, preprocess)) for p in batch_paths])

    return torch.utils.data.DataLoader(
        list(paths), batch_size=batch_size, shuffle=False, collate_fn=collate
    )
