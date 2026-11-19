"""Download an exact number of usable Open Images training photos."""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from urllib.request import urlopen

from filelock import FileLock, Timeout
from PIL import Image
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError
from urllib3.util import Retry, Timeout as HttpTimeout


DEFAULT_PROJECT_DIR = Path("S:/DSC4008-denoising")
TARGET_COUNT = 20_000
MIN_PHOTO_SIDE = 640  # The crop recipe starts with a 640 x 640 region.
SEED = 4008
DOWNLOAD_WORKERS = 24
INDEX_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train"
IMAGE_ID_PATTERN = re.compile(r"[0-9a-f]{16}")


def require_inside_project(path, project_dir):
    if not path.resolve().is_relative_to(project_dir):
        raise ValueError(f"Path points outside the project directory: {path}")


def save_file(source, destination):
    # Keep partial files beside the destination, on the same drive.
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(source, temporary_file)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_image_ids(index_path):
    with index_path.open(encoding="utf-8-sig", newline="") as file:
        image_ids = {row["ImageID"].lower() for row in csv.DictReader(file)}

    image_ids = sorted(
        image_id for image_id in image_ids if IMAGE_ID_PATTERN.fullmatch(image_id)
    )
    random.Random(SEED).shuffle(image_ids)
    return image_ids


def photo_digest(contents):
    """Return a fingerprint only when the JPEG meets the training size rule."""
    try:
        with Image.open(BytesIO(contents)) as photo:
            if photo.format != "JPEG" or min(photo.size) < MIN_PHOTO_SIDE:
                return None
            photo.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return None
    return hashlib.sha256(contents).hexdigest()


def download_photo(client, image_id):
    url = f"{IMAGE_URL}/{image_id}.jpg"
    try:
        response = client.request("GET", url)
    except HTTPError as error:
        raise RuntimeError(f"Could not download image {image_id}") from error

    if response.status in (404, 410):
        return image_id, None, None
    if response.status != 200:
        raise RuntimeError(f"Could not download image {image_id}: HTTP {response.status}")

    return image_id, response.data, photo_digest(response.data)


def scan_saved_photos(image_dir, project_dir, count):
    photos = {}
    digests = set()
    for path in sorted(image_dir.glob("*.jpg")):
        require_inside_project(path, project_dir)
        digest = photo_digest(path.read_bytes())
        if not IMAGE_ID_PATTERN.fullmatch(path.stem) or digest is None or digest in digests:
            raise ValueError(f"Invalid or duplicate photo in the output folder: {path}")
        photos[path.stem] = path.name
        digests.add(digest)

    if len(photos) > count:
        raise ValueError(f"Found {len(photos):,} saved photos but expected at most {count:,}.")
    return photos, digests


def fill_photo_folder(image_dir, image_ids, photos, digests, count):
    remaining_ids = (image_id for image_id in image_ids if image_id not in photos)
    retries = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    pending = deque()
    next_report = monotonic() + 5

    # Reuse HTTPS connections and keep workers busy between validation results.
    with (
        PoolManager(
            maxsize=DOWNLOAD_WORKERS,
            block=True,
            retries=retries,
            timeout=HttpTimeout(connect=10, read=25),
        ) as client,
        ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as workers,
    ):
        while len(photos) < count:
            queue_size = min(3 * DOWNLOAD_WORKERS, count - len(photos))
            while len(pending) < queue_size:
                image_id = next(remaining_ids, None)
                if image_id is None:
                    break
                pending.append(workers.submit(download_photo, client, image_id))

            if not pending:
                raise RuntimeError(f"Only {len(photos):,} usable photos were available.")

            # Consume results in seeded order so timing cannot change the photo list.
            image_id, contents, digest = pending.popleft().result()
            if digest is not None and digest not in digests:
                filename = f"{image_id}.jpg"
                save_file(BytesIO(contents), image_dir / filename)
                photos[image_id] = filename
                digests.add(digest)

            if len(photos) == count or monotonic() >= next_report:
                print(f"Accepted {len(photos):,} / {count:,} photos.", flush=True)
                next_report = monotonic() + 5


def prepare_training_set(project_dir, count):
    if not project_dir.is_absolute():
        raise ValueError("--project-dir must be an absolute path.")
    resolved_project = project_dir.resolve(strict=True)
    if not resolved_project.is_dir() or resolved_project.anchor != project_dir.anchor:
        raise ValueError(f"Project directory is invalid or redirects to another drive: {project_dir}")

    data_dir = resolved_project / "data"
    split_dir = data_dir / f"nafnet_{count}" / "open-images-v6" / "train"
    image_dir = split_dir / "data"
    manifest_path = split_dir / "training_photos.json"
    lock_path = split_dir / "prepare.lock"
    existing_index = data_dir / "open-images-v6" / "train" / "metadata" / "image_ids.csv"
    local_index = split_dir / "image_ids.csv"
    for path in (image_dir, manifest_path, lock_path, existing_index, local_index):
        require_inside_project(path, resolved_project)

    print(f"Writing dataset files under {split_dir}", flush=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=0):
        photos, digests = scan_saved_photos(image_dir, resolved_project, count)
        if len(photos) < count:
            if manifest_path.exists():
                raise ValueError(f"Photos are missing from the completed dataset: {image_dir}")
            index_path = existing_index if existing_index.is_file() else local_index
            if not index_path.is_file():
                with urlopen(INDEX_URL, timeout=60) as response:
                    save_file(response, index_path)
            print(f"Found {len(photos):,} saved photos. Checking new candidates.", flush=True)
            fill_photo_folder(image_dir, load_image_ids(index_path), photos, digests, count)

        manifest = {"count": count, "photos": sorted(photos.values())}
        if manifest_path.exists():
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved_manifest != manifest:
                raise ValueError(f"The finished photo folder disagrees with {manifest_path}")
        else:
            contents = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            save_file(BytesIO(contents), manifest_path)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR,
                        help="Absolute project path. Defaults to S:/DSC4008-denoising.")
    parser.add_argument("--count", type=int, default=TARGET_COUNT,
                        help="Exact number of photos to accept. Defaults to 20,000.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    try:
        manifest = prepare_training_set(args.project_dir, args.count)
    except Timeout:
        parser.exit(1, "Another download is already preparing this dataset.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Dataset preparation stopped: {error}\n")
    print(f"Verified {args.count:,} distinct photos in {manifest}")


if __name__ == "__main__":
    main()
