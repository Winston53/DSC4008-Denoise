"""Build a held-out color denoising test set using Burstormer's four gains."""

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from filelock import FileLock
from PIL import Image
from urllib3 import PoolManager
from urllib3.util import Retry, Timeout

from burst_utils import generate_clean_burst
from download_sample import (
    DEFAULT_PROJECT_DIR,
    IMAGE_ID_PATTERN,
    photo_digest,
    require_inside_project,
    save_file,
)


SCENE_COUNT = 100
TEST_SEED = 4008
OUTPUT_FOLDER = Path("data/color_test_burstormer_protocol")
TRAINING_FOLDER = Path("data/nafnet_20000/open-images-v6/train")
INDEX_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation"
# Base-10 exponents in the (read, shot) order printed in Burstormer section 4.3.
GAINS = {1: (-2.2, -2.6), 2: (-1.8, -2.2), 4: (-1.4, -1.8), 8: (-1.1, -1.5)}
WORKERS = 8


def training_photo_keys(folder, project_dir):
    require_inside_project(folder, project_dir)
    manifest = json.loads((folder / "training_photos.json").read_text(encoding="utf-8"))
    filenames = manifest["photos"]
    if len(filenames) != manifest["count"] or len(set(filenames)) != len(filenames):
        raise ValueError("The training photo list has an invalid count or duplicate IDs.")

    ids = set()
    hashes = set()
    for name in filenames:
        path = Path(name)
        if path.name != name or path.suffix != ".jpg" or not IMAGE_ID_PATTERN.fullmatch(path.stem):
            raise ValueError(f"Unexpected filename in the training list: {name}")
        photo_path = folder / "data" / name
        require_inside_project(photo_path, project_dir)
        with photo_path.open("rb") as photo:
            hashes.add(hashlib.file_digest(photo, "sha256").hexdigest())
        ids.add(path.stem)
    return ids, hashes


def get_validation_index(client, path):
    if not path.exists():
        response = client.request("GET", INDEX_URL, preload_content=False)
        try:
            if response.status != 200:
                raise RuntimeError(f"Open Images validation index returned HTTP {response.status}.")
            save_file(response, path)
        finally:
            response.close()
            response.release_conn()

    with path.open(encoding="utf-8-sig", newline="") as file:
        records = {row["ImageID"]: row for row in csv.DictReader(file)}
    return records


def download_candidate(client, image_id):
    response = client.request("GET", f"{IMAGE_URL}/{image_id}.jpg")
    if response.status in (404, 410):
        return image_id, None
    if response.status != 200:
        raise RuntimeError(f"Validation image {image_id} returned HTTP {response.status}.")
    return image_id, response.data


def select_test_photos(client, output_dir, project_dir, train_ids, train_hashes):
    source_dir = output_dir / "source_photos"
    index_path = output_dir / "validation_image_ids.csv"
    records = get_validation_index(client, index_path)
    source_dir.mkdir(exist_ok=True)

    saved = {}
    used_hashes = set(train_hashes)
    for path in sorted(source_dir.glob("*.jpg")):
        require_inside_project(path, project_dir)
        digest = photo_digest(path.read_bytes())
        if path.stem not in records or path.stem in train_ids or digest is None or digest in used_hashes:
            raise ValueError(f"Saved test photo is invalid or overlaps another photo: {path}")
        saved[path.stem] = path
        used_hashes.add(digest)
    if len(saved) > SCENE_COUNT:
        raise ValueError(f"Expected at most {SCENE_COUNT} cached validation JPEGs.")

    candidates = sorted(
        image_id for image_id in records
        if IMAGE_ID_PATTERN.fullmatch(image_id) and image_id not in train_ids and image_id not in saved
    )
    random.Random(TEST_SEED).shuffle(candidates)
    remaining = iter(candidates)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while len(saved) < SCENE_COUNT:
            batch = list(islice(remaining, min(32, SCENE_COUNT - len(saved))))
            if not batch:
                raise RuntimeError(f"Only {len(saved)} eligible held-out photos were available.")
            requests = [pool.submit(download_candidate, client, image_id) for image_id in batch]
            for request in requests:
                image_id, contents = request.result()
                if contents is None:
                    continue
                digest = photo_digest(contents)
                if digest is None or digest in used_hashes:
                    continue
                path = source_dir / f"{image_id}.jpg"
                save_file(BytesIO(contents), path)
                saved[image_id] = path
                used_hashes.add(digest)
            print(f"Held-out photos: {len(saved)} / {SCENE_COUNT}", flush=True)

    photos = [saved[image_id] for image_id in sorted(saved)]
    credits = [
        {
            "image_id": path.stem,
            "author": records[path.stem].get("Author", ""),
            "source": records[path.stem].get("OriginalLandingURL", ""),
            "license": records[path.stem].get("License", ""),
        }
        for path in photos
    ]
    return photos, credits


def make_clean_bursts(photos):
    bursts = []
    white_levels = []
    for photo in photos:
        rng = np.random.default_rng(np.random.SeedSequence([TEST_SEED, int(photo.stem, 16)]))
        with Image.open(photo) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
        burst, _ = generate_clean_burst(rgb, rng)
        white = np.float32(10 ** rng.uniform(-1, 0))
        clean = (white * burst ** np.float32(2.2)).astype(np.float32, copy=False)
        bursts.append(clean.transpose(1, 2, 0, 3))
        white_levels.append(white)
    return np.stack(bursts), np.asarray(white_levels, dtype=np.float32)


def write_gain_file(destination, clean_bursts, white_levels, image_ids, gain):
    log_read, log_shot = GAINS[gain]
    read = np.float32(10 ** log_read)
    shot = np.float32(10 ** log_shot)
    noisy = np.empty_like(clean_bursts)
    for index, image_id in enumerate(image_ids):
        rng = np.random.default_rng(np.random.SeedSequence([TEST_SEED, int(image_id, 16), gain]))
        clean = clean_bursts[index]
        read_noise = read * rng.standard_normal(clean.shape, dtype=np.float32)
        shot_noise = shot * np.sqrt(clean) * rng.standard_normal(clean.shape, dtype=np.float32)
        noisy[index] = clean + read_noise + shot_noise

    temporary_path = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as file:
            temporary_path = Path(file.name)
            np.savez(
                file,
                noisy=noisy,                             # N, H, W, 8 frames, RGB
                truth=clean_bursts[:, :, :, 0, :],        # N, H, W, RGB
                white_level=white_levels,
                sig_read=np.full(len(image_ids), read, dtype=np.float32),
                sqrt_sig_shot=np.full(len(image_ids), shot, dtype=np.float32),
                image_id=np.asarray(image_ids, dtype="U16"),
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_test_set(project_dir):
    if not project_dir.is_absolute():
        raise ValueError("--project-dir must be an absolute path.")
    project = project_dir.resolve(strict=True)
    if not project.is_dir() or project.anchor != project_dir.anchor:
        raise ValueError(f"Project directory is invalid or redirects to another drive: {project_dir}")

    output = project / OUTPUT_FOLDER
    manifest = output / "test_manifest.json"
    lock = output / "build.lock"
    gain_paths = {gain: output / f"{gain}.npz" for gain in GAINS}
    for path in (output / "source_photos", output / "validation_image_ids.csv",
                 manifest, lock, *gain_paths.values()):
        require_inside_project(path, project)
    output.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock), timeout=0):
        if manifest.is_file():
            if not all(path.is_file() for path in gain_paths.values()):
                raise ValueError(f"The completed test set is missing a gain file: {output}")
            return manifest

        print("Checking training photo IDs and exact file duplicates.", flush=True)
        train_ids, train_hashes = training_photo_keys(project / TRAINING_FOLDER, project)
        retries = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                        raise_on_status=False)
        with PoolManager(maxsize=WORKERS, block=True, retries=retries,
                         timeout=Timeout(connect=10, read=30)) as client:
            photos, credits = select_test_photos(client, output, project, train_ids, train_hashes)

        clean_bursts, white_levels = make_clean_bursts(photos)
        image_ids = [path.stem for path in photos]
        for gain, path in gain_paths.items():
            write_gain_file(path, clean_bursts, white_levels, image_ids, gain)
            print(f"Wrote {path.name}: {len(image_ids)} fixed test cases.", flush=True)

        definition = {
            "protocol": "Custom color test set using Burstormer section 4.3 gain settings",
            "source_split": "Open Images V6 validation",
            "training_photo_list": (TRAINING_FOLDER / "training_photos.json").as_posix(),
            "seed": TEST_SEED,
            "scene_count": len(image_ids),
            "case_count": len(image_ids) * len(GAINS),
            "frame_count": 8,
            "frame_size": [128, 128],
            "reference_frame": 0,
            "log10_sigma_read_shot": GAINS,
            "scenes": credits,
        }
        save_file(BytesIO((json.dumps(definition, indent=2) + "\n").encode()), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    args = parser.parse_args()
    manifest = build_test_set(args.project_dir)
    print(f"Color test set ready: {manifest}")


if __name__ == "__main__":
    main()
