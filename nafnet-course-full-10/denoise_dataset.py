"""Load single-frame NAFNet training pairs from Open Images photos."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from burst_utils import add_bpn_noise, generate_clean_burst


PROJECT_DIR = Path(__file__).resolve().parent
PILOT_IMAGE_DIR = PROJECT_DIR / "data" / "open-images-v6" / "train" / "data"
MIN_PHOTO_SIDE = 640


def photo_is_large_enough(path):
    with Image.open(path) as photo:
        width, height = photo.size
    return width >= MIN_PHOTO_SIDE and height >= MIN_PHOTO_SIDE


def read_training_paths(image_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filenames = manifest.get("photos")
    if not isinstance(filenames, list) or not filenames or len(filenames) != manifest.get("count"):
        raise ValueError(f"The photo count is invalid in {manifest_path}")

    paths = []
    seen = set()
    for filename in filenames:
        if not isinstance(filename, str):
            raise ValueError(f"Invalid photo name in {manifest_path}: {filename!r}")
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise ValueError(f"Invalid JPEG filename in {manifest_path}: {filename}")
        if filename.casefold() in seen:
            raise ValueError(f"Duplicate photo in {manifest_path}: {filename}")

        path = image_dir / filename
        if not photo_is_large_enough(path):
            raise ValueError(f"Photo must be at least {MIN_PHOTO_SIDE} pixels on both sides: {path}")
        paths.append(path)
        seen.add(filename.casefold())
    return paths


class SingleFrameDenoiseDataset(Dataset):
    def __init__(self, image_dir, training=True, rng_seed=4008, *, manifest_path=None):
        self.training = training
        self.seed = rng_seed
        image_dir = Path(image_dir)

        if manifest_path is None:
            photos = sorted(image_dir.glob("*.jpg"))
            self.image_paths = [path for path in photos if photo_is_large_enough(path)]
            print(f"Usable photographs: {len(self.image_paths)} / {len(photos)}")
        else:
            self.image_paths = read_training_paths(image_dir, Path(manifest_path))
            print(f"Training photos from manifest: {len(self.image_paths):,}")

        if not self.image_paths:
            raise ValueError(f"No usable images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.training:
            sample_seed = torch.randint(0, 2**32, ()).item()
        else:
            sample_seed = self.seed + idx

        rng = np.random.default_rng(sample_seed)

        with Image.open(self.image_paths[idx]) as photo:
            rgb = np.array(photo.convert("RGB"), dtype=np.float32) / 255.0

        clean_burst, _ = generate_clean_burst(rgb, rng)
        sample = add_bpn_noise(clean_burst, rng)

        # NAFNet learns from only the first noisy frame and its clean target.
        noisy = torch.from_numpy(sample["noisy_burst"][0]).permute(2, 0, 1).contiguous()
        clean = torch.from_numpy(sample["clean_target"]).permute(2, 0, 1).contiguous()

        return {
            "noisy": noisy,
            "clean": clean,
            "white_level": torch.tensor(sample["white_level"], dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description="Inspect one NAFNet training example.")
    parser.add_argument("--manifest", type=Path, help="Path to the completed training photo list.")
    args = parser.parse_args()

    if args.manifest is None:
        image_dir = PILOT_IMAGE_DIR
    else:
        image_dir = args.manifest.parent / "data"

    dataset = SingleFrameDenoiseDataset(image_dir, manifest_path=args.manifest)
    sample = dataset[0]

    print("Input:", sample["noisy"].shape, sample["noisy"].dtype)
    print("Target:", sample["clean"].shape, sample["clean"].dtype)
    print("White level:", sample["white_level"].item())


if __name__ == "__main__":
    main()
