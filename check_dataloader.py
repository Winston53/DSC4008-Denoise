from pathlib import Path

import torch
from torch.utils.data import DataLoader

from denoise_dataset import SingleFrameDenoiseDataset


def main():
    image_dir = (Path(__file__).resolve().parent / "data" / "open-images-v6" / "train" / "data")

    dataset = SingleFrameDenoiseDataset(image_dir, training=True)

    loader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 0, pin_memory=True)

    batch = next(iter(loader))

    print("Input batch:", batch["noisy"].shape)
    print("Target batch:", batch["clean"].shape)
    print("White levels:", batch["white_level"].shape)

    noisy = batch["noisy"].to("cuda")
    clean = batch["clean"].to("cuda")

    assert torch.isfinite(noisy).all().item()
    assert torch.isfinite(clean).all().item()

    print("Device:", noisy.device)
    print("Batch check passed")


if __name__ == "__main__":
    main()