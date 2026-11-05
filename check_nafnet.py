from pathlib import Path

import torch
from torch.utils.data import DataLoader

from denoise_dataset import SingleFrameDenoiseDataset
from nafnet import NAFNet


def main():
    torch.manual_seed(4008)

    image_dir = (Path(__file__).resolve().parent / "data" / "open-images-v6" / "train" / "data")

    dataset = SingleFrameDenoiseDataset(image_dir, training = False)
    loader = DataLoader(dataset, batch_size = 4, num_workers = 0)
    batch = next(iter(loader))

    # Random initialization: no pretrained checkpoint loaded
    model = NAFNet(
        img_channel = 3,
        width = 32,
        enc_blk_nums = [2, 2, 4, 8],
        middle_blk_num = 12,
        dec_blk_nums = [2, 2, 2, 2],
    ).to("cuda")

    noisy = batch["noisy"].to("cuda")
    clean = batch["clean"].to("cuda")

    model.eval()

    with torch.no_grad():
        prediction = model(noisy)
        loss = torch.nn.functional.l1_loss(prediction, clean)

    assert prediction.shape == clean.shape
    assert torch.isfinite(prediction).all().item()

    print("Input:", noisy.shape)
    print("Output:", prediction.shape)
    print("Target:", clean.shape)
    print(f"Initial L1 loss: {loss.item():.6f}")
    print("Forward pass passed")


if __name__ == "__main__":
    main()