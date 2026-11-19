"""
Trains on one fixed batch for 100 steps and prints before-and-after L1 loss.
Useful for debugging backpropagation, though it does not assert that loss improves.
"""

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.L1Loss()

    model.train()

    # Keep noisy and clean fixed throughout this check
    with torch.no_grad():
        initial_loss = loss_fn(model(noisy), clean).item()

    print(f"Before training: {initial_loss:.6f}")

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(noisy)
        loss = loss_fn(prediction, clean)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss became NaN or infinite.")

        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(f"Step {step:3d}/100 — L1 loss: {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        final_loss = loss_fn(model(noisy), clean).item()

    print(f"After training:  {final_loss:.6f}")

if __name__ == "__main__":
    main()