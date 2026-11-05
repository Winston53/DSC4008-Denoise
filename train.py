from pathlib import Path

import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from denoise_dataset import SingleFrameDenoiseDataset
from nafnet import NAFNet
from burst_utils import display_rgb

def main():
    torch.manual_seed(4008)

    root = Path(__file__).resolve().parent
    image_dir = root / "data" / "open-images-v6" / "train" / "data"
    checkpoint_dir = root / "checkpoints_2000"
    checkpoint_dir.mkdir(exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_data = SingleFrameDenoiseDataset(image_dir, training = True)
    validation_data = SingleFrameDenoiseDataset(image_dir, training = False)

    assert training_data.image_paths == validation_data.image_paths

    count = len(training_data)
    if count < 2:
        raise RuntimeError(f"Not enough samples in dataset: {count}")
    
    generator = torch.Generator().manual_seed(4008)
    indices = torch.randperm(count, generator = generator).tolist()

    validation_count = max(1, round(count * 0.2))
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    training_loader = DataLoader(Subset(training_data, training_indices), batch_size = 8, shuffle = True, num_workers = 2)
    validation_loader = DataLoader(Subset(validation_data, validation_indices), batch_size = 4, shuffle = False, num_workers = 0)

    print(f"Device: {device}")
    print(f"Training photos: {len(training_indices)}")
    print(f"Validation photos: {len(validation_indices)}")

    model = NAFNet(
        img_channel = 3,
        width = 32,
        enc_blk_nums = [2, 2, 4, 8],
        middle_blk_num = 12,
        dec_blk_nums = [2, 2, 2, 2],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = torch.nn.L1Loss()

    epochs = 50
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        training_total = 0.0
        
        training_bar = tqdm(
            training_loader,
            desc=f"Epoch {epoch}/{epochs} Training",
            unit="batch",
        )

        for batch in training_bar:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            optimizer.zero_grad(set_to_none = True)
            prediction = model(noisy)
            loss = loss_fn(prediction, clean)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became NaN or infinite.")
                
            loss.backward()
            optimizer.step()
            training_total += loss.item() * noisy.size(0)
            training_bar.set_postfix(L1=f"{loss.item():.6f}")

        training_loss = training_total / len(training_indices)

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc = f"Epoch {epoch}/{epochs} Validation", unit = "batch", leave = False,):
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                prediction = model(noisy)
                loss = loss_fn(prediction, clean)

                if not torch.isfinite(loss).item():
                    raise RuntimeError("Validation loss became NaN or infinite.")
                    
                validation_total += loss.item() * noisy.size(0)
                
        validation_loss = validation_total / len(validation_indices)

        print(f"Epoch {epoch:2d}/{epochs} | Training L1 loss: {training_loss:.6f} | Validation L1 loss: {validation_loss:.6f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_path = checkpoint_dir / "nafnet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "training_images": [
                        str(training_data.image_paths[i])
                        for i in training_indices
                    ],
                    "validation_images": [
                        str(validation_data.image_paths[i])
                        for i in validation_indices
                    ],
                },
                checkpoint_path
            )
            print(f"Saved best model checkpoint to: {checkpoint_path}")
    print(f"Best validation L1: {best_validation_loss:.6f}")

def preview_results():
    root = Path(__file__).resolve().parent
    image_dir = root / "data" / "open-images-v6" / "train" / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        root / "checkpoints_2000" / "nafnet_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    dataset = SingleFrameDenoiseDataset(image_dir, training=False)

    model = NAFNet(
        img_channel=3,
        width=32,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use three photographs from the saved validation split
    validation_paths = checkpoint["validation_images"][:3]
    fig, axes = plt.subplots(
        len(validation_paths), 3,
        figsize=(10, 3 * len(validation_paths)),
        squeeze=False,
    )

    with torch.no_grad():
        for row, path in enumerate(validation_paths):
            idx = dataset.image_paths.index(Path(path))
            sample = dataset[idx]

            noisy = sample["noisy"].unsqueeze(0).to(device)
            prediction = model(noisy)[0].cpu()

            white_level = sample["white_level"].item()

            images = [
                sample["noisy"],
                prediction,
                sample["clean"],
            ]

            for col, (image, title) in enumerate(zip(
                images,
                ["Noisy input", "NAFNet prediction", "Clean target"],
            )):
                image = image.permute(1, 2, 0).numpy()
                axes[row, col].imshow(display_rgb(image, white_level))
                axes[row, col].set_title(title)
                axes[row, col].axis("off")

    fig.suptitle(f"Saved model — epoch {checkpoint['epoch']}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview_results()
    else:
        main()