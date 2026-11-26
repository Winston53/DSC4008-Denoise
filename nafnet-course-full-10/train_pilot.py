"""Train single-frame NAFNet from a photo list and save epoch checkpoints."""

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from denoise_dataset import SingleFrameDenoiseDataset
from nafnet import LayerNorm2d, NAFNet


MODEL_CONFIG = {
    "img_channel": 3,
    "width": 32,
    "enc_blk_nums": [2, 2, 4, 8],
    "middle_blk_num": 12,
    "dec_blk_nums": [2, 2, 2, 2],
}
LEARNING_RATE = 1e-4


def train_epoch(model, loader, optimizer, loss_fn, device, fast=False):
    total_loss = 0.0
    memory_format = torch.channels_last if fast else torch.contiguous_format
    model.train()
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=fast, memory_format=memory_format)
        clean = batch["clean"].to(device, non_blocking=fast, memory_format=memory_format)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(noisy), clean)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        total_loss += loss_value * noisy.shape[0]
    return total_loss / len(loader.dataset)


def write_history(output, history):
    path = output / "training_loss.csv"
    pending = path.with_suffix(".csv.part")
    with pending.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "train_l1", "seconds"))
        writer.writerows(history)
    pending.replace(path)


def save_epoch(output, model, optimizer, settings, history, final, fast=False):
    target = output / ("nafnet_final.pt" if final else "nafnet_last.pt")
    pending = target.with_suffix(".pt.part")
    torch.save({
        "epoch": history[-1][0],
        "seed": settings["seed"],
        "training_photo_count": settings["training_photo_count"],
        "model_config": MODEL_CONFIG,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "settings": settings,
        "fast": fast,
        "training_history": history,
    }, pending)
    pending.replace(target)
    if final:
        (output / "nafnet_last.pt").unlink(missing_ok=True)
    write_history(output, history)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4008)
    parser.add_argument("--expected-photos", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use channels-last FP32, built-in layer normalization, and pinned data loading.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Epochs and batch size must be positive, and workers must be nonnegative.")
    if args.fast and args.device != "cuda":
        parser.error("--fast requires CUDA.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the Slurm GPU allocation and PyTorch build.")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(args.seed)

    dataset = SingleFrameDenoiseDataset(
        args.manifest.parent / "data", manifest_path=args.manifest
    )
    if args.expected_photos is not None and len(dataset) != args.expected_photos:
        raise ValueError(f"Expected {args.expected_photos:,} photos, found {len(dataset):,}.")
    sampler_rng = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=sampler_rng,
        num_workers=args.workers,
        pin_memory=args.fast,
    )
    model = NAFNet(**MODEL_CONFIG).to(device)
    if args.fast:
        model.to(memory_format=torch.channels_last)
        for layer in model.modules():
            if isinstance(layer, LayerNorm2d):
                layer.use_builtin = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.L1Loss()
    settings = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "training_photo_count": len(dataset),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    history = []
    first_epoch = 1
    if args.resume:
        checkpoint_path = args.output / "nafnet_last.pt"
        if not checkpoint_path.is_file():
            checkpoint_path = args.output / "nafnet_final.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["settings"] != settings or checkpoint["model_config"] != MODEL_CONFIG:
            raise ValueError("Resume requires the same model, training list, and data-loading settings.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["training_history"]
        first_epoch = checkpoint["epoch"] + 1
        write_history(args.output, history)
        print(f"Resuming after epoch {checkpoint['epoch']} from {checkpoint_path}", flush=True)
        if args.fast != checkpoint.get("fast", False) and first_epoch <= args.epochs:
            print(f"Changing fast mode at epoch {first_epoch} to {args.fast}.", flush=True)
        if first_epoch > args.epochs:
            if checkpoint["epoch"] != args.epochs:
                raise ValueError("The saved checkpoint is beyond the requested final epoch.")
            if checkpoint_path.name != "nafnet_final.pt":
                checkpoint_path.replace(args.output / "nafnet_final.pt")
            print(f"Final checkpoint: {args.output / 'nafnet_final.pt'}", flush=True)
            return

    print(f"Training to epoch {args.epochs} on {len(dataset):,} photos ({len(loader):,} batches/epoch).",
          flush=True)
    print(f"Batch size: {args.batch_size}; data workers: {args.workers}; fast mode: {args.fast}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        sampler_rng.manual_seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        started = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, loss_fn, device, fast=args.fast)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        history.append((epoch, loss, seconds))
        checkpoint = save_epoch(
            args.output, model, optimizer, settings, history, final=epoch == args.epochs, fast=args.fast
        )
        print(f"Epoch {epoch}/{args.epochs}: train L1 {loss:.6f}, {seconds:.1f}s; saved {checkpoint.name}",
              flush=True)
    print(f"Final checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
