"""Score a saved single-frame NAFNet on four fixed color gain files."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from burst_utils import display_rgb
from nafnet import NAFNet
from train_pilot import MODEL_CONFIG


GAINS = (1, 2, 4, 8)


def image_scores(prediction, target):
    error = prediction[8:-8, 8:-8] - target[8:-8, 8:-8]
    psnr = -10 * np.log10(np.mean(error ** 2, dtype=np.float64) + 1e-6)
    ssim = structural_similarity(
        target, prediction, data_range=1.0, channel_axis=-1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    return float(psnr), float(ssim)


def evaluate_gain(model, path, device):
    with np.load(path, allow_pickle=False) as archive:
        frame_zero = np.ascontiguousarray(archive["noisy"][:, :, :, 0, :], dtype=np.float32)
        targets = archive["truth"]
        white_levels = archive["white_level"]
        image_ids = archive["image_id"]

    scores = []
    example = None
    with torch.inference_mode():
        for image_id, noisy, target, white in zip(
            image_ids, frame_zero, targets, white_levels, strict=True
        ):
            white = float(np.asarray(white).item())
            if not np.isfinite(white) or white <= 0:
                raise ValueError(f"Invalid white level for {image_id} in {path}")
            input_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(input_tensor)[0].permute(1, 2, 0).cpu().numpy()
            if not np.isfinite(output).all():
                raise FloatingPointError(f"Non-finite model output for {image_id} in {path}")

            noisy_rgb = display_rgb(noisy, white)
            model_rgb = display_rgb(np.clip(output, 0, 1), white)
            target_rgb = display_rgb(target, white)
            noisy_psnr, noisy_ssim = image_scores(noisy_rgb, target_rgb)
            model_psnr, model_ssim = image_scores(model_rgb, target_rgb)
            scores.append((noisy_psnr, model_psnr, noisy_ssim, model_ssim))
            if example is None:
                example = (str(image_id), noisy_rgb, model_rgb, target_rgb)

    if not scores:
        raise ValueError(f"No test cases in {path}")
    return len(scores), np.mean(scores, axis=0), example


def save_examples(examples, path):
    tile = 192
    gap = 12
    row_height = tile + 29
    canvas = Image.new("RGB", (3 * tile + 4 * gap, 44 + len(examples) * row_height), "#f2f5f8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for column, heading in enumerate(("Noisy frame 0", "NAFNet output", "Clean target")):
        draw.text((gap + column * (tile + gap), 12), heading, font=font, fill="#17283b")
    for row, (gain, (image_id, *images)) in enumerate(examples):
        y = 44 + row * row_height
        for column, values in enumerate(images):
            pixels = np.rint(values * 255).astype(np.uint8)
            tile_image = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(tile_image, (gap + column * (tile + gap), y))
        draw.text((gap, y + tile + 4), f"Gain {gain}   Image {image_id}", font=font, fill="#17283b")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases-per-gain", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for evaluation.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = NAFNet(**checkpoint.get("model_config", MODEL_CONFIG)).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    examples = []
    for gain in GAINS:
        count, scores, example = evaluate_gain(model, args.test_dir / f"{gain}.npz", args.device)
        if args.expected_cases_per_gain is not None and count != args.expected_cases_per_gain:
            raise ValueError(f"Gain {gain} has {count} cases, expected {args.expected_cases_per_gain}.")
        noisy_psnr, model_psnr, noisy_ssim, model_ssim = scores
        rows.append((gain, count, noisy_psnr, model_psnr, model_psnr - noisy_psnr,
                     noisy_ssim, model_ssim, model_ssim - noisy_ssim))
        examples.append((gain, example))
        print(f"Gain {gain} ({count} cases): PSNR {noisy_psnr:.3f} → {model_psnr:.3f} dB, "
              f"SSIM {noisy_ssim:.4f} → {model_ssim:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics_by_gain.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("gain", "cases", "noisy_psnr_db", "nafnet_psnr_db", "psnr_change_db",
                         "noisy_ssim", "nafnet_ssim", "ssim_change"))
        writer.writerows(rows)
    save_examples(examples, args.output / "comparison.png")


if __name__ == "__main__":
    main()
