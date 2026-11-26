"""Evaluate single-frame NAFNet on captured JPEGs using the group notebook's metrics."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean

import cv2
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ssim

from nafnet import NAFNet


TILE_SIZE = 512
OVERLAP = 64
GAMMA = 2.2
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENES_ROOT = PROJECT_DIR / "data" / "Self_Capture_Photos"
CROP_CATALOG = "comparison_crops.json"
METRIC_COLUMNS = ("noisy_psnr_db", "denoised_psnr_db", "psnr_gain_db", "noisy_ssim",
                  "denoised_ssim", "noisy_lpips", "denoised_lpips")


def tile_offsets(length):
    last = max(0, length - TILE_SIZE)
    offsets = list(range(0, last + 1, TILE_SIZE - OVERLAP))
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def blending_weights(length):
    weights = np.ones(length, dtype=np.float32)
    ramp_size = min(OVERLAP, length // 2)
    if ramp_size:
        ramp = np.linspace(0, 1, ramp_size + 2, dtype=np.float32)[1:-1]
        weights[:ramp_size] = ramp
        weights[-ramp_size:] = ramp[::-1]
    return weights


@torch.inference_mode()
def denoise_jpeg(model, rgb, device):
    height, width, _ = rgb.shape
    merged = np.zeros(rgb.shape, dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    tile_h, tile_w = min(TILE_SIZE, height), min(TILE_SIZE, width)
    window = np.outer(blending_weights(tile_h), blending_weights(tile_w))

    for top in tile_offsets(height):
        for left in tile_offsets(width):
            region = np.s_[top:top + tile_h, left:left + tile_w]
            # The training code approximates linear RGB with this same power transform.
            # A fixed white level of 1.0 uses the JPEG's recorded brightness scale.
            pixels = rgb[region].astype(np.float32) / 255.0
            linear_rgb = pixels ** GAMMA
            tensor = torch.from_numpy(linear_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            prediction = model(tensor)[0].permute(1, 2, 0).cpu().numpy()
            merged[region] += prediction * window[..., None]
            weights[region] += window

    merged /= weights[..., None]
    if not np.isfinite(merged).all():
        raise ValueError("The model produced non-finite pixels.")
    np.clip(merged, 0.0, 1.0, out=merged)
    np.power(merged, 1.0 / GAMMA, out=merged)
    merged *= 255.0
    return np.rint(merged, out=merged).astype(np.uint8)


def save_png(image, path):
    pending = path.with_suffix(".png.part")
    image.save(pending, format="PNG")
    pending.replace(path)


def save_labeled_pair(left, right, divider_width, path, output_label="NAFNet output"):
    height, width = left.shape[:2]
    divider = np.full((height, divider_width, 3), 255, dtype=np.uint8)
    preview = Image.fromarray(np.concatenate((left, divider, right), axis=1))
    draw = ImageDraw.Draw(preview, "RGBA")
    font = ImageFont.load_default(size=max(1, min(width, height) // 32))
    padding = max(2, font.size // 4)
    labels = ((0, "Noisy input (frame 0)"), (width + divider_width, output_label))
    for offset, label in labels:
        x, y = offset + padding, padding
        _, _, text_width, text_height = draw.textbbox((0, 0), label, font=font, anchor="lt")
        draw.rounded_rectangle(
            (x, y, x + text_width + 2 * padding, y + text_height + 2 * padding),
            radius=padding, fill=(16, 25, 34, 235),
        )
        draw.text((x + padding, y + padding), label, font=font, anchor="lt", fill="white")
    save_png(preview, path)


def save_comparison(noisy, restored, path, output_label="NAFNet output"):
    if noisy.shape != restored.shape:
        raise ValueError("Noisy and denoised images must have the same dimensions.")

    height, width = noisy.shape[:2]
    preview_height = 1280
    preview_size = (int(width * (preview_height / height)), preview_height)
    left = cv2.resize(noisy, preview_size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(restored, preview_size, interpolation=cv2.INTER_AREA)
    save_labeled_pair(left, right, 6, path, output_label)


def detail_crop_box(noisy):
    height, width = noisy.shape[:2]
    crop_h = max(1, int(height * 0.35))
    crop_w = max(1, int(width * 0.35))
    gray = cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY)
    edges = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    candidates = ((int(x), int(y)) for y in np.linspace(0, height - crop_h, 5, dtype=int)
                  for x in np.linspace(0, width - crop_w, 5, dtype=int))

    def detail(position):
        x, y = position
        return edges[y:y + crop_h, x:x + crop_w].mean()

    x, y = max(candidates, key=detail)
    return x, y, crop_w, crop_h


def save_crop_comparison(noisy, restored, path, crop_box=None, output_label="NAFNet output"):
    if noisy.shape != restored.shape:
        raise ValueError("Noisy and denoised images must have the same dimensions.")
    x, y, width, height = detail_crop_box(noisy) if crop_box is None else crop_box
    if (x < 0 or y < 0 or width < 1 or height < 1
            or x + width > noisy.shape[1] or y + height > noisy.shape[0]):
        raise ValueError(f"Crop {(x, y, width, height)} does not fit image shape {noisy.shape[:2]}.")
    region = np.s_[y:y + height, x:x + width]
    save_labeled_pair(noisy[region], restored[region], 4, path, output_label)


def read_rgb(path):
    # Decode the same way as the group notebook, including EXIF orientation.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def align_ground_truth(noisy, target):
    height, width = noisy.shape[:2]
    if target.shape[:2] != (height, width):
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA)

    def normalized_gray(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return (gray - gray.mean()) / (gray.std() + 1e-6)

    size = (width // 4, height // 4)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        ecc, warp = cv2.findTransformECC(
            cv2.resize(normalized_gray(noisy), size),
            cv2.resize(normalized_gray(target), size),
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN, criteria,
        )
    except cv2.error:
        return target, "failed", None

    warp[:, 2] *= 4.0
    angle = math.degrees(math.atan2(warp[1, 0], warp[0, 0]))
    if abs(warp[0, 2]) > width * 0.15 or abs(warp[1, 2]) > height * 0.15 or abs(angle) > 10:
        return target, "rejected", ecc

    warped = cv2.warpAffine(
        target, warp, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, "applied", ecc


def image_tensor(rgb, device):
    pixels = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)


def image_scores(image, target, lpips_model):
    diff = image[:, :, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8]
    mse = diff.square().mean()
    psnr = (10.0 * torch.log10(1.0 / mse)).item()
    ssim_score = ssim(image * 255.0, target * 255.0, data_range=255.0, size_average=True).item()
    lpips_score = lpips_model(2 * image - 1, 2 * target - 1).item()
    return psnr, ssim_score, lpips_score


@torch.inference_mode()
def score_scene(name, noisy, restored, target, lpips_model, device):
    target, alignment, ecc = align_ground_truth(noisy, target)
    reference = image_tensor(target, device)
    noisy_psnr, noisy_ssim, noisy_lpips = image_scores(image_tensor(noisy, device), reference, lpips_model)
    model_psnr, model_ssim, model_lpips = image_scores(image_tensor(restored, device), reference, lpips_model)
    psnr_gain = model_psnr - noisy_psnr if model_psnr != noisy_psnr else 0.0
    return {
        "scene": name,
        "noisy_psnr_db": round(noisy_psnr, 2), "denoised_psnr_db": round(model_psnr, 2),
        "psnr_gain_db": round(psnr_gain, 2),
        "noisy_ssim": round(noisy_ssim, 4), "denoised_ssim": round(model_ssim, 4),
        "noisy_lpips": round(noisy_lpips, 4), "denoised_lpips": round(model_lpips, 4),
        "alignment_status": alignment, "alignment_ecc": None if ecc is None else round(ecc, 6),
    }


def write_scores(rows, folder):
    columns = ("scene", *METRIC_COLUMNS, "alignment_status", "alignment_ecc")
    # The notebook averages the rounded per-scene values.
    average = {key: round(mean(row[key] for row in rows), 4) for key in METRIC_COLUMNS}
    path = folder / "metrics_by_scene.csv"
    pending = path.with_suffix(".csv.part")
    with pending.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"scene": "Average", **average})
    pending.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT,
                        help="Folder containing the captured scenes (default: %(default)s).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--checkpoint", type=Path, help="Denoise with this NAFNet checkpoint.")
    input_group.add_argument("--reuse-denoised", type=Path,
                             help="Score previously saved images from this run folder.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-label", default="NAFNet output", help="Label for the denoised panel.")
    parser.add_argument("--scene", help="Folder name to test. By default, test every JPEG burst.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not args.scenes_root.is_dir():
        parser.error(f"Scene folder does not exist: {args.scenes_root}")
    scenes = [
        folder for folder in sorted(args.scenes_root.iterdir())
        if (folder / "Burst_JPEG").is_dir() and (args.scene is None or folder.name == args.scene)
    ]
    if not scenes:
        parser.error(f"No matching scene with a Burst_JPEG folder under {args.scenes_root}.")
    crop_file = args.scenes_root / CROP_CATALOG
    crop_boxes = json.loads(crop_file.read_text(encoding="utf-8")) if crop_file.is_file() else {}
    inputs = [(scene / "Burst_JPEG" / "frame_00.JPG", scene / "GT_JPEG" / "low_iso_GT.JPG")
              for scene in scenes]
    for scene, (noisy_path, target_path) in zip(scenes, inputs, strict=True):
        for path in (noisy_path, target_path):
            if not path.is_file():
                parser.error(f"Missing input JPEG: {path}")
        if args.reuse_denoised is not None:
            prediction_path = args.reuse_denoised / scene.name / "denoised.png"
            if not prediction_path.is_file():
                parser.error(f"Missing saved NAFNet output: {prediction_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use the training environment or choose --device cpu.")
    model = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = NAFNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        print(f"Checkpoint epoch {checkpoint['epoch']} on {device}. {len(scenes)} scene(s).", flush=True)
        del checkpoint
    else:
        print(f"Scoring {len(scenes)} scene(s) from {args.reuse_denoised} on {device}.", flush=True)

    torch.hub.set_dir(str(PROJECT_DIR / ".torch_cache"))
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    for scene, (noisy_path, target_path) in zip(scenes, inputs, strict=True):
        noisy = read_rgb(noisy_path)
        scene_output = args.output / scene.name
        scene_output.mkdir(parents=True, exist_ok=True)
        denoised_path = scene_output / "denoised.png"
        if model is not None:
            restored = denoise_jpeg(model, noisy, device)
            save_png(Image.fromarray(restored), denoised_path)
        else:
            source = args.reuse_denoised / scene.name / "denoised.png"
            restored = read_rgb(source)
            if restored.shape != noisy.shape:
                raise ValueError(f"Unexpected saved output dimensions: {source}")
            if source.resolve() != denoised_path.resolve():
                pending = denoised_path.with_suffix(".png.part")
                shutil.copyfile(source, pending)
                pending.replace(denoised_path)
        save_comparison(noisy, restored, scene_output / "full_comparison.png", args.output_label)
        save_crop_comparison(noisy, restored, scene_output / "crop_comparison.png",
                             crop_boxes.get(scene.name), args.output_label)
        row = score_scene(scene.name, noisy, restored, read_rgb(target_path), lpips_model, device)
        rows.append(row)
        print(f"{scene.name}: PSNR {row['noisy_psnr_db']:.2f} -> {row['denoised_psnr_db']:.2f} dB, "
              f"SSIM {row['noisy_ssim']:.4f} -> {row['denoised_ssim']:.4f}, "
              f"LPIPS {row['noisy_lpips']:.4f} -> {row['denoised_lpips']:.4f}; "
              f"alignment {row['alignment_status']} (ECC {row['alignment_ecc']})", flush=True)
    print(f"Saved: {write_scores(rows, args.output)}", flush=True)


if __name__ == "__main__":
    main()
