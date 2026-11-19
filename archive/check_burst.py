"""
Seeded visual check for synthetic noise.
Prints burst and noise-map shapes and displays the clean and noisy reference frames.
Useful when changing the noise model.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from burst_utils import (
    generate_clean_burst,
    add_bpn_noise,
    display_rgb,
)

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "open-images-v6" / "train" / "data"

rng = np.random.default_rng(4008)

# Find the first image with both width and height >= 640 pixels
for path in sorted(IMAGE_DIR.glob("*.jpg")):
    with Image.open(path) as img:
        if min(img.size) >= 640:
            image = np.array(img.convert("RGB"), dtype = np.float32) / 255.0
            break
else:
    raise ValueError("No image found with both width and height >= 640 pixels")


burst, shifts = generate_clean_burst(image, rng)
sample = add_bpn_noise(burst, rng)

noisy_burst = sample["noisy_burst"]
clean_target = sample["clean_target"]
noise_maps = sample["noise_maps"]
white_level = sample["white_level"]
sigma_read = sample["sigma_read"]
sigma_shot = sample["sigma_shot"]

print("Clean RGB burst:", burst.shape)
print("Noisy linear burst:", noisy_burst.shape)
print("Clean target:", clean_target.shape)
print("Noise maps:", noise_maps.shape)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(display_rgb(clean_target, white_level))
axes[0].set_title("Clean reference")

axes[1].imshow(display_rgb(noisy_burst[0], white_level))
axes[1].set_title("Noisy reference — frame 0")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


