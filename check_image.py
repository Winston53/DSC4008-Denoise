from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "open-images-v6" / "train" / "data"

image_path = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_path:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

selected_path = None

for path in image_path:
    with Image.open(path) as img:
        width, height = img.size
    print(f"{path.name}: {width}x{height}")

    if selected_path is None and min(width, height) >= 640:
        selected_path = path

if selected_path is None:
    raise ValueError("No image found with both width and height >= 640 pixels")

with Image.open(selected_path) as img:
    image = img.convert("RGB")

print(f"Selected image: {selected_path.name} with size {image.size}")

plt.imshow(image)
plt.title(f"{selected_path.name} — {image.width} × {image.height}")
plt.axis("off")
plt.show()