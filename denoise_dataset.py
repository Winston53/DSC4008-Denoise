from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from burst_utils import generate_clean_burst, add_bpn_noise

class SingleFrameDenoiseDataset(Dataset):
    def __init__(self, image_dir, training = True, rng_seed = 4008):
        self.training = training
        self.seed = rng_seed
        self.image_paths = []
        paths = sorted(Path(image_dir).glob("*.jpg"))

        for path in paths:
            with Image.open(path) as img:
                if min(img.size) >= 640:
                    self.image_paths.append(path)

        print(f"Usable photographs: {len(self.image_paths)} / {len(paths)}")

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

        with Image.open(self.image_paths[idx]) as img:
            img = np.array(img.convert("RGB"), dtype = np.float32) / 255.0

        burst, shifts = generate_clean_burst(img, rng)
        sample = add_bpn_noise(burst, rng)

        noisy = torch.from_numpy(sample["noisy_burst"][0]).permute(2, 0, 1).contiguous()
        clean = torch.from_numpy(sample["clean_target"]).permute(2, 0, 1).contiguous()

        return {
            "noisy": noisy,
            "clean": clean,
            "white_level": torch.tensor(sample["white_level"], dtype = torch.float32),
        }

if __name__ == "__main__":
    img_dir = (Path(__file__).resolve().parent/ "data" / "open-images-v6" / "train" / "data")
    dataset = SingleFrameDenoiseDataset(img_dir)
    sample = dataset[0]

    print("Input:", sample["noisy"].shape, sample["noisy"].dtype)
    print("Target:", sample["clean"].shape, sample["clean"].dtype)
    print("White level:", sample["white_level"].item())
