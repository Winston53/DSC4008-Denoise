import numpy as np

def generate_clean_burst(image, rng):
    height, width = image.shape[:2]
    top = rng.integers(0, height - 640 + 1)
    left = rng.integers(0, width - 640 + 1)
    region = image[top: top + 640, left:left + 640]

    frames = []
    shifts = []

    # Generate a burst of 8 frames with random shifts
    large_motion_probability = min(rng.poisson(1.5) / 8, 1.0)

    for i in range(8):
        if i == 0:
            dy, dx = 0, 0
        else:
            limit = 16 if rng.random() < large_motion_probability else 2
            dy, dx = rng.integers(-limit * 4, limit * 4 + 1, size = 2)

        y, x = 64 + dy, 64 + dx
        crop = region[y: y + 512, x: x + 512]

        frame = crop.reshape(128, 4, 128, 4, 3).mean(axis = (1, 3))
        frames.append(frame)
        shifts.append((dy / 4, dx / 4))

    burst = np.stack(frames)

    return burst, shifts

def add_bpn_noise(burst, rng):
    white_level = float(10 ** rng.uniform(-1.0, 0.0))
    clean_burst = (white_level * burst ** 2.2).astype(np.float32)
    clean_target = clean_burst[0].copy()

    sigma_read = float(10 ** rng.uniform(-3.0, -1.5))
    sigma_shot = float(10 ** rng.uniform(-2.0, -1.0))
    
    read_noise = sigma_read * rng.standard_normal(clean_burst.shape)
    shot_noise = (np.sqrt(clean_burst) * sigma_shot * rng.standard_normal(clean_burst.shape))
    noisy_burst = (clean_burst + read_noise + shot_noise).astype(np.float32)
    noise_maps = np.sqrt(sigma_read ** 2 + sigma_shot ** 2 * np.maximum(noisy_burst, 0.0)).astype(np.float32)

    return {
        "noisy_burst": noisy_burst,
        "clean_target": clean_target,
        "noise_maps": noise_maps,
        "white_level": white_level,
        "sigma_read": sigma_read,
        "sigma_shot": sigma_shot
    }

def display_rgb(linear_image, white_level):
    x = linear_image / white_level

    b = 0.0031308
    gamma = 1.0 / 2.4
    a = 1.0 / (1.0 / (b ** gamma * (1.0 - gamma)) - 1.0)
    k = (1.0 + a) * gamma * b ** (gamma - 1.0)

    rgb = np.where(x < b, k * x, (1.0 + a) * np.maximum(x, b) ** gamma - a)
    rgb = np.where(x > 1.0, (1.0 + a) * gamma * (x - 1.0) + 1.0, rgb)

    return np.clip(rgb, 0.0, 1.0)