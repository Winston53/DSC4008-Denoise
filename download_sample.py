from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def main():
    fo.config.dataset_zoo_dir = str(DATA_DIR)

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split = "train",
        dataset_name = "oi_v6_2000",
        label_types = [],
        max_samples = 2000,
        shuffle = True,
        seed = 4008,
        num_workers = 4
    )
    print(len(dataset))
    print(dataset.first().filepath)

if __name__ == "__main__":
    main()
