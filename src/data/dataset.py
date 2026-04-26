from pathlib import Path
import ast

from PIL import Image
from torch.utils.data import Dataset


class MSCTDDataset(Dataset):

    def __init__(
        self,
        data_root,
        split="train",
        transform=None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        ende_dir = self.data_root / "ende"

        self.index_file = ende_dir / f"image_index_{split}.txt"
        self.label_file = ende_dir / f"sentiment_{split}.txt"

        # image folders
        image_dirs = {
            "train": self.data_root / "train_images",
            "dev": self.data_root / "dev_images",
            "test": self.data_root / "test_images",
        }

        if split not in image_dirs:
            raise ValueError(f"Unsupported split: {split}")

        self.image_dir = image_dirs[split]

        self.image_indices = self._load_indices()
        self.labels = self._load_labels()

        min_len = min(len(self.labels), len(self.image_indices))

        self.labels = self.labels[:min_len]
        self.image_indices = self.image_indices[:min_len]

        

    def _load_indices(self):
        indices = []

        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                try:
                    parsed = ast.literal_eval(line)
                except Exception:
                    parsed = []

                indices.append(parsed)

        return indices

    def _load_labels(self):
        labels = []

        with open(self.label_file, "r", encoding="utf-8") as f:
            for line in f:
                labels.append(int(line.strip()))

        return labels

    def __len__(self):
        return min(len(self.labels), len(self.image_indices))

    def _load_image(self, index_list):

        if not index_list:
            return Image.new("RGB", (224, 224))

        image_id = index_list[0]
        img_path = self.image_dir / f"{image_id}.jpg"

        if not img_path.exists():
            return Image.new("RGB", (224, 224))

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        return img

    def __getitem__(self, idx):

        label = self.labels[idx]
        index_list = self.image_indices[idx]

        img = self._load_image(index_list)

        if self.transform:
            img = self.transform(img)

        return img, label
