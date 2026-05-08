from pathlib import Path
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset


IMAGE_DIRS = {
    "train": "train_images",
    "dev": "dev_images",
    "test": "test_images",
}


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

        # Kept for future dialogue/text-fusion work; full-image training uses image IDs directly.
        self.index_file = ende_dir / f"image_index_{split}.txt"
        self.label_file = ende_dir / f"sentiment_{split}.txt"

        if split not in IMAGE_DIRS:
            raise ValueError(f"Unsupported split: {split}")

        self.image_dir = self.data_root / IMAGE_DIRS[split]
        self.labels = self._load_labels()
        self.image_file_count = self._count_image_files()
        self.samples, self.missing_image_paths = self._build_samples()
        self.sample_labels = [label for _, label in self.samples]
        self.label_distribution = Counter(self.sample_labels)
        self.diagnostics = {
            "split": self.split,
            "label_lines": len(self.labels),
            "image_files": self.image_file_count,
            "usable_samples": len(self.samples),
            "missing_images": len(self.missing_image_paths),
            "label_distribution": dict(sorted(self.label_distribution.items())),
        }

        if not self.samples:
            raise ValueError(
                f"No usable samples found for split '{self.split}'. "
                f"Checked image directory: {self.image_dir}"
            )

    def _load_labels(self):
        labels = []

        with open(self.label_file, "r", encoding="utf-8") as f:
            for line in f:
                labels.append(int(line.strip()))

        return labels

    def _count_image_files(self):
        return sum(1 for _ in self.image_dir.glob("*.jpg"))

    def _build_samples(self):
        samples = []
        missing_image_paths = []

        for image_id, label in enumerate(self.labels):
            img_path = self.image_dir / f"{image_id}.jpg"

            if img_path.exists():
                samples.append((img_path, label))
            else:
                missing_image_paths.append(img_path)

        return samples, missing_image_paths

    def get_diagnostics(self):
        return {
            **self.diagnostics,
            "missing_image_paths": [str(path) for path in self.missing_image_paths],
        }

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image file: {img_path}") from exc

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = self._load_image(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label
