import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


FACE_DIRS = {
    "train": "face_train_images",
    "dev": "face_dev_images",
    "test": "face_test_images",
}


class FaceSentimentDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        if split not in FACE_DIRS:
            raise ValueError(f"Unsupported split: {split}")

        self.data_root = Path(data_root)
        self.split = split
        self.image_dir = self.data_root / FACE_DIRS[split]
        self.metadata_file = self.data_root / f"face_metadata_{split}.csv"
        self.label_file = self.data_root / "ende" / f"sentiment_{split}.txt"
        self.labels = self._load_labels()
        self.transform = transform or self._default_transform()
        self.samples, self.missing_face_paths = self._build_samples()
        self.sample_labels = [sample["label"] for sample in self.samples]
        self.label_distribution = Counter(self.sample_labels)
        self.strategy_distribution = Counter(
            sample["used_strategy"] for sample in self.samples
        )
        self.face_count_distribution = Counter(
            sample["face_count"] for sample in self.samples
        )
        self.diagnostics = {
            "split": self.split,
            "label_lines": len(self.labels),
            "usable_samples": len(self.samples),
            "missing_face_images": len(self.missing_face_paths),
            "label_distribution": dict(sorted(self.label_distribution.items())),
            "strategy_distribution": dict(sorted(self.strategy_distribution.items())),
            "face_count_distribution": dict(sorted(self.face_count_distribution.items())),
        }

        if not self.samples:
            raise FileNotFoundError(
                f"No face crops found for split '{self.split}' in {self.image_dir}. "
                "Generate them first with python3 -m src.face.face_detection "
                "on macOS/Linux or python -m src.face.face_detection on Windows."
            )

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_labels(self):
        with open(self.label_file, "r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip()]

    def _build_samples(self):
        if self.metadata_file.exists():
            return self._build_samples_from_metadata()

        return self._build_samples_from_image_dir()

    def _build_samples_from_metadata(self):
        samples = []
        missing_face_paths = []

        with open(self.metadata_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = int(row["image_id"])
                face_path = Path(row["face_image_path"])

                if not face_path.exists():
                    missing_face_paths.append(face_path)
                    continue

                samples.append(self._make_sample(
                    image_id=image_id,
                    face_path=face_path,
                    face_count=int(row["face_count"]),
                    had_face=row["had_face"] == "True",
                    used_strategy=row["used_strategy"],
                ))

        return samples, missing_face_paths

    def _build_samples_from_image_dir(self):
        if not self.image_dir.exists():
            return [], []

        samples = []
        for face_path in sorted(self.image_dir.glob("*.jpg"), key=lambda path: int(path.stem)):
            image_id = int(face_path.stem)
            samples.append(self._make_sample(
                image_id=image_id,
                face_path=face_path,
                face_count=0,
                had_face=False,
                used_strategy="metadata_missing",
            ))

        return samples, []

    def _make_sample(self, image_id, face_path, face_count, had_face, used_strategy):
        if image_id < 0 or image_id >= len(self.labels):
            raise ValueError(
                f"Face crop image id {image_id} is outside label range for split "
                f"'{self.split}' ({len(self.labels)} labels)."
            )

        return {
            "image_id": image_id,
            "face_path": face_path,
            "label": self.labels[image_id],
            "face_count": face_count,
            "had_face": had_face,
            "used_strategy": used_strategy,
        }

    def get_diagnostics(self):
        return {
            **self.diagnostics,
            "missing_face_paths": [str(path) for path in self.missing_face_paths],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        face_path = sample["face_path"]

        try:
            with Image.open(face_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open face crop: {face_path}") from exc

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]
