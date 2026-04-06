from pathlib import Path
import ast

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FullImageDataset(Dataset):
    def __init__(self, split="train"):
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "raw"
        ende_dir = data_dir / "ende"
        image_dir = data_dir / f"{split}_images"

        self.index_file = ende_dir / f"image_index_{split}.txt"
        self.label_file = ende_dir / f"sentiment_{split}.txt"
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        with open(self.index_file, "r", encoding="utf-8") as f:
            index_lines = f.readlines()

        with open(self.label_file, "r", encoding="utf-8") as f:
            label_lines = f.readlines()

        for idx_line, lbl_line in zip(index_lines, label_lines):
            image_indices = ast.literal_eval(idx_line.strip())
            label = int(lbl_line.strip())

            if len(image_indices) == 0:
                continue

            first_image_id = image_indices[0]
            image_path = self.image_dir / f"{first_image_id}.jpg"

            if image_path.exists():
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label