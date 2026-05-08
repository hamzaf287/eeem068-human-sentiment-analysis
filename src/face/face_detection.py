import csv
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image


SPLIT_CONFIG = {
    "train": ("train_images", "face_train_images"),
    "dev": ("dev_images", "face_dev_images"),
    "test": ("test_images", "face_test_images"),
}


class FaceExtractor:
    def __init__(self, device="cpu"):
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def process_image(self, image_path, output_path, target_size=(224, 224)):
        image_path = Path(image_path)
        output_path = Path(output_path)

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image file: {image_path}") from exc

        boxes, _ = self.mtcnn.detect(img)
        face_count = 0 if boxes is None else len(boxes)

        if face_count > 0:
            largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            x1, y1, x2, y2 = [int(coord) for coord in largest_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            crop = img.crop((x1, y1, x2, y2))
            used_strategy = "largest_face"
        else:
            crop = self._center_crop(img, target_size)
            used_strategy = "center_crop_no_face"

        crop = crop.resize(target_size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)

        return {
            "image_id": int(image_path.stem),
            "original_image_path": str(image_path),
            "face_image_path": str(output_path),
            "face_count": face_count,
            "had_face": face_count > 0,
            "used_strategy": used_strategy,
        }

    @staticmethod
    def _center_crop(img, target_size):
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = left + crop_size
        bottom = top + crop_size
        return img.crop((left, top, right, bottom))


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_split(data_root, split, extractor):
    input_dir_name, output_dir_name = SPLIT_CONFIG[split]
    input_dir = data_root / input_dir_name
    output_dir = data_root / output_dir_name
    metadata_path = data_root / f"face_metadata_{split}.csv"

    image_paths = sorted(input_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {input_dir}")

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_dir / image_path.name
        rows.append(extractor.process_image(image_path, output_path))

        if index % 100 == 0 or index == len(image_paths):
            print(f"{split}: processed {index}/{len(image_paths)} images")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "original_image_path",
                "face_image_path",
                "face_count",
                "had_face",
                "used_strategy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    face_count = sum(row["face_count"] for row in rows)
    no_face_count = sum(1 for row in rows if not row["had_face"])
    print(
        f"{split}: saved {len(rows)} face crops to {output_dir} "
        f"({face_count} detected faces, {no_face_count} center crops)"
    )
    print(f"{split}: saved metadata to {metadata_path}")


def main():
    data_root = Path("data/raw")
    device = get_device()
    print("Using device:", device)

    extractor = FaceExtractor(device=device)

    for split in SPLIT_CONFIG:
        extract_split(data_root, split, extractor)


if __name__ == "__main__":
    main()
