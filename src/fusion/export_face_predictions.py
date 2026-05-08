import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.face.face_dataset import FaceSentimentDataset
from src.models.face_model import FaceSentimentModel


SPLIT = "test"
BATCH_SIZE = 32
CLASS_PROB_COLUMNS = ["face_neutral", "face_negative", "face_positive"]


def get_device():
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing face checkpoint: {checkpoint_path}. "
            "Run python3 -m src.training.train_face_model first."
        )

    model = FaceSentimentModel(num_classes=3).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_predictions():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    checkpoint_path = project_root / "best_face_model.pth"
    output_path = project_root / "face_predictions.csv"
    metadata_path = data_root / f"face_metadata_{SPLIT}.csv"
    device = get_device()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing face metadata: {metadata_path}. "
            "Run python3 -m src.face.face_detection first."
        )

    dataset = FaceSentimentDataset(
        data_root=data_root,
        split=SPLIT,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    model = load_model(checkpoint_path, device)

    rows = []
    start_index = 0

    with torch.no_grad():
        for images, labels in data_loader:
            batch_size = images.size(0)
            batch_samples = dataset.samples[start_index:start_index + batch_size]
            start_index += batch_size

            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu().tolist()

            for sample, true_label, class_probs in zip(batch_samples, labels.tolist(), probs):
                row = {
                    "image_id": int(sample["image_id"]),
                    "true_label": int(true_label),
                    "face_count": int(sample["face_count"]),
                    "had_face": bool(sample["had_face"]),
                }
                row.update(dict(zip(CLASS_PROB_COLUMNS, class_probs)))
                rows.append(row)

    rows.sort(key=lambda row: row["image_id"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "true_label",
                *CLASS_PROB_COLUMNS,
                "face_count",
                "had_face",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} face predictions to {output_path}")


def main():
    try:
        export_predictions()
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()
