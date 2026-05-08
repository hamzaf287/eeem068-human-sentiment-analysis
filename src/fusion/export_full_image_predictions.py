import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MSCTDDataset
from src.data.transforms import get_eval_transforms
from src.models.full_image_model import get_full_image_model


SPLIT = "test"
BATCH_SIZE = 32
CLASS_PROB_COLUMNS = ["full_neutral", "full_negative", "full_positive"]


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
            f"Missing full-image checkpoint: {checkpoint_path}. "
            "Run python3 -m src.training.train_full_image first."
        )

    model = get_full_image_model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_predictions():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    checkpoint_path = project_root / "best_full_image_model.pth"
    output_path = project_root / "full_image_predictions.csv"
    device = get_device()

    dataset = MSCTDDataset(
        data_root=data_root,
        split=SPLIT,
        transform=get_eval_transforms(),
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
                image_path, _ = sample
                row = {
                    "image_id": int(image_path.stem),
                    "true_label": int(true_label),
                }
                row.update(dict(zip(CLASS_PROB_COLUMNS, class_probs)))
                rows.append(row)

    rows.sort(key=lambda row: row["image_id"])

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "true_label", *CLASS_PROB_COLUMNS],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} full-image predictions to {output_path}")


def main():
    try:
        export_predictions()
    except FileNotFoundError as exc:
        print(exc)


if __name__ == "__main__":
    main()
