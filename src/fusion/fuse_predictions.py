import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


LABEL_TO_NAME = {
    0: "neutral",
    1: "negative",
    2: "positive",
}
CLASS_LABELS = sorted(LABEL_TO_NAME)
CLASS_NAMES = [LABEL_TO_NAME[label] for label in CLASS_LABELS]
FULL_COLUMNS = ["full_neutral", "full_negative", "full_positive"]
FACE_COLUMNS = ["face_neutral", "face_negative", "face_positive"]
FUSED_COLUMNS = ["fusion_neutral", "fusion_negative", "fusion_positive"]


def read_predictions(path, key_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {key_name} predictions: {path}. "
            "Run the prediction export scripts first."
        )

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["image_id"]): row for row in reader}


def get_weights(face_count):
    if face_count == 0:
        return 1.0, 0.0

    if face_count == 1:
        return 0.75, 0.25

    return 0.70, 0.30


def as_probs(row, columns):
    return [float(row[column]) for column in columns]


def fuse_probabilities(full_probs, face_probs, face_count):
    full_weight, face_weight = get_weights(face_count)
    return [
        full_weight * full_prob + face_weight * face_prob
        for full_prob, face_prob in zip(full_probs, face_probs)
    ]


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Fusion Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def fuse_predictions():
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / "full_image_predictions.csv"
    face_path = project_root / "face_predictions.csv"
    output_path = project_root / "fusion_predictions.csv"
    confusion_matrix_path = project_root / "fusion_confusion_matrix.png"
    report_path = project_root / "fusion_classification_report.txt"

    full_predictions = read_predictions(full_path, "full-image")
    face_predictions = read_predictions(face_path, "face")
    shared_ids = sorted(set(full_predictions) & set(face_predictions))

    if not shared_ids:
        raise ValueError(
            "No overlapping image IDs found between full-image and face predictions."
        )

    rows = []
    true_labels = []
    predicted_labels = []

    for image_id in shared_ids:
        full_row = full_predictions[image_id]
        face_row = face_predictions[image_id]
        true_label = int(full_row["true_label"])
        face_true_label = int(face_row["true_label"])

        if true_label != face_true_label:
            raise ValueError(
                f"Label mismatch for image_id={image_id}: "
                f"full={true_label}, face={face_true_label}"
            )

        face_count = int(face_row["face_count"])
        full_probs = as_probs(full_row, FULL_COLUMNS)
        face_probs = as_probs(face_row, FACE_COLUMNS)
        fused_probs = fuse_probabilities(full_probs, face_probs, face_count)
        predicted_label = max(range(len(fused_probs)), key=fused_probs.__getitem__)

        row = {
            "image_id": image_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "face_count": face_count,
            "had_face": face_row["had_face"],
        }
        row.update(dict(zip(FUSED_COLUMNS, fused_probs)))
        rows.append(row)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "true_label",
                "predicted_label",
                "face_count",
                "had_face",
                *FUSED_COLUMNS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    cm = confusion_matrix(true_labels, predicted_labels, labels=CLASS_LABELS)
    save_confusion_matrix(cm, CLASS_NAMES, confusion_matrix_path)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Fused {len(rows)} predictions")
    print(f"Fusion Accuracy: {accuracy:.4f}")
    print(f"Fusion Macro F1: {macro_f1:.4f}")
    print(f"Saved fusion predictions to {output_path}")
    print(f"Saved confusion matrix to {confusion_matrix_path}")
    print(f"Saved classification report to {report_path}")


def main():
    try:
        fuse_predictions()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
