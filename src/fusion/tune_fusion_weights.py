import csv
from itertools import product
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


CLASS_LABELS = [0, 1, 2]
FULL_COLUMNS = ["full_neutral", "full_negative", "full_positive"]
FACE_COLUMNS = ["face_neutral", "face_negative", "face_positive"]
SINGLE_FACE_FULL_WEIGHTS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MULTI_FACE_FULL_WEIGHTS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def read_predictions(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} predictions: {path}. "
            "Run the prediction export scripts first."
        )

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["image_id"]): row for row in reader}


def as_probs(row, columns):
    return [float(row[column]) for column in columns]


def get_weights(face_count, single_face_full_weight, multi_face_full_weight):
    if face_count == 0:
        return 1.0, 0.0

    if face_count == 1:
        return single_face_full_weight, 1.0 - single_face_full_weight

    return multi_face_full_weight, 1.0 - multi_face_full_weight


def predict_label(full_probs, face_probs, face_count, single_face_full_weight, multi_face_full_weight):
    full_weight, face_weight = get_weights(
        face_count,
        single_face_full_weight,
        multi_face_full_weight,
    )
    fused_probs = [
        full_weight * full_prob + face_weight * face_prob
        for full_prob, face_prob in zip(full_probs, face_probs)
    ]
    return max(range(len(fused_probs)), key=fused_probs.__getitem__)


def evaluate_config(full_predictions, face_predictions, single_face_full_weight, multi_face_full_weight):
    shared_ids = sorted(set(full_predictions) & set(face_predictions))
    if not shared_ids:
        raise ValueError("No overlapping image IDs found between prediction files.")

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

        predicted_label = predict_label(
            full_probs=as_probs(full_row, FULL_COLUMNS),
            face_probs=as_probs(face_row, FACE_COLUMNS),
            face_count=int(face_row["face_count"]),
            single_face_full_weight=single_face_full_weight,
            multi_face_full_weight=multi_face_full_weight,
        )

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

    return {
        "single_face_full_weight": single_face_full_weight,
        "single_face_face_weight": 1.0 - single_face_full_weight,
        "multi_face_full_weight": multi_face_full_weight,
        "multi_face_face_weight": 1.0 - multi_face_full_weight,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sample_count": len(shared_ids),
    }


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "single_face_full_weight",
                "single_face_face_weight",
                "multi_face_full_weight",
                "multi_face_face_weight",
                "accuracy",
                "macro_f1",
                "sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def print_results(results, limit=20):
    print("Top fusion weight configurations by Macro F1:")
    print(
        "rank  single(full/face)  multi(full/face)   "
        "accuracy  macro_f1  samples"
    )
    for rank, result in enumerate(results[:limit], start=1):
        print(
            f"{rank:>4}  "
            f"{result['single_face_full_weight']:.2f}/"
            f"{result['single_face_face_weight']:.2f}          "
            f"{result['multi_face_full_weight']:.2f}/"
            f"{result['multi_face_face_weight']:.2f}          "
            f"{result['accuracy']:.4f}    "
            f"{result['macro_f1']:.4f}    "
            f"{result['sample_count']}"
        )


def tune_fusion_weights():
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / "full_image_predictions.csv"
    face_path = project_root / "face_predictions.csv"
    output_path = project_root / "fusion_weight_search_results.csv"

    full_predictions = read_predictions(full_path, "full-image")
    face_predictions = read_predictions(face_path, "face")
    results = []

    for single_face_full_weight, multi_face_full_weight in product(
        SINGLE_FACE_FULL_WEIGHTS,
        MULTI_FACE_FULL_WEIGHTS,
    ):
        results.append(evaluate_config(
            full_predictions=full_predictions,
            face_predictions=face_predictions,
            single_face_full_weight=single_face_full_weight,
            multi_face_full_weight=multi_face_full_weight,
        ))

    results.sort(key=lambda result: result["macro_f1"], reverse=True)
    save_results(results, output_path)
    print_results(results)

    best = results[0]
    print("\nBest configuration:")
    print(
        "single face: "
        f"{best['single_face_full_weight']:.2f} full + "
        f"{best['single_face_face_weight']:.2f} face"
    )
    print(
        "multiple faces: "
        f"{best['multi_face_full_weight']:.2f} full + "
        f"{best['multi_face_face_weight']:.2f} face"
    )
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Saved full search results to {output_path}")

    return best


def main():
    try:
        tune_fusion_weights()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
