import csv
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


CLASS_LABELS = [0, 1, 2]
TEXT_COLUMNS = ["text_neutral", "text_negative", "text_positive"]
FULL_COLUMNS = ["full_neutral", "full_negative", "full_positive"]
FACE_COLUMNS = ["face_neutral", "face_negative", "face_positive"]

PREVIOUS_BEST_ACCURACY = 0.5988
PREVIOUS_BEST_MACRO_F1 = 0.5872

TEXT_WEIGHTS = [weight / 100 for weight in range(70, 96)]
FULL_WEIGHTS = [weight / 100 for weight in range(3, 26)]
MIN_FACE_WEIGHT = 0.00
MAX_FACE_WEIGHT = 0.15


def read_predictions(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} predictions: {path}. "
            "Run the matching prediction export or training script first."
        )

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["image_id"]): row for row in reader}


def as_probs(row, columns):
    return [float(row[column]) for column in columns]


def valid_weight_configs():
    for text_weight in TEXT_WEIGHTS:
        for full_weight in FULL_WEIGHTS:
            face_weight = round(1.0 - text_weight - full_weight, 10)
            if MIN_FACE_WEIGHT <= face_weight <= MAX_FACE_WEIGHT:
                yield text_weight, full_weight, face_weight


def no_face_weights(text_weight, full_weight):
    total = text_weight + full_weight
    if total <= 0:
        raise ValueError("Text and full-image weights cannot both be zero.")

    return text_weight / total, full_weight / total, 0.0


def get_weights(face_count, text_weight, full_weight, face_weight):
    if face_count == 0:
        return no_face_weights(text_weight, full_weight)

    return text_weight, full_weight, face_weight


def predict_label(text_probs, full_probs, face_probs, face_count, weights):
    text_weight, full_weight, face_weight = get_weights(face_count, *weights)
    fused_probs = [
        text_weight * text_prob
        + full_weight * full_prob
        + face_weight * face_prob
        for text_prob, full_prob, face_prob in zip(text_probs, full_probs, face_probs)
    ]
    return max(range(len(fused_probs)), key=fused_probs.__getitem__)


def evaluate_config(text_predictions, full_predictions, face_predictions, weights):
    shared_ids = sorted(
        set(text_predictions) & set(full_predictions) & set(face_predictions)
    )
    if not shared_ids:
        raise ValueError("No overlapping image IDs found between prediction files.")

    true_labels = []
    predicted_labels = []

    for image_id in shared_ids:
        text_row = text_predictions[image_id]
        full_row = full_predictions[image_id]
        face_row = face_predictions[image_id]
        true_label = int(text_row["true_label"])
        full_label = int(full_row["true_label"])
        face_label = int(face_row["true_label"])

        if true_label != full_label or true_label != face_label:
            raise ValueError(
                f"Label mismatch for image_id={image_id}: "
                f"text={true_label}, full={full_label}, face={face_label}"
            )

        predicted_label = predict_label(
            text_probs=as_probs(text_row, TEXT_COLUMNS),
            full_probs=as_probs(full_row, FULL_COLUMNS),
            face_probs=as_probs(face_row, FACE_COLUMNS),
            face_count=int(face_row["face_count"]),
            weights=weights,
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

    text_weight, full_weight, face_weight = weights
    no_face_text_weight, no_face_full_weight, _ = no_face_weights(
        text_weight,
        full_weight,
    )

    return {
        "text_weight": text_weight,
        "full_weight": full_weight,
        "face_weight": face_weight,
        "no_face_text_weight": no_face_text_weight,
        "no_face_full_weight": no_face_full_weight,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sample_count": len(shared_ids),
    }


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text_weight",
                "full_weight",
                "face_weight",
                "no_face_text_weight",
                "no_face_full_weight",
                "accuracy",
                "macro_f1",
                "sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def print_result_table(title, results, limit=10):
    print(title)
    print(
        "rank  text/full/face  no-face(text/full)  "
        "accuracy  macro_f1  samples"
    )
    for rank, result in enumerate(results[:limit], start=1):
        print(
            f"{rank:>4}  "
            f"{result['text_weight']:.2f}/"
            f"{result['full_weight']:.2f}/"
            f"{result['face_weight']:.2f}        "
            f"{result['no_face_text_weight']:.2f}/"
            f"{result['no_face_full_weight']:.2f}              "
            f"{result['accuracy']:.4f}    "
            f"{result['macro_f1']:.4f}    "
            f"{result['sample_count']}"
        )


def print_results(results, limit=10):
    macro_sorted_results = sorted(
        results,
        key=lambda result: (result["macro_f1"], result["accuracy"]),
        reverse=True,
    )
    accuracy_sorted_results = sorted(
        results,
        key=lambda result: (result["accuracy"], result["macro_f1"]),
        reverse=True,
    )

    print("Top multimodal fusion weight configurations by Macro F1:")
    print_result_table(
        title="Primary ranking: Macro F1",
        results=macro_sorted_results,
        limit=limit,
    )
    print()
    print_result_table(
        title="Secondary ranking: Accuracy",
        results=accuracy_sorted_results,
        limit=limit,
    )


def tune_text_multimodal_fusion():
    project_root = Path(__file__).resolve().parents[2]
    text_path = project_root / "text_predictions.csv"
    full_path = project_root / "full_image_predictions.csv"
    face_path = project_root / "face_predictions.csv"
    output_path = project_root / "multimodal_fusion_weight_search_results.csv"

    text_predictions = read_predictions(text_path, "text")
    full_predictions = read_predictions(full_path, "full-image")
    face_predictions = read_predictions(face_path, "face")
    results = []

    for weights in valid_weight_configs():
        results.append(
            evaluate_config(
                text_predictions=text_predictions,
                full_predictions=full_predictions,
                face_predictions=face_predictions,
                weights=weights,
            )
        )

    if not results:
        raise ValueError("No valid weight configurations were generated.")

    results.sort(
        key=lambda result: (result["macro_f1"], result["accuracy"]),
        reverse=True,
    )
    save_results(results, output_path)
    print_results(results)

    best = results[0]
    print("\nBest configuration:")
    print(
        "images with faces: "
        f"{best['text_weight']:.2f} text + "
        f"{best['full_weight']:.2f} full + "
        f"{best['face_weight']:.2f} face"
    )
    print(
        "no-face fallback: "
        f"{best['no_face_text_weight']:.2f} text + "
        f"{best['no_face_full_weight']:.2f} full"
    )
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print("\nPrevious best:")
    print(f"Accuracy: {PREVIOUS_BEST_ACCURACY:.4f}")
    print(f"Macro F1: {PREVIOUS_BEST_MACRO_F1:.4f}")
    print("\nDelta vs previous best:")
    print(f"Accuracy: {best['accuracy'] - PREVIOUS_BEST_ACCURACY:+.4f}")
    print(f"Macro F1: {best['macro_f1'] - PREVIOUS_BEST_MACRO_F1:+.4f}")
    print(f"Saved full search results to {output_path}")

    return best


def main():
    try:
        tune_text_multimodal_fusion()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
