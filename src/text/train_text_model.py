import csv
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_TO_NAME = {
    0: "neutral",
    1: "negative",
    2: "positive",
}
CLASS_LABELS = sorted(LABEL_TO_NAME)
CLASS_NAMES = [LABEL_TO_NAME[label] for label in CLASS_LABELS]
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
EMBEDDING_BATCH_SIZE = 64
CLASSIFIER_C_VALUES = [0.1, 0.3, 1.0, 3.0]


class DependencyError(RuntimeError):
    pass


class TransformerLoadError(RuntimeError):
    pass


def get_device():
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_split(data_root, split):
    text_file = data_root / "ende" / f"english_{split}.txt"
    label_file = data_root / "ende" / f"sentiment_{split}.txt"

    texts = text_file.read_text(encoding="utf-8").splitlines()
    labels = [
        int(line.strip())
        for line in label_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(texts) != len(labels):
        raise ValueError(
            f"Text/label length mismatch for {split}: "
            f"{len(texts)} texts vs {len(labels)} labels."
        )

    invalid_labels = sorted(set(labels) - set(CLASS_LABELS))
    if invalid_labels:
        raise ValueError(f"Invalid labels in {label_file}: {invalid_labels}")

    image_ids = np.arange(len(labels), dtype=np.int64)
    return image_ids, texts, np.array(labels, dtype=np.int64)


def load_transformer(device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise DependencyError(
            "Missing dependency: transformers. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except OSError as exc:
        raise TransformerLoadError(
            f"Could not load {MODEL_NAME}. Check internet access or make sure the "
            "model is already cached locally."
        ) from exc

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model


def load_cached_embeddings(cache_path, expected_count):
    if not cache_path.exists():
        return None

    cached = np.load(cache_path)
    embeddings = cached["embeddings"]
    labels = cached["labels"]
    image_ids = cached["image_ids"]

    if len(embeddings) != expected_count:
        return None

    return image_ids, embeddings, labels


def save_cached_embeddings(cache_path, image_ids, embeddings, labels):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image_ids=image_ids,
        embeddings=embeddings,
        labels=labels,
    )


def extract_embeddings_for_split(
    split,
    image_ids,
    texts,
    labels,
    tokenizer,
    transformer,
    device,
    cache_dir,
):
    cache_path = cache_dir / f"text_embeddings_{split}.npz"
    cached = load_cached_embeddings(cache_path, expected_count=len(texts))
    if cached is not None:
        cached_ids, embeddings, cached_labels = cached
        print(f"{split}: loaded cached embeddings from {cache_path}")
        return cached_ids, embeddings, cached_labels

    embeddings = []
    print(f"{split}: extracting DistilBERT embeddings for {len(texts)} samples")

    with torch.no_grad():
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[start:start + EMBEDDING_BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = transformer(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

            processed = min(start + EMBEDDING_BATCH_SIZE, len(texts))
            if processed % 1024 == 0 or processed == len(texts):
                print(f"{split}: embedded {processed}/{len(texts)}")

    embeddings = np.vstack(embeddings)
    save_cached_embeddings(cache_path, image_ids, embeddings, labels)
    print(f"{split}: saved embeddings to {cache_path}")

    return image_ids, embeddings, labels


def get_class_probabilities(classifier, embeddings):
    raw_probs = classifier.predict_proba(embeddings)
    probs = np.zeros((len(embeddings), len(CLASS_LABELS)), dtype=np.float64)

    for source_index, label in enumerate(classifier.classes_):
        target_index = CLASS_LABELS.index(int(label))
        probs[:, target_index] = raw_probs[:, source_index]

    return probs


def evaluate_classifier(classifier, embeddings, labels):
    predictions = classifier.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(
        labels,
        predictions,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_f1, predictions


def train_classifier(train_embeddings, train_labels, dev_embeddings, dev_labels):
    best_classifier = None
    best_dev_macro_f1 = -1.0
    best_c = None

    for c_value in CLASSIFIER_C_VALUES:
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])
        classifier.fit(train_embeddings, train_labels)
        _, dev_macro_f1, _ = evaluate_classifier(
            classifier,
            dev_embeddings,
            dev_labels,
        )
        print(f"Dev Macro F1 with C={c_value}: {dev_macro_f1:.4f}")

        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_classifier = classifier
            best_c = c_value

    print(f"Selected LogisticRegression C={best_c} (dev_macro_f1={best_dev_macro_f1:.4f})")
    return best_classifier


def save_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Text Model Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_predictions(image_ids, labels, probs, save_path):
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "true_label",
                "text_neutral",
                "text_negative",
                "text_positive",
            ],
        )
        writer.writeheader()

        for image_id, label, class_probs in zip(image_ids, labels, probs):
            writer.writerow({
                "image_id": int(image_id),
                "true_label": int(label),
                "text_neutral": class_probs[0],
                "text_negative": class_probs[1],
                "text_positive": class_probs[2],
            })


def train_text_model():
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    cache_dir = project_root / "outputs" / "text_embeddings"
    model_path = project_root / "best_text_model.joblib"
    predictions_path = project_root / "text_predictions.csv"
    confusion_matrix_path = project_root / "text_confusion_matrix.png"
    report_path = project_root / "text_classification_report.txt"
    device = get_device()

    print("Using device:", device)
    print(f"Transformer backbone: {MODEL_NAME} (frozen)")

    train_ids, train_texts, train_labels = load_split(data_root, "train")
    dev_ids, dev_texts, dev_labels = load_split(data_root, "dev")
    test_ids, test_texts, test_labels = load_split(data_root, "test")
    print(f"Train samples: {len(train_texts)}")
    print(f"Dev samples: {len(dev_texts)}")
    print(f"Test samples: {len(test_texts)}")

    tokenizer, transformer = load_transformer(device)

    _, train_embeddings, train_labels = extract_embeddings_for_split(
        "train",
        train_ids,
        train_texts,
        train_labels,
        tokenizer,
        transformer,
        device,
        cache_dir,
    )
    _, dev_embeddings, dev_labels = extract_embeddings_for_split(
        "dev",
        dev_ids,
        dev_texts,
        dev_labels,
        tokenizer,
        transformer,
        device,
        cache_dir,
    )
    test_ids, test_embeddings, test_labels = extract_embeddings_for_split(
        "test",
        test_ids,
        test_texts,
        test_labels,
        tokenizer,
        transformer,
        device,
        cache_dir,
    )

    classifier = train_classifier(
        train_embeddings,
        train_labels,
        dev_embeddings,
        dev_labels,
    )
    joblib.dump(classifier, model_path)
    print(f"Saved text classifier to {model_path}")

    test_accuracy, test_macro_f1, test_predictions = evaluate_classifier(
        classifier,
        test_embeddings,
        test_labels,
    )
    test_probs = get_class_probabilities(classifier, test_embeddings)

    print("\nText model test results:")
    print(f"Test Acc: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    cm = confusion_matrix(test_labels, test_predictions, labels=CLASS_LABELS)
    print("\nConfusion Matrix:")
    print(cm)
    save_confusion_matrix(cm, confusion_matrix_path)
    print(f"Saved confusion matrix to {confusion_matrix_path.name}")

    report = classification_report(
        test_labels,
        test_predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {report_path.name}")

    save_predictions(test_ids, test_labels, test_probs, predictions_path)
    print(f"Saved text predictions to {predictions_path.name}")


def main():
    try:
        train_text_model()
    except DependencyError as exc:
        print(exc)
    except TransformerLoadError as exc:
        print(exc)


if __name__ == "__main__":
    main()
