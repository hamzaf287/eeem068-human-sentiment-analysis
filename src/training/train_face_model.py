from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.face.face_dataset import FaceSentimentDataset
from src.models.face_model import FaceSentimentModel


LABEL_TO_NAME = {
    0: "neutral",
    1: "negative",
    2: "positive",
}
CLASS_LABELS = sorted(LABEL_TO_NAME)
CLASS_NAMES = [LABEL_TO_NAME[label] for label in CLASS_LABELS]

BATCH_SIZE = 32
NUM_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 0.0001


def get_device():
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_dataloader(data_root, split, batch_size=BATCH_SIZE, shuffle=False):
    dataset = FaceSentimentDataset(data_root=data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def print_dataset_summary(name, dataset):
    diagnostics = dataset.get_diagnostics()
    distribution = {
        LABEL_TO_NAME.get(label, str(label)): count
        for label, count in diagnostics["label_distribution"].items()
    }

    print(f"{name} samples: {diagnostics['usable_samples']}")
    print(f"{name} missing face crops: {diagnostics['missing_face_images']}")
    print(f"{name} class distribution: {distribution}")
    print(f"{name} crop strategy distribution: {diagnostics['strategy_distribution']}")


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )

    return avg_loss, accuracy, macro_f1, all_labels, all_preds


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
        title="Face Model Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="macro F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_dataloaders(data_root):
    train_loader = get_dataloader(data_root, "train", shuffle=True)
    dev_loader = get_dataloader(data_root, "dev", shuffle=False)
    test_loader = get_dataloader(data_root, "test", shuffle=False)
    return train_loader, dev_loader, test_loader


def print_missing_crops_message(error):
    print(error)
    print("Generate face crops first from the repository root:")
    print("  python3 -m src.face.face_detection  # macOS/Linux")
    print("  python -m src.face.face_detection   # Windows")


def train():
    device = get_device()
    print("Using device:", device)

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    checkpoint_path = project_root / "best_face_model.pth"
    confusion_matrix_path = project_root / "face_confusion_matrix.png"
    report_path = project_root / "face_classification_report.txt"
    history_path = project_root / "face_training_history.png"

    try:
        train_loader, dev_loader, test_loader = build_dataloaders(data_root)
    except FileNotFoundError as exc:
        print_missing_crops_message(exc)
        return

    print_dataset_summary("Train", train_loader.dataset)
    print_dataset_summary("Dev", dev_loader.dataset)
    print_dataset_summary("Test", test_loader.dataset)

    model = FaceSentimentModel(num_classes=len(CLASS_NAMES)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_macro_f1 = -1.0
    no_improve_epochs = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    print("Class counts:", Counter(train_loader.dataset.sample_labels))

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved best face model to {checkpoint_path.name} "
                f"(val_macro_f1={val_macro_f1:.4f}, val_acc={val_acc:.4f})"
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= PATIENCE:
            print("Early stopping triggered")
            break

    print("\nTraining complete!")
    print(f"Best Validation Macro F1: {best_val_macro_f1:.4f}")
    save_training_history(history, history_path)
    print(f"Saved training history to {history_path.name}")

    best_model = FaceSentimentModel(num_classes=len(CLASS_NAMES)).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, test_macro_f1, test_labels, test_preds = evaluate(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nBest face model test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds, labels=CLASS_LABELS)
    print("\nConfusion Matrix:")
    print(cm)
    save_confusion_matrix(cm, CLASS_NAMES, confusion_matrix_path)
    print(f"Saved confusion matrix to {confusion_matrix_path.name}")

    report = classification_report(
        test_labels,
        test_preds,
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


if __name__ == "__main__":
    train()
