from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data.dataloaders import get_dataloaders
from src.models.full_image_model import get_full_image_model


LABEL_TO_NAME = {
    0: "neutral",
    1: "negative",
    2: "positive",
}
CLASS_LABELS = sorted(LABEL_TO_NAME)
CLASS_NAMES = [LABEL_TO_NAME[i] for i in CLASS_LABELS]
POSITIVE_LABEL = 2

# Quick daytime run defaults. For overnight/full training, use:
# BATCH_SIZE = 32 if stable, NUM_EPOCHS = 20, PATIENCE = 3.
BATCH_SIZE = 32
NUM_EPOCHS = 5
PATIENCE = 2
UNFREEZE_LAYER4 = False
POSITIVE_WEIGHT_BOOST = 1.0


def get_device():
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1, all_labels, all_preds


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
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
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


def configure_trainable_layers(model, unfreeze_layer4=False):
    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def print_dataset_summary(name, dataset):
    diagnostics = dataset.get_diagnostics()
    print(
        f"{name} samples: {diagnostics['usable_samples']} "
        f"(missing images: {diagnostics['missing_images']})"
    )


def train():
    # 1. Device
    device = get_device()
    print("Using device:", device)
    print(
        "Training config: "
        f"batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, patience={PATIENCE}, "
        f"unfreeze_layer4={UNFREEZE_LAYER4}, "
        f"positive_weight_boost={POSITIVE_WEIGHT_BOOST}"
    )

    # 2. Paths
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    checkpoint_path = project_root / "best_full_image_model.pth"
    confusion_matrix_path = project_root / "full_image_confusion_matrix.png"
    report_path = project_root / "full_image_classification_report.txt"
    history_path = project_root / "full_image_training_history.png"

    # 3. Data
    train_loader, dev_loader, test_loader = get_dataloaders(
        data_root=data_root,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print_dataset_summary("Train", train_loader.dataset)
    print_dataset_summary("Dev", dev_loader.dataset)
    print_dataset_summary("Test", test_loader.dataset)

    # 4. Model
    model = get_full_image_model().to(device)
    configure_trainable_layers(model, unfreeze_layer4=UNFREEZE_LAYER4)

    training_mode = "classifier + layer4" if UNFREEZE_LAYER4 else "classifier only"
    print("Training mode:", training_mode)

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # 5. Compute class weights from training labels
    class_counts = Counter(train_loader.dataset.sample_labels)
    print("Class counts:", class_counts)

    total = sum(class_counts.values())
    num_classes = len(CLASS_NAMES)

    class_weights = []
    for label_id in CLASS_LABELS:
        weight = total / (num_classes * class_counts[label_id])
        class_weights.append(weight)

    positive_class_index = CLASS_LABELS.index(POSITIVE_LABEL)
    class_weights[positive_class_index] *= POSITIVE_WEIGHT_BOOST
    print(f"Applied positive class weight boost: x{POSITIVE_WEIGHT_BOOST}")

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    # 6. Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = list(model.fc.parameters())
    if UNFREEZE_LAYER4:
        trainable_params = list(model.layer4.parameters()) + trainable_params
    optimizer = torch.optim.Adam(trainable_params, lr=0.0001)

    # 7. Training state
    best_val_macro_f1 = -1.0
    no_improve_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    # 8. Training loop
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

        avg_train_loss = running_train_loss / len(train_loader)

        val_loss, val_acc, val_macro_f1, _, _ = evaluate(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Saved best model to {checkpoint_path.name} "
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

    print("\nTraining history:")
    for i in range(len(history["train_loss"])):
        print(
            f"Epoch {i + 1}: "
            f"train_loss={history['train_loss'][i]:.4f}, "
            f"val_loss={history['val_loss'][i]:.4f}, "
            f"val_acc={history['val_acc'][i]:.4f}, "
            f"val_macro_f1={history['val_macro_f1'][i]:.4f}"
        )

    save_training_history(history, history_path)
    print(f"Saved training history to {history_path.name}")

    # 9. Final test evaluation using best model
    best_model = get_full_image_model().to(device)
    configure_trainable_layers(best_model, unfreeze_layer4=UNFREEZE_LAYER4)

    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, test_macro_f1, test_labels, test_preds = evaluate(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nBest model test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")

    # 10. Confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=sorted(LABEL_TO_NAME))
    print("\nConfusion Matrix:")
    print(cm)

    save_confusion_matrix(cm, CLASS_NAMES, confusion_matrix_path)
    print(f"Saved confusion matrix to {confusion_matrix_path.name}")

    # 11. Classification report
    report = classification_report(
        test_labels,
        test_preds,
        labels=sorted(LABEL_TO_NAME),
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
