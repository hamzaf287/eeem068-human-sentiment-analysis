import matplotlib.pyplot as plt
from src.data.dataloaders import get_dataloaders
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def show_samples():
    train_loader, _, _ = get_dataloaders(DATA_DIR, batch_size=8)

    images, labels = next(iter(train_loader))

    images = images.permute(0, 2, 3, 1).numpy()  # CHW → HWC

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()


if __name__ == "__main__":
    show_samples()