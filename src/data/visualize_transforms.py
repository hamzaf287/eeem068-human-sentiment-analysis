import matplotlib.pyplot as plt
from PIL import Image
from src.data.transforms import get_train_transforms
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def show_transform():
    img_path = next((DATA_DIR / "train_images").glob("*.jpg"))
    img = Image.open(img_path).convert("RGB")

    transform = get_train_transforms()
    transformed = transform(img)

    transformed = transformed.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(transformed)
    axes[1].set_title("Transformed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("transform_comparison.png")
    plt.show()


if __name__ == "__main__":
    show_transform()