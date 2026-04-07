from pathlib import Path
import ast

from PIL import Image
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
ENDE_DIR = DATA_DIR / "ende"

INDEX_FILE = ENDE_DIR / "image_index_test.txt"
LABEL_FILE = ENDE_DIR / "sentiment_test.txt"
TEST_IMG_DIR = DATA_DIR / "test_images"


def read_first_index_line(file_path: Path) -> list[int]:
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    return ast.literal_eval(first_line)


def read_first_label(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    return int(first_line)


def label_to_text(label: int) -> str:
    mapping = {
        0: "Neutral",
        1: "Negative",
        2: "Positive",
    }
    return mapping.get(label, f"Unknown ({label})")


def main() -> None:
    indices = read_first_index_line(INDEX_FILE)
    label = read_first_label(LABEL_FILE)

    print("First test sample image indices:", indices)
    print("Label:", label)
    print("Label text:", label_to_text(label))

    if not indices:
        print("No image indices found for the first sample.")
        return

    first_image_path = TEST_IMG_DIR / f"{indices[0]}.jpg"
    print("First image path:", first_image_path)
    print("Does this image exist?", first_image_path.exists())

    if not first_image_path.exists():
        print("Image file not found.")
        return

    img = Image.open(first_image_path)
    print("Image size:", img.size)
    print("Image mode:", img.mode)

    plt.imshow(img)
    plt.title(f"Sample Image | Label: {label_to_text(label)}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
