from pathlib import Path
from src.data.dataloaders import get_dataloaders


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


LABEL_TO_NAME = {
    0: "neutral",
    1: "negative",
    2: "positive",
}


def format_distribution(distribution):
    return {
        LABEL_TO_NAME.get(label, str(label)): count
        for label, count in sorted(distribution.items())
    }


def print_dataset_summary(name, dataset):
    diagnostics = dataset.get_diagnostics()

    print(f"{name} samples:", len(dataset))
    print(f"{name} label lines:", diagnostics["label_lines"])
    print(f"{name} image files:", diagnostics["image_files"])
    print(f"{name} missing images:", diagnostics["missing_images"])
    print(
        f"{name} class distribution:",
        format_distribution(diagnostics["label_distribution"])
    )


def main():
    train_loader, dev_loader, test_loader = get_dataloaders(DATA_DIR, num_workers=0)

    print_dataset_summary("Train", train_loader.dataset)
    print_dataset_summary("Dev", dev_loader.dataset)
    print_dataset_summary("Test", test_loader.dataset)

    img, label = next(iter(train_loader))

    print("Batch image shape:", img.shape)
    print("Batch labels:", label[:5])


if __name__ == "__main__":
    main()
