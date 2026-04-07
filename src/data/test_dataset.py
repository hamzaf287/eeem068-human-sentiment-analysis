from pathlib import Path
from src.data.dataloaders import get_dataloaders


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def main():

    train_loader, dev_loader, test_loader = get_dataloaders(DATA_DIR)

    print("Train samples:", len(train_loader.dataset))
    print("Dev samples:", len(dev_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    img, label = next(iter(train_loader))

    print("Batch image shape:", img.shape)
    print("Batch labels:", label[:5])


if __name__ == "__main__":
    main()