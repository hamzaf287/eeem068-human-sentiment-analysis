import matplotlib.pyplot as plt
from collections import Counter
from src.data.dataset import MSCTDDataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def plot_distribution(split="train"):
    dataset = MSCTDDataset(DATA_DIR, split=split)

    labels = dataset.labels
    counts = Counter(labels)

    classes = ["Neutral", "Negative", "Positive"]
    values = [counts.get(i, 0) for i in range(3)]

    plt.figure()
    plt.bar(classes, values)
    plt.title(f"{split.capitalize()} Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig(f"{split}_distribution.png")
    plt.show()


if __name__ == "__main__":
    plot_distribution("train")