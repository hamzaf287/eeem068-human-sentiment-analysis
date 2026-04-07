from pathlib import Path

import torch

from src.data.dataloaders import get_dataloaders
from src.models.full_image_model import get_full_image_model


def main():
    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Load dataset
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "raw"
    train_loader, _, _ = get_dataloaders(data_root, batch_size=8, num_workers=0)

    # 3. Load pretrained model
    model = get_full_image_model().to(device)

    # 4. Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    # 5. Training mode
    model.train()

    running_loss = 0.0

    # 6. Train for 1 epoch
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # clear old gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Training finished. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
