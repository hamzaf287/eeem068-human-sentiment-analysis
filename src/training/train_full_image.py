from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models

from src.models.full_image_model import get_full_image_model
from src.data.full_image_dataset import FullImageDataset

def main():
    # 1. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Load dataset
    train_dataset = FullImageDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 3. Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    # 4. Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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