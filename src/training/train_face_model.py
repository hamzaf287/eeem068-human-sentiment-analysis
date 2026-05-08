import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.face.face_dataset import FaceSentimentDataset
from src.models.face_model import FaceSentimentModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = FaceSentimentDataset(
        image_dir='data/raw/face_train_images', 
        labels_file='data/raw/ende/sentiment_train.txt'
    )
    test_dataset = FaceSentimentDataset(
        image_dir='data/raw/face_test_images', 
        labels_file='data/raw/ende/sentiment_test.txt'
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FaceSentimentModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Face Model Test Accuracy: {accuracy:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), 'best_face_model.pth')
    print("Saved best_face_model.pth to project root!")

if __name__ == "__main__":
    main()