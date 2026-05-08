import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FaceSentimentDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        # Ensure we sort the files so they match line-by-line with the labels text file
        # Assuming filenames are numbers like "0.jpg", "1.jpg", we sort numerically
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')], 
                                  key=lambda x: int(os.path.splitext(x)[0]))
        
        with open(labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f.readlines() if line.strip()]
            
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label