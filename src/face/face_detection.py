import os
import glob
import torch
from facenet_pytorch import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(keep_all=True, device=device)
    
    def process_image(self, image_path, output_path, target_size=(224, 224)):
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return False

        boxes, _ = self.mtcnn.detect(img)
        
        if boxes is not None and len(boxes) > 0:
            largest_box = None
            max_area = 0
            for box in boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = box
            
            x1, y1, x2, y2 = [int(coord) for coord in largest_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            
            face_img = img.crop((x1, y1, x2, y2))
        else:
            width, height = img.size
            new_width, new_height = min(width, target_size[0]), min(height, target_size[1])
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            face_img = img.crop((left, top, right, bottom))
        
        face_img = face_img.resize(target_size, Image.Resampling.LANCZOS)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        face_img.save(output_path)
        return True

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FaceExtractor(device=device)
    
    datasets = {
        'train': ('data/raw/train_images', 'data/raw/face_train_images'),
        'dev': ('data/raw/dev_images', 'data/raw/face_dev_images'),
        'test': ('data/raw/test_images', 'data/raw/face_test_images')
    }
    
    for split, (in_dir, out_dir) in datasets.items():
        print(f"Processing {split} set...")
        os.makedirs(out_dir, exist_ok=True)
        
        image_paths = glob.glob(os.path.join(in_dir, '*.jpg'))
        if not image_paths:
            print(f"No images found in {in_dir}. Check your paths!")
            continue
            
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            out_path = os.path.join(out_dir, filename)
            extractor.process_image(img_path, out_path)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images in {split}...")
                
        print(f"Finished {split} set! Saved to {out_dir}")