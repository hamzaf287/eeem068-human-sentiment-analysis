from data.full_image_dataset import FullImageDataset

dataset = FullImageDataset(split="train")

print("Total samples:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)