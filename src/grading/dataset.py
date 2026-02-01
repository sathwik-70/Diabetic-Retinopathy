import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
try:
    from src import config
except ImportError:
    import config

class IDRiDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Format: Image name, Retinopathy grade, ...
        # CSV has 'IDRiD_001', file is 'IDRiD_001.jpg'
        img_id = self.annotations.iloc[index, 0]
        # Ensure we construct the filename correctly
        img_name = os.path.join(self.root_dir, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
             # Fallback or error handling if extension is different or file missing
             # Based on ls, they are .jpg
             print(f"Warning: File not found {img_name}")
             # Return a dummy tensor or handle appropriately? 
             # For now let it fail to be noticed
             raise

        label = int(self.annotations.iloc[index, 1]) # Column 1 is Retinopathy grade

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            # RandomResizedCrop helps with scale invariance and generalization
            transforms.RandomResizedCrop(size=(config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.8, 1.0)),
            # Stronger Augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45), # Increased rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None), # Removed scale as RRC handles it
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            # CenterCrop is sometimes better for validation if we used RRC, 
            # but Resize is standard if we want to see the whole image. 
            # Sticking to Resize as retina images borders are important.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
