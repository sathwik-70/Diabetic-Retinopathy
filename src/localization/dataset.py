import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import random

try:
    from src import config
except ImportError:
    import config

class LocalizationDataset(Dataset):
    def __init__(self, root_dir, od_csv, fovea_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSVs
        # Skip bad lines if any
        self.od_df = pd.read_csv(od_csv, on_bad_lines='skip')
        self.fovea_df = pd.read_csv(fovea_csv, on_bad_lines='skip')
        
        # Clean columns (remove extra spaces)
        self.od_df.columns = [c.strip() for c in self.od_df.columns]
        self.fovea_df.columns = [c.strip() for c in self.fovea_df.columns]
        
        # Filter valid rows (Image No must be string starting with IDRiD)
        self.od_df = self.od_df[self.od_df['Image No'].notna()]
        self.fovea_df = self.fovea_df[self.fovea_df['Image No'].notna()]
        
        # Merge on 'Image No' or ID
        # Rename cols to avoid collision
        self.od_df = self.od_df[['Image No', 'X- Coordinate', 'Y - Coordinate']].rename(
            columns={'X- Coordinate': 'OD_X', 'Y - Coordinate': 'OD_Y'}
        )
        self.fovea_df = self.fovea_df[['Image No', 'X- Coordinate', 'Y - Coordinate']].rename(
            columns={'X- Coordinate': 'Fovea_X', 'Y - Coordinate': 'Fovea_Y'}
        )
        
        # Inner join to ensure we have both (or outer if we handle missing)
        # Using inner join for now
        self.data = pd.merge(self.od_df, self.fovea_df, on='Image No', how='inner')
        
        self.images = self.data['Image No'].values
        self.coords = self.data[['OD_X', 'OD_Y', 'Fovea_X', 'Fovea_Y']].values.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id = self.images[index]
        img_name = f"{img_id}.jpg" # Check if .jpg exists
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
             # Try finding it recursively or handle error
             # Assuming standard layout
             raise FileNotFoundError(f"Image {img_name} not found in {self.root_dir}")

        w, h = image.size
        coords = self.coords[index].copy() # [OD_X, OD_Y, Fovea_X, Fovea_Y]
        
        # Resize logic
        # We resize image to IMAGE_SIZE
        # We must scale coordinates
        scale_x = config.IMAGE_SIZE / w
        scale_y = config.IMAGE_SIZE / h
        
        image = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        coords[0] *= scale_x
        coords[2] *= scale_x
        coords[1] *= scale_y
        coords[3] *= scale_y
        
        # Data Augmentation (Flip)
        # Need to flip coords too
        
        # Convert to Tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Normalize coords to [0, 1] for stability?
        # Or keep pixels. ResNet Regressor usually handles pixel values fine if normalized properly
        # Let's normalize to [0, 1] relative to image size
        coords[0] /= config.IMAGE_SIZE
        coords[1] /= config.IMAGE_SIZE
        coords[2] /= config.IMAGE_SIZE
        coords[3] /= config.IMAGE_SIZE
        
        return image, torch.tensor(coords, dtype=torch.float32)
