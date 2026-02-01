import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random

try:
    from src import config
except ImportError:
    import config

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, mask_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.split = split
        self.transform = transform
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        
        # Lesion codes
        self.lesions = {
            'MA': '1. Microaneurysms',
            'HE': '2. Haemorrhages',
            'EX': '3. Hard Exudates',
            'SE': '4. Soft Exudates',
            'OD': '5. Optic Disc'
        }
        # Order of channels: MA, HE, EX, SE, OD
        self.channels = ['MA', 'HE', 'EX', 'SE', 'OD']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_id = os.path.splitext(img_filename)[0] # e.g. IDRiD_01
        
        # Load Image
        img_path = os.path.join(self.root_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        # Load Masks
        masks = []
        for code in self.channels:
            # Mask format: IDRiD_01_MA.tif
            # Located in mask_dir / Subfolder / IDRiD_01_MA.tif
            mask_subdir = self.lesions[code]
            mask_filename = f"{img_id}_{code}.tif"
            mask_path = os.path.join(self.mask_dir, mask_subdir, mask_filename)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L") # Grayscale
            else:
                # Create empty mask if not found
                # Use image size
                w, h = image.size
                mask = Image.new('L', (w, h), 0)
            
            masks.append(mask)
            
        # Apply Transforms to both Image and Masks
        if self.transform:
             # Custom transform handling manually or return logic
             pass

        # For simplicity, we implement basic transforms here directly to ensure sync
        # Resize
        target_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
        image = TF.resize(image, target_size)
        masks = [TF.resize(m, target_size, interpolation=Image.NEAREST) for m in masks]
        
        if self.split == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                masks = [TF.hflip(m) for m in masks]
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                masks = [TF.vflip(m) for m in masks]
                
            # Random Rotation
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle)
                masks = [TF.rotate(m, angle) for m in masks]

        # Convert to Tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        masks_tensor = []
        for m in masks:
            m_np = np.array(m)
            # Threshold: masks are not 0-255, found 76. distinct > 0 is correct
            m_np = (m_np > 0).astype(np.float32)
            masks_tensor.append(torch.from_numpy(m_np))
            
        masks_tensor = torch.stack(masks_tensor, dim=0) # (5, H, W)
        
        return image, masks_tensor
