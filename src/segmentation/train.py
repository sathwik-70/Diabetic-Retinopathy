import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
import sys
import os
import numpy as np

try:
    from src import config
    from src.segmentation.dataset import SegmentationDataset
    from src.segmentation.model import UNet
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
    from src.segmentation.dataset import SegmentationDataset
    from src.segmentation.model import UNet

def dice_coeff(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_segmentation(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_dice = 0.0
            
            # Iterate over data
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_dice += dice_coeff(outputs, masks).item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(config.BASE_DIR, 'models', 'segmentation_unet.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    model.load_state_dict(best_model_wts)
    return model

    model.load_state_dict(best_model_wts)
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    # Datasets
    train_dataset = SegmentationDataset(
        root_dir=config.SEG_TRAIN_DIR,
        mask_dir=config.SEG_MASKS_TRAIN_DIR,
        split='train'
    )
    
    val_dataset = SegmentationDataset(
        root_dir=config.SEG_TEST_DIR, # Use test set as val for now
        mask_dir=config.SEG_MASKS_TEST_DIR,
        split='val'
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4), # Lower batch size for UNet
        'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    }
    
    # Model
    # 5 Output channels (MA, HE, EX, SE, OD)
    model = UNet(n_channels=3, n_classes=5)
    
    # Loss: ComboLoss (BCE + Dice)
    from src.segmentation.loss import ComboLoss
    criterion = ComboLoss(alpha=0.5, ce_ratio=0.5)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.resume:
        save_path = os.path.join(config.BASE_DIR, 'models', 'segmentation_unet.pth')
        if os.path.exists(save_path):
            print(f"Resuming from checkpoint: {save_path}")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print("No checkpoint found to resume from. Starting from scratch.")
    
    # 50 Epochs for full training
    train_segmentation(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
