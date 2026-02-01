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
    from src.localization.dataset import LocalizationDataset
    from src.localization.model import get_model
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
    from src.localization.dataset import LocalizationDataset
    from src.localization.model import get_model

def train_localization(model, dataloaders, criterion, optimizer, num_epochs=25):
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
            
            # Iterate over data
            for inputs, coords in dataloaders[phase]:
                inputs = inputs.to(device)
                coords = coords.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, coords)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} MSE Loss: {epoch_loss:.6f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(config.BASE_DIR, 'models', 'localization_resnet.pth')
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
    # Note: Using LOC_TRAIN_DIR which points to Segmentation Images 
    # (assuming images are shared. Need to verify if Localization has own images - config suggests they might)
    # Step 213 in config: LOC_TRAIN_IMAGES = os.path.join(LOC_IMAGES_DIR, "a. Training Set")
    
    train_dataset = LocalizationDataset(
        root_dir=config.LOC_TRAIN_IMAGES,
        od_csv=config.LOC_OD_TRAIN_CSV,
        fovea_csv=config.LOC_FOVEA_TRAIN_CSV
    )
    
    val_dataset = LocalizationDataset(
        root_dir=config.LOC_TEST_IMAGES,
        od_csv=config.LOC_OD_TEST_CSV,
        fovea_csv=config.LOC_FOVEA_TEST_CSV
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    }
    
    model = get_model()
    
    # MSE Loss for regression
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.resume:
        save_path = os.path.join(config.BASE_DIR, 'models', 'localization_resnet.pth')
        if os.path.exists(save_path):
            print(f"Resuming from checkpoint: {save_path}")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(save_path, map_location=device))
        else:
            print("No checkpoint found to resume from. Starting from scratch.")
    
    # 50 Epochs for full training
    train_localization(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
