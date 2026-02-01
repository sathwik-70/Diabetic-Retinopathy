import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
import sys
import os

try:
    from src import config
    from src.grading.dataset import IDRiDDataset, get_transforms
    from src.grading.model import get_model
except ImportError:
    # Allow running from src folder or root
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    from src.grading.dataset import IDRiDDataset, get_transforms
    from src.grading.model import get_model
    # from src.grading.loss import FocalLoss # Not used currently

# Make sure FocalLoss is imported regardless of path
from src.grading.loss import FocalLoss

import json
import matplotlib.pyplot as plt

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, patience=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Store metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Log metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Step scheduler if it is ReduceLROnPlateau
                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        # CosineAnnealing steps per epoch
                        scheduler.step() 

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                    print(f"Saved best model with Acc: {best_acc:.4f}")
                
                # Early Stopping Check on Loss
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"EarlyStopping counter: {epochs_no_improve} out of {patience}")
                    
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Save history
    with open(os.path.join(config.BASE_DIR, 'metrics.json'), 'w') as f:
        json.dump(history, f)
        
    # Plot curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig(os.path.join(config.BASE_DIR, 'loss_curve.png'))
    print("Metrics and plots saved.")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    args = parser.parse_args()

    # Datasets with appropriate transforms
    dataset_full = IDRiDDataset(
        csv_file=config.TRAIN_LABELS_PATH,
        root_dir=config.TRAIN_IMAGES_DIR,
        transform=get_transforms('train') # Strong augmentation for train
    )
    
    dataset_val = IDRiDDataset(
        csv_file=config.TRAIN_LABELS_PATH,
        root_dir=config.TRAIN_IMAGES_DIR,
        transform=get_transforms('val') # No augmentation for val
    )

    # Manual split to keep track of indices for weight calculation
    import numpy as np
    indices = list(range(len(dataset_full)))
    # Fixed seed for reproducibility of split
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(indices)
    
    split = int(np.floor(0.2 * len(dataset_full)))
    train_idx, val_idx = indices[split:], indices[:split]

    import os
    num_workers = min(4, os.cpu_count() or 1)
    print(f"Using {num_workers} workers for data loading")
    
    dataloaders = {
        'train': DataLoader(dataset_full, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers),
        'val': DataLoader(dataset_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    }

    # Calculate Class Weights for Focal Loss alpha (optional but recommended)
    print("Calculating class weights...")
    all_labels = dataset_full.annotations.iloc[:, 1].values
    train_labels = all_labels[train_idx]
    
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    total_samples = len(train_labels)
    class_weights = total_samples / (config.NUM_CLASSES * (class_counts + 1e-6))
    
    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {class_weights}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    
    # Loss Function: Focal Loss
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    
    model = get_model()

    if args.resume:
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"Resuming from checkpoint: {config.MODEL_SAVE_PATH}")
            try:
                model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            except RuntimeError:
                print("Checkpoint architecture mismatch (probably switching models). Starting from scratch.")
        else:
            print("No checkpoint found to resume from. Starting from scratch.")

    # Use AdamW
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    train_model(model, dataloaders, criterion, optimizer, scheduler=scheduler, num_epochs=args.epochs, patience=20)

if __name__ == '__main__':
    main()
