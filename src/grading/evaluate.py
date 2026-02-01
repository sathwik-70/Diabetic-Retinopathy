import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.grading import dataset, model

def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    # Use 'get_transforms("val")' for testing
    test_dataset = dataset.IDRiDDataset(
        csv_file=config.TEST_LABELS_PATH,
        root_dir=config.TEST_IMAGES_DIR,
        transform=dataset.get_transforms('val')
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load Model
    dr_model = model.get_model(num_classes=config.NUM_CLASSES, pretrained=False)
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading model from {config.MODEL_SAVE_PATH}")
        dr_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Model file not found!")
        return

    dr_model.to(device)
    dr_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = dr_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds)
    print(report)
    
    print("\nConfusion Matrix:")
    cm_text = str(confusion_matrix(all_labels, all_preds))
    print(cm_text)
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    with open(os.path.join(config.BASE_DIR, 'evaluation_results.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm_text)
        f.write(f"\n\nOverall Accuracy: {acc:.4f}")

    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(config.NUM_CLASSES)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.BASE_DIR, 'confusion_matrix.png'))
    print("Confusion matrix saved.")

if __name__ == "__main__":
    evaluate()
