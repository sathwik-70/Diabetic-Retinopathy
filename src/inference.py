import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

try:
    from src import config
    from src.grading.model import get_model as get_grading_model
    from src.segmentation.model import UNet
    from src.localization.model import get_model as get_localization_model
except ImportError:
    import sys
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__)) # src
    project_root = os.path.dirname(project_root) # root
    sys.path.append(project_root)

    import src.config as config
    from src.grading.model import get_model as get_grading_model
    from src.segmentation.model import UNet
    from src.localization.model import get_model as get_localization_model

class RetinaAnalyzer:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.grading_model = None
        self.segmentation_model = None
        self.localization_model = None
        self._load_models()
        
    def _load_models(self):
        # 1. Grading Model
        print("Loading Grading Model...")
        self.grading_model = get_grading_model(pretrained=False)
        self.grading_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
        self.grading_model.to(self.device).eval()
        
        # 2. Segmentation Model
        print("Loading Segmentation Model...")
        seg_path = os.path.join(config.BASE_DIR, 'models', 'segmentation_unet.pth')
        if os.path.exists(seg_path):
            self.segmentation_model = UNet(n_channels=3, n_classes=5)
            self.segmentation_model.load_state_dict(torch.load(seg_path, map_location=self.device))
            self.segmentation_model.to(self.device).eval()
        else:
            print(f"Warning: Segmentation model not found at {seg_path}")
            
        # 3. Localization Model
        print("Loading Localization Model...")
        loc_path = os.path.join(config.BASE_DIR, 'models', 'localization_resnet.pth')
        if os.path.exists(loc_path):
            self.localization_model = get_localization_model(pretrained=False)
            self.localization_model.load_state_dict(torch.load(loc_path, map_location=self.device))
            self.localization_model.to(self.device).eval()
        else:
             print(f"Warning: Localization model not found at {loc_path}")

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size # W, H
        
        # Preprocess
        img_tensor = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        img_tensor = TF.to_tensor(img_tensor)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        results = {}
        
        # Grading
        with torch.no_grad():
            logits = self.grading_model(img_tensor)
            grade = torch.argmax(logits, dim=1).item()
            results['grade'] = grade
            results['grade_probs'] = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Segmentation
        if self.segmentation_model:
            with torch.no_grad():
                logits = self.segmentation_model(img_tensor)
                preds = torch.sigmoid(logits)
                masks = (preds > 0.5).float().cpu().numpy()[0] # (5, H, W)
                results['masks'] = masks
                
        # Localization
        if self.localization_model:
            with torch.no_grad():
                coords = self.localization_model(img_tensor).cpu().numpy()[0] # [OD_x, OD_y, Fov_x, Fov_y] normalized
                # Scale back to ORIGINAL image size
                w, h = original_size
                # Note: Model was trained with normalized coords relative to IMAGE_SIZE (which matches input tensor)
                # But wait, in dataset.py I normalized by config.IMAGE_SIZE.
                # So output is 0-1 relative to crop.
                # But valid range is 0-1.
                
                od_x = coords[0] * w 
                od_y = coords[1] * h 
                fov_x = coords[2] * w 
                fov_y = coords[3] * h 
                
                results['localization'] = {
                    'OD': (od_x, od_y),
                    'Fovea': (fov_x, fov_y)
                }
                
        return image, results

    def visualize(self, image, results, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # 1. Original Image with Localization
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Grade: {results['grade']}")
        plt.axis('off')
        
        if 'localization' in results:
            od = results['localization']['OD']
            fov = results['localization']['Fovea']
            plt.plot(od[0], od[1], 'ro', markersize=10, label='Optic Disc')
            plt.plot(fov[0], fov[1], 'bx', markersize=10, label='Fovea')
            plt.legend()
            
        # 2. Segmentation Maps (Overlay)
        if 'masks' in results:
            masks = results['masks'] # (5, 300, 300)
            # Resize masks to original image size for display
            w, h = image.size
            
            combined_mask = np.zeros((h, w, 3))
            colors = [
                (1, 0, 0), # MA - Red
                (0, 1, 0), # HE - Green
                (0, 0, 1), # EX - Blue
                (1, 1, 0), # SE - Yellow
                (0, 1, 1)  # OD - Cyan
            ]
            
            labels = ['MA', 'HE', 'EX', 'SE', 'OD']
            
            for i in range(5):
                m = masks[i]
                # Resize m to h, w
                m_img = Image.fromarray((m * 255).astype(np.uint8))
                m_img = m_img.resize((w, h), resample=Image.NEAREST)
                m_np = np.array(m_img) / 255.0
                
                # Add color
                c = np.array(colors[i]).reshape(1, 1, 3)
                combined_mask += np.expand_dims(m_np, axis=2) * c
                
            combined_mask = np.clip(combined_mask, 0, 1)
            
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            plt.imshow(combined_mask, alpha=0.5)
            plt.title("Lesion Segmentation")
            plt.axis('off')

            # Legend hack
            patches = [plt.plot([],[], marker="s", ms=10, ls="", mec=None, color=colors[i], 
                        label="{:s}".format(labels[i]))[0]  for i in range(len(labels))]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--output', type=str, default='inference_result.png', help='Output path')
    args = parser.parse_args()
    
    analyzer = RetinaAnalyzer()
    img, res = analyzer.predict(args.image)
    analyzer.visualize(img, res, save_path=args.output)
