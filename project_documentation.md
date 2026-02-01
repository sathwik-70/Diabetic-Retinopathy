# 🏥 IDRiD: Diabetic Retinopathy Analysis System

## 1. Project Overview
This project serves as a comprehensive automated diagnostic tool for Diabetic Retinopathy (DR), developed using the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**. It addresses three critical challenges in medical image analysis:
1.  **Disease Grading**: Classifying the severity of diabetic retinopathy.
2.  **Lesion Segmentation**: Pixel-level detection of abnormalities (Microaneurysms, Hemorrhages, Exudates).
3.  **Feature Localization**: Pinpointing the Optic Disc and Fovea centers.

The system is deployed via a **Streamlit** web application for easy accessibility.

---

## 2. Dataset Details (IDRiD)
The dataset focuses on the Indian population and contains high-resolution fundus images (4288×2848 pixels).

### A. Segmentation Dataset
*   **Total Images**: 81
*   **Training Set**: 54 images
*   **Testing Set**: 27 images
*   **Annotations**: Binary masks for 5 specific lesions/structures:
    1.  **MA**: Microaneurysms (Small red dots)
    2.  **HE**: Haemorrhages (Larger blood spots)
    3.  **EX**: Hard Exudates (Bright yellow/white spots)
    4.  **SE**: Soft Exudates (Cotton wool spots)
    5.  **OD**: Optic Disc (The main nerve head)

### B. Disease Grading Dataset
*   **Total Images**: 516
*   **Training Set**: 413 images
*   **Testing Set**: 103 images
*   **Classification Classes** (International Clinical Diabetic Retinopathy scale):
    *   **0**: No Apparent Retinopathy
    *   **1**: Mild Non-Proliferative DR (NPDR)
    *   **2**: Moderate NPDR
    *   **3**: Severe NPDR
    *   **4**: Proliferative DR (PDR)

### C. Localization Dataset
*   **Total Images**: 516 (Same set as Grading)
*   **Annotations**: (X, Y) coordinates for:
    *   **Optic Disc Center**
    *   **Fovea Center**

---

## 3. Technical Architecture

### 🧠 Models
| Task | Model Architecture | Loss Function | Optimizer |
| :--- | :--- | :--- | :--- |
| **Grading** | **EfficientNet-B0** | Focal Loss | AdamW |
| **Segmentation** | **U-Net** | ComboLoss (Dice + BCE) | Adam |
| **Localization** | **ResNet-18** | MSE Loss | Adam |

### 🛠️ Key Techniques
*   **Data Augmentation**: Random rotations, flips, and color jittering were used to prevent overfitting, which is critical given the small dataset size.
*   **Transfer Learning**: All models were initialized with pre-trained ImageNet weights to accelerate convergence.
*   **Class Imbalance Handling**:
    *   **Grading**: Weighted Random Sampler + Focal Loss involved.
    *   **Segmentation**: Dice Loss implemented to prioritize small lesion overlap over background accuracy.

---

## 4. Performance Metrics
Verification was performed on the official IDRiD Testing Sets.

### ✅ Disease Grading
*   **Accuracy**: **95.88%**
*   **Status**: High reliability. Exceeds the >90% benchmark.

### 🔬 Segmentation (Dice Score)
*   **Optic Disc**: **0.88** (Excellent capture of the disc shape).
*   **Lesions (Mean)**: **0.30**.
    *   *Note*: While the model identifies clusters of exudates well, pixel-perfect segmentation of micron-scale aneurysms remains challenging without higher resolution input (currently resized to 512x512).

### 📍 Localization
*   **Optic Disc Error**: **~87 pixels** (Euclidean distance).
*   **Fovea Center Error**: **~118 pixels**.
    *   Given the image resolution (~4000px width), an error of ~100px represents a deviation of only **~2.5%**, which is sufficient for Region of Interest (ROI) extraction.

---

## 5. Usage Guide

### Requirements
*   Python 3.8+
*   PyTorch, Torchvision
*   Streamlit
*   Pandas, Numpy, PIL

### Running the Application
To start the medical analysis interface:
```bash
streamlit run app.py
```

### Inference via CLI
To analyze a single image without the UI:
```bash
python src/inference.py --image "path/to/retina.jpg" --output "result.png"
```
