# 🏥 IDRiD: Unified Diabetic Retinopathy Analysis System

[![Streamlit Preview](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diabeticretinopathy404.streamlit.app)

**🟢 Live Preview Available At:** [diabeticretinopathy404.streamlit.app](https://diabeticretinopathy404.streamlit.app)

This project is a comprehensive end-to-end Deep Learning diagnostic tool designed to analyze high-resolution retinal fundus images. Built upon the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**, it addresses three distinct diagnostic challenges: Disease Severity Grading, Pixel-Level Lesion Segmentation, and Key Feature Localization.

---

## 🏗️ System Architecture

The overarching system architecture decouples the frontend UI from the deep learning inference engine. It uses a centralized `RetinaAnalyzer` to dispatch image tensors to three independent architectural pathways.

```mermaid
graph TD
    UI[Streamlit UI] -->|"Uploads Image"| Analyzer[RetinaAnalyzer]
    
    subgraph Deep Learning Engine
        Analyzer --> |"Dispatches Tensor"| EF[EfficientNet-B0<br>Grading]
        Analyzer --> |"Dispatches Tensor"| UN[U-Net<br>Segmentation]
        Analyzer --> |"Dispatches Tensor"| RN[ResNet-18<br>Localization]
    end
    
    EF --> |"Class Probabilities"| Aggregator[Results Aggregator]
    UN --> |"Multi-class Masks"| Aggregator
    RN --> |"(X, Y) Coordinates"| Aggregator
    
    Aggregator -->|"JSON Response"| UI
    
    style UI fill:#ff4b4b,stroke:#a00,stroke-width:2px,color:#fff
    style Analyzer fill:#4e79a7,stroke:#2c527e,stroke-width:2px,color:#fff
    style Deep Learning Engine fill:#f9f9f9,stroke:#666,stroke-width:1px,stroke-dasharray: 5, 5
```

---

## 🔄 Inference Pipeline

The data pipeline maps the journey of a fundus image from raw user upload through multi-stage preprocessing into concurrent model evaluation, and finally to visual composition.

```mermaid
flowchart LR
    A([Raw Image]) --> B{Validation Filter}
    B -- Invalid --> X([Reject])
    B -- Valid --> C[Resize & Normalize]
    
    C --> D1(Grading Branch)
    C --> D2(Segmentation Branch)
    C --> D3(Localization Branch)
    
    D1 --> E1[Extract Severity Grade]
    D2 --> E2[Extract 5 Lesion Masks]
    D3 --> E3[Extract OD/Fovea Coords]
    
    E1 --> F(Aggregation)
    E2 --> F
    E3 --> F
    
    F --> G[Render Visual Overlays]
    G --> H([Final Output Display])
```

---

## 🧩 UML Class Diagram

The object-oriented structure of the backend relies on specific wrapper classes mapping to individual PyTorch neural networks optimized for IDRiD subsets.

```mermaid
classDiagram
    class RetinaAnalyzer {
        -device: torch.device
        -grading_model: nn.Module
        -seg_model: nn.Module
        -loc_model: nn.Module
        +__init__(device)
        +load_models()
        +preprocess(img: Image) Tensor
        +predict(img_path: str) dict
    }
    
    class GradingModel {
        -architecture: EfficientNetB0
        -loss: FocalLoss
        -weights: grading_best.pth
        +forward(x: Tensor) logits
    }
    
    class SegmentationModel {
        -architecture: UNet
        -loss: ComboLoss
        -weights: best_unet_dice.pth
        +forward(x: Tensor) masks
    }
    
    class LocalizationModel {
        -architecture: ResNet18
        -loss: MSELoss
        -weights: localization_best.pth
        +forward(x: Tensor) coordinates
    }
    
    RetinaAnalyzer *-- GradingModel : Owns
    RetinaAnalyzer *-- SegmentationModel : Owns
    RetinaAnalyzer *-- LocalizationModel : Owns
```

---

## ⏱️ Evaluation Sequence Diagram

This sequence diagram illustrates chronological interactions during a single analysis request on the web platform, demonstrating asynchronous and parallel computation boundaries.

```mermaid
sequenceDiagram
    participant User
    participant Streamlit_App as app.py
    participant Analyzer as inference.py
    participant Models as PyTorch Models
    
    User->>Streamlit_App: Uploads Retinal Image (.jpg)
    activate Streamlit_App
    Streamlit_App->>Streamlit_App: Validate Image (Red Dominance)
    Streamlit_App->>Analyzer: predict(temp_path)
    activate Analyzer
    
    Analyzer->>Analyzer: Transform (Resize, ToTensor, Normalize)
    
    par Parallel Inference
        Analyzer->>Models: Grading.forward()
        Models-->>Analyzer: Return Grade (0-4)
    and
        Analyzer->>Models: Segmentation.forward()
        Models-->>Analyzer: Return Binary Masks
    and
        Analyzer->>Models: Localization.forward()
        Models-->>Analyzer: Return (X, Y) Pairs
    end
    
    Analyzer->>Streamlit_App: Consolidated Results Dictionary
    deactivate Analyzer
    
    Streamlit_App->>Streamlit_App: Alpha-blend Masks onto Original Image
    Streamlit_App->>Streamlit_App: Draw Localization Markers
    Streamlit_App-->>User: Display Graded Risk & Visual Mappings
    deactivate Streamlit_App
```

---

## 📋 Project Status

The project currently achieves **>95% accuracy** on Disease Severity Grading (ICDR Scale) and successfully draws high-precision pixel masks for Microaneurysms, Hard/Soft Exudates, and Haemorrhages. All required models are completely operational and accessible via the `app.py` Streamlit entrypoint.

*Documentation visually generated for isolated remote push.*
