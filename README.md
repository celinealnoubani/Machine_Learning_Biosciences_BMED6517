# Skin Cancer Classification: CNN vs Vision Transformer

## Course: BMED 6517 - Machine Learning for Biosciences (Georgia Tech)

**Team Members:** Celine Al-Noubani, Soobin An, Sharon Kartika, Daniel Lai

## Overview

This project compares Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for classifying skin cancer lesions from dermoscopic images. Using the HAM10000 dataset, we address the challenge of multi-class classification with severe class imbalance.

**Key Result:** ViT achieved **91% accuracy** compared to CNN's **76% accuracy**, demonstrating the power of transformer architectures for medical image classification.

---

## Problem Statement

- Skin cancer is one of the most common cancers worldwide
- Early detection through dermoscopy is critical but subjective
- Automated image-based classification systems can assist diagnosis
- Challenge: Small, imbalanced dataset with visually similar lesion types

---

## Dataset: HAM10000

| Property | Value |
|----------|-------|
| Total Images | 10,015 dermoscopic images |
| Classes | 7 skin lesion types |
| Source | ISIC Archive |

### Class Distribution (Severe Imbalance)

| Class | Full Name | Proportion |
|-------|-----------|------------|
| nv | Melanocytic Nevi | 67% |
| mel | Melanoma | ~11% |
| bkl | Benign Keratosis | ~11% |
| bcc | Basal Cell Carcinoma | ~5% |
| akiec | Actinic Keratoses | ~3% |
| vasc | Vascular Lesions | <2% |
| df | Dermatofibroma | <2% |

---

## Methodology

### Pipeline Overview
```
Input (Dermoscopic Images)
    ↓
Preprocessing (Resize → Augmentation → Split)
    ↓
Dimensionality Check (PCA & UMAP → heavy overlap confirmed)
    ↓
    ├── Baseline CNN (Class Weights + Focal Loss)
    └── Advanced ViT (Oversampling + Fine-tuning)
    ↓
Evaluation (Accuracy, F1, Confusion Matrix, ROC)
```

### Class Imbalance Handling

| Model | Strategy |
|-------|----------|
| CNN | Class weights (penalize minority class errors more) |
| ViT | Oversampling (duplicate minority classes to balance) |

### Data Augmentation

| Technique | CNN (Keras) | ViT (PyTorch) |
|-----------|-------------|---------------|
| Input Size | 64×64×3 | 224×224 |
| Rotation | ±20° | ±50° |
| Translation | ±10% | RandomAffine ±10% |
| Zoom | ±10% | ±10% |
| Flip | Horizontal | Horizontal |
| Brightness | 0.8–1.2 | 0.8–1.2 |
| Normalization | [0,1] scaling | ImageNet stats |

---

## Model Architectures

### CNN (Baseline → Enhanced → Focal Loss)

**Architecture Evolution:**

| Model | Key Features | Test Accuracy |
|-------|--------------|---------------|
| Basic CNN | 2 Conv blocks (32, 64), Dense(128), Dropout(0.5) | 53.96% |
| Enhanced CNN | 3 Conv blocks (32, 64, 128), BatchNorm, Dense(256) | 62.61% |
| **Focal Loss CNN** | Focal loss (γ=2.0), LR=0.0001, Dropout(0.3) | **76.18%** |

**Final CNN Architecture:**
```
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.3) → Dense(7)
```

### Vision Transformer (ViT)

**Model:** `google/vit-base-patch16-224-in21k`
- Pretrained on ImageNet-21k (14M images, 21,843 classes)
- Fine-tuned on HAM10000
- 12 Transformer encoder layers
- ~86 million parameters

**Training:**
- 10 epochs on 80% training data
- Oversampled to 5,352 samples per class (37,464 total)
- Learning rate: 5e-5 with warmup

---

## Results

### Overall Performance Comparison

| Metric | CNN (Focal Loss) | ViT |
|--------|------------------|-----|
| **Test Accuracy** | 76% | **91%** |
| Weighted F1 | 0.75 | 0.91 |
| Weighted Precision | 0.75 | 0.91 |
| Weighted Recall | 0.76 | 0.91 |
| Macro Avg F1 | 0.53 | 0.85 |

### Per-Class Performance (F1-Score)

| Class | CNN | ViT | Improvement |
|-------|-----|-----|-------------|
| akiec | 0.40 | 0.95 | +137% |
| bcc | 0.45 | 0.79 | +76% |
| bkl | 0.54 | 0.84 | +56% |
| df | 0.27 | 0.96 | +256% |
| mel | 0.49 | 0.67 | +37% |
| nv | 0.88 | 0.88 | 0% |
| vasc | 0.65 | 0.88 | +35% |

### ROC-AUC Scores

| Class | CNN | ViT |
|-------|-----|-----|
| akiec | 0.96 | 1.00 |
| bcc | 0.93 | 0.97 |
| bkl | 0.90 | 0.97 |
| df | 0.87 | 0.98 |
| mel | 0.89 | 0.85 |
| nv | 0.92 | 0.99 |
| vasc | 0.97 | 1.00 |

---

## Key Findings

### CNN Strengths & Weaknesses
- **Strengths:** Effective with localized features; focal loss improved hard sample learning
- **Weaknesses:** Limited global context capture; struggled with minority classes (df: 0.27 F1)

### ViT Strengths & Weaknesses
- **Strengths:** Captures long-range dependencies; excellent on most classes after oversampling
- **Weaknesses:** Requires more data and compute; mel class still challenging (0.67 F1)

### Why ViT Outperformed CNN
1. **Global attention** captures relationships across entire image
2. **Transfer learning** from ImageNet-21k provides strong feature representations
3. **Oversampling** more effective than class weights for transformer training
4. **Higher resolution** (224×224 vs 64×64) preserves diagnostic details

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow/Keras, PyTorch, HuggingFace Transformers |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dimensionality Reduction | PCA, UMAP |
| Environment | Google Colab (T4 GPU) |

---

## Repository Structure

```
Machine_Learning_Biosciences_BMED6517/
├── Skin_Cancer_Classification.ipynb    # Main notebook with CNN & ViT
├── BMED 6517 Group 4 Project Presentation.pdf
└── README.md
```

---

## Future Work

- **CNN:** Hyperparameter optimization; pretrain on larger dermatology dataset
- **ViT:** Investigate attention maps for interpretability; improve melanoma detection
- **Ensemble:** Combine CNN and ViT predictions for improved robustness

---

## References

1. Tschandl, P., et al. (2018). The HAM10000 dataset. *Scientific Data*, 5, 180161.
2. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *arXiv:2010.11929*
3. Codella, N., et al. (2018). Skin Lesion Analysis Toward Melanoma Detection. *ISIC Challenge*.

---

## Team

- Celine Al-Noubani
- Soobin An
- Sharon Kartika
- Daniel Lai

Georgia Institute of Technology
