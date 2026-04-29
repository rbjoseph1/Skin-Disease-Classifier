# Skin Disease Classification Using Convolutional Neural Networks
A comparative study of CNN architectures for automated skin disease classification across 23 categories, comparing training from scratch approaches against transfer learning.

## Author
Rose Joseph — Georgia State University

## Project Overview
This project investigates how different CNN training strategies affect classification performance on a challenging 23-class skin disease dataset. It began from a simple assumption: that a CNN trained from scratch could meaningfully classify skin diseases. Five models are compared to test that assumption systematically:

- **Baseline CNN** — standard 4-block CNN trained from scratch
- **Augmented CNN** — same architecture with data augmentation
- **Dropout CNN** — same architecture with dropout regularization
- **ResNet50 (Frozen)** — pretrained ResNet50 with frozen convolutional base
- **ResNet50 (Unfrozen)** — pretrained ResNet50 fine-tuned end-to-end

## Dataset
- 19,559 images across 23 skin disease classes
- 15,557 training images / 4,002 test images
- 541 cross-boundary duplicates removed to prevent data leakage
- 6.63:1 class imbalance ratio addressed via class-weighted loss function
- Images normalized using dataset-specific channel means (R: 0.541, G: 0.414, B: 0.382)

## Results
| Model | Accuracy | F1 Score | Overfit Gap |
|-------|----------|----------|-------------|
| Baseline CNN | 29.56% | 29.17% | 10.7% |
| Augmented CNN | 33.58% | 33.76% | 20.8% |
| Dropout CNN | 26.44% | 25.11% | -3.7% |
| ResNet50 (Frozen) | 34.48% | 33.62% | ~4% |
| **ResNet50 (Unfrozen)** | **53.47%** | **53.79%** | ~18% |

> Random chance baseline for a 23-class problem: **4.3%**  
> Best model improvement over random chance: **12.4×**  
> Transfer learning gain over best scratch CNN: **+19.89 percentage points**

## Standout Per-Class Results (ResNet50 Unfrozen)
| Class | F1 Score |
|-------|----------|
| Nail Fungus | 82% 🏆 |
| Acne and Rosacea | 78% |
| Melanoma | 66% |
| Systemic Disease | 34% |
| Cellulitis | 27% |

## Key Findings
- All scratch CNNs significantly exceed random chance, confirming that learning occurred
- Scratch CNNs plateau around 33–34% regardless of regularization strategy — a data ceiling, not an architecture ceiling
- Augmentation improves accuracy but substantially increases the overfitting gap (20.8%)
- Dropout at 50 epochs produces a **negative** overfit gap (-3.7%), indicating underfitting — regularization costs require sufficient training budget to overcome
- Frozen transfer learning matches the best scratch CNN without any task-specific feature learning
- Unfrozen ResNet50 outperforms the best scratch CNN by nearly **20 percentage points**, nearly doubling performance
- The primary bottleneck for from-scratch training is data volume, not architecture

## Project Structure
```
├── data/
│   ├── train/          # 15,557 training images across 23 classes
│   └── test/           # 4,002 test images across 23 classes
├── models/
│   ├── baseline_cnn.py
│   ├── augmented_cnn.py
│   ├── dropout_cnn.py
│   └── resnet50_transfer.py
├── figures/
│   ├── class_distribution.png
│   ├── architecture.png
│   ├── baseline_curves.png
│   ├── confusion_matrix.png
│   └── sample_grid.png
├── paper/
│   └── skin_disease_paper.tex
└── README.md
```

## Requirements
```
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
```

## Reproducing Results
```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline CNN
python models/baseline_cnn.py

# Train with augmentation
python models/augmented_cnn.py

# Train with dropout
python models/dropout_cnn.py

# Fine-tune ResNet50
python models/resnet50_transfer.py
```

## Paper
The full write-up is available in `paper/skin_disease_paper.tex`, formatted for ACM SIGCONF. It covers dataset preprocessing, model architecture, training configuration, results analysis, and discussion of findings.

## GitHub
[https://github.com/rbjoseph1/Skin-Disease-Classifier](https://github.com/rbjoseph1/Skin-Disease-Classifier)