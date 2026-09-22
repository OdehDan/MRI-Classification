# Brain Tumor MRI Classification

Transfer-learning benchmark comparing six pretrained CNN architectures on multi-class brain tumor classification from MRI scans.

## Overview

This project fine-tunes six ImageNet-pretrained convolutional neural networks to classify brain MRI scans into four categories: **glioma**, **meningioma**, **no tumor**, and **pituitary tumor**. Each architecture is trained under identical conditions (same data splits, optimizer, learning rate, and epoch budget) so their performance can be compared directly.

## Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, CC BY 4.0)
- **Classes:** glioma, meningioma, notumor, pituitary
- **Splits:**
  | Split | Images |
  |---|---|
  | Training | 3,919 |
  | Validation | 1,681 |
  | Test | 1,600 |

  The validation set is a 70/30 split carved out of the original "Training" folder; the "Testing" folder is held out and used only for final evaluation.

## Architectures Compared

Each model uses ImageNet-pretrained weights with the final classification layer replaced by a 4-class linear layer:

- EfficientNet-B0
- ResNet-50
- VGG16
- MobileNetV2
- DenseNet-121
- AlexNet

## Method

- **Preprocessing:** images resized to 224x224 and normalized with ImageNet mean/std
- **Augmentation (train only):** random horizontal flip (p=0.5), random rotation (+/-10 degrees)
- **Loss:** Cross-entropy
- **Optimizer:** Adam, learning rate 1e-4
- **LR schedule:** `ReduceLROnPlateau` (factor 0.5, patience 3)
- **Early stopping:** patience of 5 epochs on validation loss
- **Epoch budget:** up to 10 epochs per architecture
- **Batch size:** 32

For each architecture, the best checkpoint (lowest validation loss) is saved and reloaded for evaluation on the test set. Results are cached in `results_backup.json` so training can resume without repeating already-completed architectures.

## Results

Test set performance (1,600 held-out images):

| Architecture | Test Accuracy |
|---|---|
| DenseNet-121 | 95.31% |
| ResNet-50 | 95.13% |
| EfficientNet-B0 | 95.00% |
| VGG16 | 94.63% |
| MobileNetV2 | 94.63% |
| AlexNet | 92.31% |

DenseNet-121 and ResNet-50 gave the strongest overall accuracy, while AlexNet trailed the rest, most notably on glioma recall.

## Requirements

- Python 3
- PyTorch, torchvision
- numpy, matplotlib, seaborn
- scikit-learn
- tqdm
- kaggle (for dataset download)

Install with:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm kaggle
```

## Usage

1. Get a Kaggle API token from your [Kaggle account settings](https://www.kaggle.com/settings) and have it ready.
2. Open `MRI.ipynb` in Jupyter/Colab.
3. Run the cells in order:
   - Enter your Kaggle API token when prompted to download the dataset.
   - Data loading, splitting, and augmentation cells build the train/val/test loaders.
   - The training loop iterates through all six architectures, saving the best checkpoint (`<architecture>_best.pth`) and appending metrics to `results_backup.json` for each.
4. Metrics (accuracy, per-class precision/recall, macro F1) and loss curves are printed and plotted per architecture as training completes.

## Repository Structure

```
.
├── MRI.ipynb            # Main notebook: data prep, training, evaluation
├── results_backup.json  # Cached per-architecture results (generated on run)
└── *_best.pth            # Best checkpoint per architecture (generated on run)
```

## Notes

- Training was run on a T4 GPU (Google Colab).
- Because results are cached in `results_backup.json`, re-running the notebook skips architectures already marked complete — delete the file to retrain from scratch.
