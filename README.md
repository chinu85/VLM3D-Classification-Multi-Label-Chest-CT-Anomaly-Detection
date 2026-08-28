# VLM3D Classification: Multi-Label Chest CT Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Sourabh Kumawat  
**Affiliation:** University College Dublin (UCD)  
**Contact:** [sourabh.kumawat@ucdconnect.ie](mailto:sourabh.kumawat@ucdconnect.ie)

---

## Overview

This repository contains the complete end-to-end classification system for the **VLM3D Challenge Task 1** (Multi-Label 3D Chest CT Pathology Classification) evaluated on the **CT-RATE** benchmark (11,404 training volumes, 3,039 validation volumes across 18 distinct thoracic abnormalities).

The pipeline combines:
1. **Dynamic HU Windowing:** Resolves metadata-shifting artifacts by dynamically setting intensity windows from patient-specific percentiles, preventing blank tensor clipping.
2. **Hybrid Architecture Ensemble:** Fuses representations from **DenseNet201** (multi-scale convolutional feature reuse) and **SwinViT-3D** initialized from **SuPreM** (supervised pre-training across 2,100 3D CT volumes).
3. **Imbalance-Tuned Asymmetric Loss (ASL):** Dynamic hard-negative suppression with customized gamma overrides for low-prevalence thoracic findings.
4. **8-Fold Test-Time Augmentation (TTA):** Spatial axis-flip permutation averaging along all 3D orthogonal planes.
5. **Nelder-Mead Metric Optimization:** Continuous simplex-based ensemble weight search and per-class decision threshold calibration.

### Headline Results

| Method | Macro AUROC | Macro F1 |
|---|:---:|:---:|
| CT-CLIP (Hamamci et al., 2024) | 0.7350 | — |
| CT-Net (Hamamci et al., 2024) | 0.7620 | — |
| 3D ResNet-10 (Baseline) | 0.7245 | 0.4489 |
| DenseNet201 + ASL | 0.7932 | 0.5076 |
| SwinViT-3D + SuPreM (Solo with 8-fold TTA) | 0.7925 | 0.4655 |
| **Final Ensemble (DenseNet + SwinViT-3D + 8-fold TTA + Opt. Thr.)** | **0.8025** | **0.5212** |

---

## 18-Class Performance Breakdown

| # | Pathology Finding | Standalone Peak AUROC | Standalone Flat F1 | Final Ensemble F1 (Opt. Threshold) |
|:---:|---|:---:|:---:|:---:|
| 1 | Medical material | 0.7785 | 0.2876 | **0.3866** |
| 2 | Arterial wall calcification | 0.8711 | 0.6058 | **0.7040** |
| 3 | Cardiomegaly | 0.8958 | 0.4138 | **0.5578** |
| 4 | Pericardial effusion | 0.8374 | 0.3036 | **0.4423** |
| 5 | Coronary artery calcification | 0.8698 | 0.5807 | **0.6475** |
| 6 | Hiatal hernia | 0.7062 | 0.2883 | **0.3646** |
| 7 | Lymphadenopathy | 0.7098 | 0.4161 | **0.5147** |
| 8 | Emphysema | 0.7584 | 0.3936 | **0.4927** |
| 9 | Atelectasis | 0.7129 | 0.4016 | **0.4829** |
| 10 | Lung nodule | 0.6630 | 0.6400 | **0.6451** |
| 11 | Lung opacity | 0.7800 | 0.5643 | **0.6784** |
| 12 | Pulmonary fibrotic sequela | 0.6505 | 0.4650 | **0.4734** |
| 13 | Pleural effusion | 0.9284 | 0.5779 | **0.7436** |
| 14 | Mosaic attenuation pattern | 0.8476 | 0.3428 | **0.4640** |
| 15 | Peribronchial thickening | 0.7654 | 0.2719 | **0.4158** |
| 16 | Air trapping | 0.8401 | 0.4655 | **0.5857** |
| 17 | Bronchiectasis | 0.7540 | 0.3850 | **0.3989** |
| 18 | Pleural thickening | 0.8391 | 0.3037 | **0.3836** |
| — | **Macro Average** | **0.7893** | **0.4282** | **0.5212** |

---

## Repository Structure

```
├── data_loader.py            # Dynamic HU windowing, spatial resampling, and batch generator
├── model.py                  # PyTorch / MONAI DenseNet201 and SwinViT-3D model architectures
├── train.py                  # Training pipeline with ASL loss, AMP, and cosine scheduling
├── optimize_thresholds.py    # Nelder-Mead threshold search for per-class F1 maximization
├── ensemble_models.py        # 8-fold TTA and multi-backbone Nelder-Mead blending
├── inference.py              # Standalone containerized prediction script
├── submit_suprem_train.sh    # SLURM script for SwinViT-3D + SuPreM initial training
├── submit_suprem_resume.sh   # SLURM script for 150-epoch warm-restart training
├── submit_ensemble_suprem.sh # SLURM script for final 8-fold TTA ensemble evaluation
├── Dockerfile                # Docker specification for competition submission container
├── reseach paper/            # Complete LaTeX and docx manuscripts with tables and figures
└── README.md
```

---

## Setup & Installation

### Environment Setup
```bash
conda create -n vlm3d python=3.10 -y
conda activate vlm3d

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel scipy scikit-learn pandas numpy tqdm
```

### Pre-trained SuPreM Weights
Download the pre-trained supervised weights:
```bash
mkdir -p pretrain_weights
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pt -O pretrain_weights/model_swinvit.pt
```

---

## Quickstart

### 1. Training SwinViT-3D
```bash
python train.py \
    --use_swin \
    --ssl_weights_path pretrain_weights/model_swinvit.pt \
    --checkpoint_path best_model_checkpoint_swin_suprem.pth \
    --best_model_path best_model_swin_suprem.pth \
    --num_epochs 150 \
    --lr 1e-4 \
    --reset_scheduler \
    --weak_class_gamma 2.0
```

### 2. Evaluating the Ensemble with 8-Fold TTA
```bash
python ensemble_models.py \
    --densenet_path best_model.pth \
    --swin_path best_model_swin_suprem.pth \
    --use_tta
```

### 3. Docker Verification
```bash
docker build -t vlm3d-submission .
bash test_docker.sh
```

---

## Citation

If you find this work useful in your research, please cite our challenge paper:

```bibtex
@article{kumawat2026vlm3d,
  title={VLM3D Classification: Multi-Label Chest CT Anomaly Detection},
  author={Kumawat, Sourabh},
  journal={University College Dublin Technical Report},
  year={2026}
}
```
