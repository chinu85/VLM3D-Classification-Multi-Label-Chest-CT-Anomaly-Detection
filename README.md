# VLM3D Classification: Multi-Label Chest CT Anomaly Detection

A volumetric 3D chest CT multi-label classification pipeline for the **VLM3D Challenge Task 1** evaluated on the **CT-RATE** benchmark (18 thoracic pathologies).

---

## Key Results

| Method | Macro AUROC | Macro F1 |
|---|:---:|:---:|
| CT-CLIP (Hamamci et al., 2024) | 0.7350 | — |
| CT-Net (Hamamci et al., 2024) | 0.7620 | — |
| 3D ResNet-10 (Baseline) | 0.7245 | 0.4489 |
| DenseNet201 + ASL | 0.7932 | 0.5076 |
| SwinViT-3D + SuPreM (Solo with 8-fold TTA) | 0.7925 | 0.4655 |
| **Final Ensemble (DenseNet + SwinViT-3D + 8-fold TTA + Opt. Thr.)** | **0.8025** | **0.5212** |

---

## Core Pipeline

1. **Dynamic HU Windowing:** Dynamic percentile-based clipping per volume to preserve pulmonary tissue integrity and eliminate blank tensor artifacts.
2. **Backbones:** Fused representations from 3D DenseNet201 (dense multi-scale feature reuse) and SwinViT-3D initialized with SuPreM supervised domain weights.
3. **Asymmetric Loss (ASL):** Dynamic hard-negative suppression with per-class gamma overrides for rare thoracic findings.
4. **Post-Processing:** 8-fold spatial axis-flip Test-Time Augmentation (TTA), Nelder-Mead ensemble blending, and per-class decision threshold calibration.

---

## Repository Structure

```
├── data_loader.py            # Dynamic HU windowing, resampling, and DataLoader
├── model.py                  # DenseNet201 and SwinViT-3D architectures
├── train.py                  # Training pipeline with ASL, AMP, and cosine scheduling
├── optimize_thresholds.py    # Nelder-Mead threshold search for per-class F1
├── ensemble_models.py        # 8-fold TTA and Nelder-Mead ensemble blending
├── inference.py              # Standalone prediction script
├── submit_suprem_train.sh    # SLURM script for SwinViT-3D + SuPreM initial training
├── submit_suprem_resume.sh   # SLURM script for 150-epoch warm-restart training
├── submit_ensemble_suprem.sh # SLURM script for final 8-fold TTA ensemble evaluation
├── Dockerfile                # Docker specification for submission
├── test_docker.sh            # Container verification script
├── reseach paper/            # Long paper manuscript (LaTeX & DOCX)
└── vlm3d_challenge_guide.md  # VLM3D challenge requirements guide
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
