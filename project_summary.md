# Project Summary: VLM3D Multi-Abnormality Chest CT Classification

Comprehensive technical reference for the **VLM3D Classification Challenge** (MICCAI 2024).
Covers dataset, model architectures, hyperparameters, training configurations, and all results to date.

---

## 1. Project Overview & Clinical Objectives

The goal is to develop a deep learning pipeline to automate **multi-label classification** of 3D Chest CT scans, detecting **18 distinct pathologies** simultaneously from a single volumetric scan.

### The 18 CT-RATE Pathology Classes
| # | Class Name | Prevalence (approx.) |
|---|-----------|---------------------|
| 1 | Medical material | Common |
| 2 | Arterial wall calcification | Common |
| 3 | Cardiomegaly | Moderate |
| 4 | Pericardial effusion | Rare |
| 5 | Coronary artery wall calcification | Moderate |
| 6 | Hiatal hernia | Rare |
| 7 | Lymphadenopathy | Moderate |
| 8 | Emphysema | Moderate |
| 9 | Atelectasis | Common |
| 10 | Lung nodule | Common |
| 11 | Lung opacity | Common |
| 12 | Pulmonary fibrotic sequela | Moderate |
| 13 | Pleural effusion | Moderate |
| 14 | Mosaic attenuation pattern | Rare |
| 15 | Peribronchial thickening | Moderate |
| 16 | Air trapping | Moderate |
| 17 | Bronchiectasis | Rare |
| 18 | Pleural thickening | Moderate |

### Clinical Relevance
- **Clinical Triage**: Accelerate diagnosis of urgent cases (Pneumothorax, Fractures) in emergency workflows
- **Radiology Automation**: Standardize reporting and lay the groundwork for automated radiology report generation
- **Radiologist Assistance**: Reduce cognitive load and diagnostic variability

---

## 2. Dataset: CT-RATE

| Property | Value |
|----------|-------|
| **Dataset Name** | CT-RATE (Chest CT with Radiology Reports and Annotations) |
| **Task** | Multi-label 3D volumetric classification |
| **Total Volumes** | 14,443 unique 3D NIfTI volumes |
| **Training Split** | **11,404 volumes** |
| **Validation Split** | **3,039 volumes** |
| **Early Baseline Subset** | 2,000 volumes (pipeline prototyping only) |
| **Volume Format** | `.nii.gz` compressed NIfTI |
| **Typical Volume Size** | ~1.5 GB uncompressed per scan |
| **Voxel Spacing (original)** | Variable (0.5mm – 3.0mm, manufacturer-dependent) |
| **Labels per scan** | Multi-hot binary vector of shape `(18,)` |
| **Class Imbalance** | Severe — rare classes (Hiatal hernia, Bronchiectasis) appear in <5% of scans |
| **Storage Location (HPC)** | `/scratch/25208443/data_volumes/dataset/` |
| **Label CSV Location** | `/scratch/25208443/dataset/dataset/multi_abnormality_labels/` |

---

## 3. Preprocessing Pipeline

| Step | Configuration |
|------|--------------|
| **Orientation** | Reoriented to RAS (Right-Anterior-Superior) axcodes |
| **Voxel Resampling** | Bilinear resampling to uniform **1.5 × 1.5 × 1.5 mm** spacing |
| **Intensity Normalization** | Dynamic percentile windowing: `lower=5%, upper=95%` → `[0.0, 1.0]` |
| **Foreground Crop** | `CropForegroundd` removes empty air padding |
| **Spatial Padding** | Padded to minimum `(160, 160, 112)` voxels |
| **Train Crop** | Random spatial crop `160 × 160 × 112` |
| **Val Crop** | Fixed center crop `160 × 160 × 112` |
| **Augmentations (train)** | Random flips (all 3 axes, p=0.5), Random affine (rotate ±0.15 rad, scale ±10%), Gaussian noise (p=0.2, σ=0.05), Gibbs noise (p=0.15), Intensity scale & shift |
| **Caching** | MONAI `PersistentDataset` at `/scratch/25208443/monai_cache_v4/` |
| **Metadata Stripping** | Custom `NakedDataset` wrapper strips MONAI MetaTensor to raw `torch.Tensor` |

---

## 4. Model Architectures

### Model A — 3D ResNet-10 (Baseline)
| Property | Value |
|----------|-------|
| **Architecture** | MONAI 3D ResNet-10 |
| **Backbone Parameters** | ~14.7M |
| **Input Shape** | `(1, 1, 128, 128, 96)` |
| **Output** | 18-class logits (linear head) |
| **Pre-training** | None (random init) |

---

### Model B — 3D DenseNet201 (Current Best)
| Property | Value |
|----------|-------|
| **Architecture** | MONAI 3D DenseNet201 |
| **Total Parameters** | ~20.2M |
| **Input Shape** | `(B, 1, 160, 160, 112)` |
| **Output** | 18-class logits (built-in head, `out_channels=18`) |
| **Pre-training** | None (random init from MONAI) |
| **Dropout** | `p=0.1` |

---

### Model C — 3D Swin-Transformer / SwinViT-3D (Current Training)
| Property | Value |
|----------|-------|
| **Architecture** | MONAI SwinUNETR encoder (SwinViT-3D) + custom classification head |
| **Total Parameters** | **153** (parameter groups) / ~62M weights |
| **Backbone (SwinViT)** | `feature_size=48` → 768-dim bottleneck |
| **Input Shape** | `(1, 1, 160, 160, 112)` (batch size forced to 1 for VRAM) |
| **Effective Batch Size** | **8** (via gradient accumulation over 8 steps) |
| **Classification Head** | `AdaptiveAvgPool3d(1)` → `Dropout(p=0.1)` → `Linear(768, 18)` |
| **Pre-training** | MONAI SSL pre-trained weights: `/scratch/25208443/pretrain_weights/model_swinvit.pt` |
| **Loaded SSL Layers** | **94 / 153** encoder layers loaded from pre-trained checkpoint |
| **Gradient Checkpointing** | **Disabled** (`use_checkpoint=False`) — enabled version silently blocks gradients under AMP |

---

## 5. Training Configurations (Per Model)

### Model A — ResNet-10
| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 8 |
| Learning Rate | `1e-3` |
| Optimizer | AdamW |
| Scheduler | None |
| Loss | Weighted Binary Cross-Entropy |
| Epochs | 50 |
| Dataset | 2,000 subset |
| Hardware | A100 40GB |

---

### Model B — DenseNet201
| Hyperparameter | Value |
|----------------|-------|
| Batch Size | **8** |
| Learning Rate | `5e-4` |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (`T_max=100`) |
| Loss | **Asymmetric Loss** (`γ_neg=4.0`, `γ_pos=0.0`, `clip=0.05`) |
| Epochs | 100 (cut at ~70 by Slurm) |
| Dataset | 11,404 training / 3,039 validation |
| Mixed Precision | AMP float16 |
| Hardware | A100 40GB |
| Post-processing | Per-class threshold sweep `[0.01, 0.99]` (99 steps) |

---

### Model C — SwinViT-3D (Current)
| Hyperparameter | Value |
|----------------|-------|
| Batch Size | **1** (VRAM constraint) |
| Effective Batch Size | **8** (gradient accumulation × 8 steps) |
| **Backbone LR** | `1e-5` (10× lower than head) |
| **Head LR** | `1e-4` |
| Optimizer | AdamW (`weight_decay=1e-4`) |
| **Scheduler** | CosineAnnealingLR (`T_max=100`, `eta_min=1e-7`) |
| Loss | **ASL + Label Smoothing** (`γ_neg=4.0`, `γ_pos=0.0`, `clip=0.05`, `ε=0.1`) |
| **Two-Stage Unfreezing** | Epochs 1–3: head only (2 params); Epoch 4+: full model (153 params) |
| **Class Imbalance** | `WeightedRandomSampler` (weight range: `[1.00, 12.93]`) |
| Hardware | A100 40-80GB |
| Mixed Precision | AMP float16 |
| Gradient Accumulation | 8 steps |
| Step Time (Stage 1) | ~0.07 sec/step |
| Step Time (Stage 2) | ~0.18 sec/step (backbone active) |
| Time per Epoch | ~48 min (training + validation) |

---

## 6. Results History

| Model | Dataset | Loss | Post-Processing | Val AUROC | Val F1 | Status |
|-------|---------|------|----------------|-----------|--------|--------|
| **3D ResNet-10** | 2,000 subset | Weighted BCE | Flat 0.5 threshold | 0.7245 | 0.4489 | ✅ Baseline established |
| **DenseNet201 Run 1** | 11,404 | Weighted BCE | Flat 0.5 threshold | 0.7812 | 0.4743 | ✅ Successful scaling |
| **DenseNet201 Run 3** | 11,404 | ASL | Flat 0.5 threshold | **0.7932** | 0.4843 | ✅ Stage 3 completed |
| **DenseNet201 (Optimized)** | 11,404 | ASL | **Per-class thresholds** | **0.7931** | **0.5076** | ✅ **Current Best** |
| **SwinViT-3D (Job 411160)** | 11,404 | ASL | Flat 0.5 | 0.5419 | 0.2166 | ❌ Gradient blocked (frozen backbone bug) |
| **SwinViT-3D (Job 416113)** | 11,404 | ASL + Label Smooth | Flat 0.5 | **0.6390** | 0.3346 | 🛑 Stopped at epoch 26 |
| **SwinViT-3D (Job 416941)** | 11,404 | ASL + Label Smooth | Per-class thresholds | **0.6379** | **0.3703** | ✅ Completed (backbone remained frozen; F1 +3.13% via threshold sweep) |

### SwinViT-3D Epoch-by-Epoch (Job 416941 - Resumed from Epoch 19)
- Resumed at Epoch 19 with backbone frozen (unfreezing logic bypassed).
- Epoch 19: Train Loss: 0.1792 | Val Loss: 0.1595 | Val AUROC: 0.6362 | Val F1: 0.3356
- Epoch 100: Train Loss: 0.1781 | Val Loss: 0.1596 | Val AUROC: 0.6370 | Val F1: 0.3344
- Threshold sweep on final checkpoint (Job 416940): Macro F1 improved to **0.3703** (macro AUROC 0.6379 with 8-Fold TTA)

---

## 7. Known Issues & Fixes Applied

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **Gradient blocking** (Job 411160) | `use_checkpoint=True` + AMP float16 = reentrant checkpointing blocks encoder gradients | `use_checkpoint=False` in SwinUNETR |
| **Learning rate plateau** (Job 416113, epoch 13–26) | Custom warmup scheduler not decaying LR after warmup | Replaced with `CosineAnnealingLR(T_max=100, eta_min=1e-7)` |
| **F1 frozen at ~0.33** | Hard 0.5 threshold for all classes regardless of class frequency | Per-class threshold sweep `[0.01, 0.99]` (same as DenseNet: +2.51% F1) |
| **Checkpoint crash on resume** | Old optimizer (1 param group) vs new optimizer (2 param groups) state mismatch | Staged checkpoint loading: model weights always load; optimizer state optional |
| **Black tensor preprocessing** | Static HU clipping clamped 99.9% of voxels to 0.0 | Dynamic percentile windowing (5%–95%) |
| **DataLoader crashes** | MONAI MetaTensor metadata incompatibility across hospitals | `NakedDataset` strips to raw `torch.Tensor` |
| **Shared memory overflow** | `/dev/shm` buffer exhausted on Slurm with `num_workers=8` | `mp.set_sharing_strategy('file_system')` |
| **Backbone frozen on resume** | `epoch == freeze_epochs` unfreeze condition was bypassed when resuming from epoch > 3 | Changed check to `epoch >= freeze_epochs` and check if backbone is frozen before unfreezing |

---

## 8. Current Status & Next Steps

**Training Status:**
- The training completed 100 epochs (Job 416941), but due to a checkpoint resume bug, the backbone remained frozen from epoch 19 onwards.
- We ran threshold optimization (Job 416940) on the resulting model, boosting Macro F1 by +3.13% (from 0.3390 to 0.3703).

**Next Steps:**
1. **Fix Applied:** Modified `train.py` to ensure that resuming from checkpoints will correctly unfreeze the backbone.
2. **Retrain options:**
   - **Option A (Recommended):** Resume from the Epoch 4 checkpoint (before the plateau) with the fixed unfreezing logic. This allows the Swin backbone to train with gradients for 96 epochs with proper cosine LR decay.
   - **Option B:** Re-run from scratch (epoch 0) with the new scheduler and unfreezing logic.
   - **Option C:** Resume from Epoch 19 checkpoint but with the backbone unfrozen (will only train the backbone for 81 epochs instead of 96).

**HPC Environment:**
- Cluster: UCD SONIC
- GPU: NVIDIA A100 (40–80 GB VRAM)
- Python: 3.12.11
- PyTorch: 2.6.0+cu124
- Virtual env: `/scratch/25208443/vlm3d_env/`
