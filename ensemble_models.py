"""
ensemble_models.py — VLM3D Ensemble Evaluation & Threshold Optimisation

Runs both models over the full validation set with correct 8-fold TTA, then:
  1. Optionally optimises ensemble weights via Nelder-Mead (--optimize_weights flag)
  2. Per-class threshold sweep to maximise macro F1
  3. Saves everything to a JSON file that inference.py reads directly

Usage (SONIC HPC):
    python -u ensemble_models.py \
        --densenet_path best_model.pth \
        --swin_path best_model_swin_suprem.pth \
        --save_path optimized_thresholds_ensemble_suprem.json \
        --optimize_weights
"""

import os
import torch
import numpy as np
import json
from scipy.optimize import minimize
from model import ChestCTClassificationModel, ChestCTSwinClassificationModel
from data_loader import get_dataloaders
from sklearn.metrics import f1_score, roc_auc_score

CLASS_NAMES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]

# 8-fold TTA: dim 2=D, 3=H, 4=W in a (B,1,D,H,W) batch tensor.
# Empty list = original (no flip) — this is the identity pass that was missing before.
TTA_FLIP_DIMS = [
    [],           # original — no flip
    [2],          # flip D
    [3],          # flip H
    [4],          # flip W
    [2, 3],
    [2, 4],
    [3, 4],
    [2, 3, 4],
]


def load_clean_state_dict(model, path, device):
    """
    Robustly loads a checkpoint handling torch.compile ('_orig_mod.')
    and DataParallel ('module.') prefixes.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned_state[k[len("_orig_mod."):]] = v
        elif k.startswith("module."):
            cleaned_state[k[len("module."):]] = v
        else:
            cleaned_state[k] = v

    model.load_state_dict(cleaned_state)
    return model


@torch.no_grad()
def run_tta_inference(model, val_loader, device):
    """
    Run full validation set inference with 8-fold flip TTA.
    Returns:
        preds  — np.ndarray (N, 18) of averaged sigmoid probabilities
        labels — np.ndarray (N, 18) of ground-truth binary labels
    """
    model.eval()
    all_preds = []
    all_labels = []

    for i, batch in enumerate(val_loader):
        inputs = batch["image"].to(device)   # (B, 1, D, H, W)
        labels = batch["label"]              # (B, 18)

        probs_sum = torch.zeros(inputs.shape[0], 18, device=device)

        for flip_dims in TTA_FLIP_DIMS:
            inp = torch.flip(inputs, dims=flip_dims) if flip_dims else inputs
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(inp)          # (B, 18)
            probs_sum += torch.sigmoid(logits.float())

        avg_probs = probs_sum / len(TTA_FLIP_DIMS)  # (B, 18)
        all_preds.append(avg_probs.cpu().numpy())
        all_labels.append(labels.numpy())

        if (i + 1) % 10 == 0:
            print(f"  Processed batch [{i+1}/{len(val_loader)}]", flush=True)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


def optimise_weights_nelder_mead(densenet_preds, swin_preds, labels):
    """
    Find optimal ensemble weights (w_dn, w_sw) that maximise macro AUROC.
    Weights are constrained to be non-negative and sum to 1.
    Returns (w_dn, w_sw).
    """
    print("\nOptimising ensemble weights via Nelder-Mead (maximising macro AUROC)...", flush=True)

    def neg_auroc(w):
        w_dn = np.clip(w[0], 0.0, 1.0)
        w_sw = 1.0 - w_dn
        preds = w_dn * densenet_preds + w_sw * swin_preds
        try:
            return -roc_auc_score(labels, preds, average="macro")
        except Exception:
            return 0.0

    result = minimize(neg_auroc, x0=[0.5], method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 500})

    w_dn = float(np.clip(result.x[0], 0.0, 1.0))
    w_sw = 1.0 - w_dn
    print(f"  Optimal weights: densenet201={w_dn:.4f}, swin={w_sw:.4f}", flush=True)
    return w_dn, w_sw


def optimise_thresholds(preds, labels):
    """
    Per-class threshold sweep [0.01, 0.99] to maximise F1 for each class.
    Returns:
        thresholds   — dict {class_name: {threshold, default_f1, optimized_f1}}
        f1_default   — macro F1 at flat 0.5
        f1_optimised — macro F1 at per-class thresholds
    """
    thresholds_out = {}
    f1_before = []
    f1_after = []
    sweep = np.linspace(0.01, 0.99, 99)

    print("\nOptimising thresholds per class...", flush=True)
    for idx, name in enumerate(CLASS_NAMES):
        y_true = labels[:, idx]
        y_prob = preds[:, idx]

        default_f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        f1_before.append(default_f1)

        best_t, best_f1 = 0.5, default_f1
        for t in sweep:
            tf1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            if tf1 > best_f1:
                best_f1, best_t = tf1, t
        f1_after.append(best_f1)

        thresholds_out[name] = {
            "threshold":     float(best_t),
            "default_f1":    float(default_f1),
            "optimized_f1":  float(best_f1),
        }
        print(f"  Class {idx+1:02d} ({name}): "
              f"Default F1 = {default_f1:.4f} | "
              f"Opt threshold = {best_t:.2f} | "
              f"Opt F1 = {best_f1:.4f}", flush=True)

    return thresholds_out, float(np.mean(f1_before)), float(np.mean(f1_after))


def ensemble_evaluation(
    valid_csv,
    valid_images_dir,
    densenet_path="best_model.pth",
    swin_path="best_model_swin_suprem.pth",
    save_path="optimized_thresholds_ensemble_suprem.json",
    weight_densenet=0.5,
    weight_swin=0.5,
    optimize_weights=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Validation dataloader — batch 4 for VRAM safety (Swin is large)
    _, val_loader, _ = get_dataloaders(None, valid_csv, None, valid_images_dir, batch_size=4)
    if val_loader is None:
        raise ValueError("Failed to initialise validation dataloader.")

    # Load DenseNet201
    print(f"\nLoading DenseNet201 from '{densenet_path}'...", flush=True)
    densenet_model = ChestCTClassificationModel(num_classes=18).to(device)
    load_clean_state_dict(densenet_model, densenet_path, device)
    densenet_model.eval()
    print("  Running inference...", flush=True)
    densenet_preds, labels = run_tta_inference(densenet_model, val_loader, device)

    # Load SwinViT-3D (SuPreM)
    print(f"\nLoading SwinViT from '{swin_path}'...", flush=True)
    swin_model = ChestCTSwinClassificationModel(num_classes=18).to(device)
    load_clean_state_dict(swin_model, swin_path, device)
    swin_model.eval()
    print("  Running inference...", flush=True)
    swin_preds, _ = run_tta_inference(swin_model, val_loader, device)

    # Optimise or use provided ensemble weights
    if optimize_weights:
        w_dn, w_sw = optimise_weights_nelder_mead(densenet_preds, swin_preds, labels)
    else:
        total = weight_densenet + weight_swin
        w_dn = weight_densenet / total
        w_sw = weight_swin / total
        print(f"\nUsing fixed weights: densenet201={w_dn:.4f}, swin={w_sw:.4f}", flush=True)

    # Ensemble probabilities
    preds = w_dn * densenet_preds + w_sw * swin_preds

    # Per-model solo AUROC (diagnostic)
    auroc_dn = roc_auc_score(labels, densenet_preds, average="macro")
    auroc_sw = roc_auc_score(labels, swin_preds, average="macro")
    auroc_ens = roc_auc_score(labels, preds, average="macro")

    print(f"\nEnsemble (flat 0.5 threshold): AUROC {auroc_ens:.4f} | "
          f"F1 {float(np.mean([f1_score(labels[:, i], (preds[:, i] >= 0.5).astype(int), zero_division=0) for i in range(18)])):.4f}",
          flush=True)
    print(f"  (DenseNet solo AUROC: {auroc_dn:.4f} | Swin solo AUROC: {auroc_sw:.4f})", flush=True)

    # Per-class threshold optimisation
    class_thresholds, f1_default, f1_opt = optimise_thresholds(preds, labels)

    print("\n" + "=" * 60, flush=True)
    print(f"Ensemble Macro AUROC              : {auroc_ens:.4f}", flush=True)
    print(f"Ensemble Macro F1 (flat 0.5)      : {f1_default:.4f}", flush=True)
    print(f"Ensemble Macro F1 (optimised thr) : {f1_opt:.4f}", flush=True)
    print(f"Absolute F1 improvement           : {f1_opt - f1_default:+.4f}", flush=True)
    print("=" * 60, flush=True)

    # Save JSON — inference.py reads weights and class_metrics from this file
    output = {
        "macro_auroc":            float(auroc_ens),
        "densenet_solo_auroc":    float(auroc_dn),
        "swin_solo_auroc":        float(auroc_sw),
        "macro_f1_flat_0.5":      float(f1_default),
        "macro_f1_optimised":     float(f1_opt),
        "weights": {
            "densenet201": float(w_dn),
            "swin":        float(w_sw),
        },
        "class_metrics": class_thresholds,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved to: {os.path.abspath(save_path)}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble DenseNet201 + SwinViT-3D with 8-fold TTA and threshold optimisation."
    )
    parser.add_argument("--densenet_path",   type=str, default="best_model.pth")
    parser.add_argument("--swin_path",       type=str, default="best_model_swin_suprem.pth")
    parser.add_argument("--save_path",       type=str, default="optimized_thresholds_ensemble_suprem.json")
    parser.add_argument("--weight_densenet", type=float, default=0.5,
                        help="Fixed DenseNet weight (ignored if --optimize_weights)")
    parser.add_argument("--weight_swin",     type=float, default=0.5,
                        help="Fixed Swin weight (ignored if --optimize_weights)")
    parser.add_argument("--optimize_weights", action="store_true", default=False,
                        help="Run Nelder-Mead to find optimal ensemble weights")
    args = parser.parse_args()

    valid_csv_path    = "/scratch/25208443/dataset/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    valid_images_dir  = "/scratch/25208443/data_volumes/dataset/valid/"

    ensemble_evaluation(
        valid_csv=valid_csv_path,
        valid_images_dir=valid_images_dir,
        densenet_path=args.densenet_path,
        swin_path=args.swin_path,
        save_path=args.save_path,
        weight_densenet=args.weight_densenet,
        weight_swin=args.weight_swin,
        optimize_weights=args.optimize_weights,
    )
