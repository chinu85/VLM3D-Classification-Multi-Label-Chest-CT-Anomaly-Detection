import os
import torch
import numpy as np
import json
from model import ChestCTClassificationModel, ChestCTSwinClassificationModel
from data_loader import get_dataloaders
from sklearn.metrics import f1_score, roc_auc_score

def optimize_thresholds(valid_csv, valid_images_dir, model_path="best_model.pth", save_path="optimized_thresholds.json", use_swin=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load validation dataloader (batch size 8 or 4 for VRAM safety)
    # Pass None for train path to only load validation data
    _, val_loader, _ = get_dataloaders(None, valid_csv, None, valid_images_dir, batch_size=8)
    if val_loader is None:
        raise ValueError("Failed to initialize validation dataloader.")

    # 2. Load model
    print(f"Loading {'Swin-Transformer' if use_swin else 'DenseNet201'} model weights from '{model_path}'...", flush=True)
    if use_swin:
        model = ChestCTSwinClassificationModel(num_classes=18).to(device)
    else:
        model = ChestCTClassificationModel(num_classes=18).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Validation Inference with 8-Fold TTA
    print("Running inference on validation set (with 8-Fold TTA)...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            probs_sum = torch.zeros_like(labels, dtype=torch.float32)
            configurations = [
                ([], []),
                ([2],),
                ([3],),
                ([4],),
                ([2, 3],),
                ([2, 4],),
                ([3, 4],),
                ([2, 3, 4],)
            ]
            
            for config in configurations:
                if len(config) > 0:
                    inputs_aug = torch.flip(inputs, dims=config[0])
                else:
                    inputs_aug = inputs
                out = model(inputs_aug)
                probs_sum += torch.sigmoid(out).float()
                
            outputs_prob = probs_sum / len(configurations)
            
            all_preds.append(outputs_prob.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"Processed batch [{i+1}/{len(val_loader)}]", flush=True)

    preds = np.concatenate(all_preds, axis=0)  # Shape: (N, 18)
    labels = np.concatenate(all_labels, axis=0)  # Shape: (N, 18)

    # 4. Threshold Sweeping for each class
    print("\nOptimizing thresholds per class...", flush=True)
    optimized_thresholds = {}
    f1_before = []
    f1_after = []
    
    class_names = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly",
        "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
        "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
        "Mosaic attenuation pattern", "Peribronchial thickening", "Air trapping",
        "Bronchiectasis", "Pleural thickening"
    ]
    # Fallback to index if list length doesn't match class count
    if len(class_names) != preds.shape[1]:
        class_names = [f"Class_{i}" for i in range(preds.shape[1])]


    for class_idx in range(preds.shape[1]):
        y_true = labels[:, class_idx]
        y_prob = preds[:, class_idx]

        # F1 score with default 0.5 threshold
        y_pred_default = (y_prob >= 0.5).astype(int)
        default_f1 = f1_score(y_true, y_pred_default, zero_division=0)
        f1_before.append(default_f1)

        # Sweep thresholds
        best_threshold = 0.5
        best_f1 = default_f1
        
        # Test 99 thresholds from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)
        for t in thresholds:
            y_pred_temp = (y_prob >= t).astype(int)
            temp_f1 = f1_score(y_true, y_pred_temp, zero_division=0)
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = t

        f1_after.append(best_f1)
        optimized_thresholds[class_names[class_idx]] = {
            "threshold": float(best_threshold),
            "default_f1": float(default_f1),
            "optimized_f1": float(best_f1)
        }
        
        print(f"Class {class_idx+1:02d} ({class_names[class_idx]}): Default F1 = {default_f1:.4f} | Opt Threshold = {best_threshold:.2f} | Opt F1 = {best_f1:.4f}", flush=True)

    macro_f1_default = np.mean(f1_before)
    macro_f1_opt = np.mean(f1_after)
    macro_auroc = roc_auc_score(labels, preds, average="macro")

    print("\n" + "="*50, flush=True)
    print(f"Overall Validation Macro AUROC: {macro_auroc:.4f}", flush=True)
    print(f"Overall Validation Macro F1 (Rigid 0.5): {macro_f1_default:.4f}", flush=True)
    print(f"Overall Validation Macro F1 (Optimized): {macro_f1_opt:.4f}", flush=True)
    print(f"Absolute F1 improvement: {macro_f1_opt - macro_f1_default:+.4f}", flush=True)
    print("="*50, flush=True)

    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump({
            "macro_auroc": float(macro_auroc),
            "macro_f1_rigid_0.5": float(macro_f1_default),
            "macro_f1_optimized": float(macro_f1_opt),
            "class_metrics": optimized_thresholds
        }, json_file, indent=4)
        
    print(f"Saved optimized thresholds details to: {os.path.abspath(save_path)}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize class-specific F1 thresholds.")
    parser.add_argument("--use_swin", action="store_true", help="Use SOTA 3D Swin-Transformer model backbone instead of legacy DenseNet201.")
    args = parser.parse_args()

    # HPC validation paths
    valid_csv_path = "/scratch/25208443/dataset/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    valid_images_dir = "/scratch/25208443/data_volumes/dataset/valid/"
    
    if args.use_swin:
        model_path = "best_model_swin.pth"
        save_path = "optimized_thresholds_swin.json"
    else:
        model_path = "best_model.pth"
        save_path = "optimized_thresholds.json"
        
    optimize_thresholds(
        valid_csv=valid_csv_path, 
        valid_images_dir=valid_images_dir,
        model_path=model_path,
        save_path=save_path,
        use_swin=args.use_swin
    )
