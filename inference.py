#!/usr/bin/env python3
"""
VLM3D Challenge -- Inference Script
Reads NIfTI volumes from /input/, runs DenseNet201 + SwinViT-3D ensemble
with 8-Fold TTA, applies per-class optimized thresholds, and writes
18-length binary prediction vectors to /output/<scan_id>.json
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import torch.nn as nn
from monai.networks.nets import DenseNet201, SwinUNETR
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRangePercentiles, CropForeground, SpatialPad,
    CenterSpatialCrop, EnsureType,
)

CLASS_NAMES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Air trapping",
    "Bronchiectasis", "Pleural thickening",
]


# -- Model definitions (must match training exactly) --------------------------

class ChestCTClassificationModel(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.backbone = DenseNet201(
            spatial_dims=3, in_channels=1,
            out_channels=num_classes, dropout_prob=0.1,
        )

    def forward(self, x):
        return self.backbone(x)


class ChestCTSwinClassificationModel(nn.Module):
    def __init__(self, num_classes=18, feature_size=48):
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=1, out_channels=num_classes,
            feature_size=feature_size, use_checkpoint=False,
        )
        self.pool    = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc      = nn.Linear(feature_size * 16, num_classes)

    def forward(self, x):
        features   = self.backbone.swinViT(x)
        bottleneck = features[4]
        pooled     = self.pool(bottleneck).squeeze(-1).squeeze(-1).squeeze(-1)
        return self.fc(self.dropout(pooled))


# -- Preprocessing (identical to validation transforms in data_loader.py) -----

def get_inference_transform():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        ScaleIntensityRangePercentiles(
            lower=5.0, upper=95.0, b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForeground(),
        SpatialPad(spatial_size=(160, 160, 112)),
        CenterSpatialCrop(roi_size=(160, 160, 112)),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])


# -- 8-Fold TTA ---------------------------------------------------------------
# Axes refer to SPATIAL dimensions (D=0, H=1, W=2).
# After unsqueeze(0) the tensor is (1, 1, D, H, W), so we add 2 to get the
# actual dim index:  axis 0 -> dim 2 (D), axis 1 -> dim 3 (H), axis 2 -> dim 4 (W)
# BUG FIX: previous version used [1],[2],[3] which mapped to dims [3],[4],[5] --
#           dim 5 does not exist on a 5-D tensor and causes an IndexError crash.
TTA_CONFIGS = [
    [],           # original
    [0],          # flip D  -> torch dim 2
    [1],          # flip H  -> torch dim 3
    [2],          # flip W  -> torch dim 4
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 1, 2],
]


@torch.no_grad()
def tta_inference(model, volume_tensor, device):
    """
    8-Fold TTA on a single volume.
    volume_tensor: (1, D, H, W) tensor (channel-first, no batch dim).
    Returns numpy array (18,) of averaged sigmoid probabilities.
    """
    model.eval()
    batch = volume_tensor.unsqueeze(0).to(device)   # (1, 1, D, H, W)

    probs_sum = torch.zeros(18, device=device)
    for axes in TTA_CONFIGS:
        # axes are spatial (0=D,1=H,2=W); add 2 to get correct tensor dim index
        inp = torch.flip(batch, dims=[a + 2 for a in axes]) if axes else batch
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(inp)                     # (1, 18)
        probs_sum += torch.sigmoid(logits.float()).squeeze(0)

    return (probs_sum / len(TTA_CONFIGS)).cpu().numpy()


# -- Main inference loop ------------------------------------------------------

def run_inference(
    input_dir="/input",
    output_dir="/output",
    densenet_path="/models/best_model.pth",
    swin_path="/models/best_model_swin_suprem.pth",
    thresholds_path="/models/optimized_thresholds_ensemble_3model.json",
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}", flush=True)

    # -- Load thresholds and ensemble weights from JSON -----------------------
    print(f"[inference] Loading thresholds from {thresholds_path}", flush=True)
    with open(thresholds_path) as f:
        thresh_data = json.load(f)

    # Per-class decision thresholds (ordered by CLASS_NAMES)
    thresholds = np.array([
        thresh_data["class_metrics"][c]["threshold"] for c in CLASS_NAMES
    ], dtype=np.float32)

    # Ensemble weights -- read from JSON (Nelder-Mead optimised: 0.549 / 0.451)
    # Falls back to 0.5/0.5 if key is missing for backwards compatibility.
    weights = thresh_data.get("weights", {"densenet201": 0.5, "swin": 0.5})
    w_dn = float(weights.get("densenet201", 0.5))
    w_sw = float(weights.get("swin", 0.5))
    # Normalise in case they don't sum to 1 (e.g. 3-model JSON with swin2=0)
    total = w_dn + w_sw
    w_dn /= total
    w_sw /= total
    print(f"[inference] Ensemble weights: DenseNet={w_dn:.4f}, Swin={w_sw:.4f}", flush=True)
    print(f"[inference] Thresholds: {thresholds.tolist()}", flush=True)

def load_clean_state_dict(model, path, device):
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


    # -- Load models ----------------------------------------------------------
    print(f"[inference] Loading DenseNet201 from {densenet_path}", flush=True)
    densenet = ChestCTClassificationModel(num_classes=18).to(device)
    load_clean_state_dict(densenet, densenet_path, device)
    densenet.eval()

    print(f"[inference] Loading SwinViT-3D from {swin_path}", flush=True)
    swin = ChestCTSwinClassificationModel(num_classes=18).to(device)
    load_clean_state_dict(swin, swin_path, device)
    swin.eval()

    # -- Preprocessing --------------------------------------------------------
    transform = get_inference_transform()

    # -- Discover NIfTI files -------------------------------------------------
    nifti_files = sorted(set(
        p for pattern in [
            os.path.join(input_dir, "*.nii.gz"),
            os.path.join(input_dir, "*.nii"),
            os.path.join(input_dir, "**", "*.nii.gz"),
            os.path.join(input_dir, "**", "*.nii"),
        ]
        for p in glob.glob(pattern, recursive=True)
    ))

    if not nifti_files:
        print(f"[inference] WARNING: No NIfTI files found in {input_dir}", flush=True)
        sys.exit(0)

    print(f"[inference] Found {len(nifti_files)} volume(s).", flush=True)

    # -- Process each scan ----------------------------------------------------
    for nifti_path in nifti_files:
        scan_id = (os.path.basename(nifti_path)
                   .replace(".nii.gz", "").replace(".nii", ""))
        print(f"[inference] Processing: {scan_id}", flush=True)

        try:
            volume = transform(nifti_path)              # (1, D, H, W)

            probs_dn = tta_inference(densenet, volume, device)
            probs_sw = tta_inference(swin,     volume, device)

            probs  = w_dn * probs_dn + w_sw * probs_sw
            binary = [int(probs[i] >= thresholds[i]) for i in range(18)]

            result = {
                "scan_id":            scan_id,
                "binary_prediction":  binary,
                "class_probabilities": {
                    c: float(probs[i]) for i, c in enumerate(CLASS_NAMES)
                },
                "thresholds_used": {
                    c: float(thresholds[i]) for i, c in enumerate(CLASS_NAMES)
                },
            }

            out_path = os.path.join(output_dir, f"{scan_id}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"[inference]   binary: {binary}", flush=True)
            print(f"[inference]   saved:  {out_path}", flush=True)

        except Exception as e:
            print(f"[inference] ERROR on {scan_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"[inference] Done. {len(nifti_files)} volume(s) processed.", flush=True)


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM3D Ensemble Inference")
    parser.add_argument("--input_dir",       default="/input")
    parser.add_argument("--output_dir",      default="/output")
    parser.add_argument("--densenet_path",   default="/models/best_model.pth")
    parser.add_argument("--swin_path",       default="/models/best_model_swin_suprem.pth")
    parser.add_argument("--thresholds_path", default="/models/optimized_thresholds_ensemble_3model.json")
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        densenet_path=args.densenet_path,
        swin_path=args.swin_path,
        thresholds_path=args.thresholds_path,
    )
