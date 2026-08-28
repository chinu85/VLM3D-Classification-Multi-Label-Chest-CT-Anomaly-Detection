import os
import torch
import torch.nn as nn
import numpy as np
from model import ChestCTClassificationModel, ChestCTSwinClassificationModel
from data_loader import get_dataloaders
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

CLASS_NAMES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening"
]

class AsymmetricLossWithLogits(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    Separately tunes positive/negative focusing with optional label smoothing
    to prevent overconfidence on majority classes.

    Supports per-class gamma_neg via `class_gamma_neg` tensor (shape [num_classes]).
    When provided, `gamma_neg` acts as the default fallback for any class not overridden.
    Weak classes (e.g. Lung nodule, Fibrotic sequela, Bronchiectasis) can receive a
    lower gamma_neg to soften the hard-negative suppression that hurts rare positives.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8,
                 label_smoothing=0.1, class_gamma_neg=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.label_smoothing = label_smoothing
        # Optional per-class gamma_neg tensor: shape (num_classes,)
        # Registered as a buffer so it moves to the correct device with .to(device)
        if class_gamma_neg is not None:
            self.register_buffer("class_gamma_neg", class_gamma_neg.float())
        else:
            self.class_gamma_neg = None

    def forward(self, inputs, targets):
        # Label smoothing: soft targets instead of hard 0/1
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        xs_pos = torch.sigmoid(inputs)
        xs_neg = 1.0 - xs_pos

        # Asymmetric clipping for easy negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Positive loss
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        if self.gamma_pos > 0:
            loss_pos *= (1.0 - xs_pos) ** self.gamma_pos

        # Negative loss — use per-class gamma_neg if available, else scalar
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.class_gamma_neg is not None:
            # class_gamma_neg: (C,)  ->  broadcast to (B, C)
            gamma = self.class_gamma_neg.unsqueeze(0)   # (1, C)
            loss_neg = loss_neg * (1.0 - xs_neg) ** gamma
        elif self.gamma_neg > 0:
            loss_neg *= (1.0 - xs_neg) ** self.gamma_neg

        # Combined multi-label loss
        loss = - (loss_pos + loss_neg)
        return loss.mean()

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Linear warmup for `warmup_epochs` then cosine annealing to 0.
    Critical for Transformers — prevents destroying pretrained features in early epochs.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_ssl_weights(model, ssl_weights_path, device):
    """
    Helper function to load local SOTA Self-Supervised Medical Pre-trained weights into
    the Swin-Transformer (SwinViT-3D) backbone. Gracefully falls back to standard training if blocked or unavailable.
    """
    if not ssl_weights_path or not os.path.exists(ssl_weights_path):
        print("=> No local SSL pre-trained weights path specified or found. Training Swin from scratch.", flush=True)
        return
        
    print(f"=> Loading SOTA Swin SSL weights from '{ssl_weights_path}'...", flush=True)
    try:
        weights = torch.load(ssl_weights_path, map_location=device, weights_only=False)
        state_dict = weights.get("state_dict", weights.get("net", weights))
            
        model_dict = model.state_dict()
        load_dict = {}
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
                
            # Map SwinUNETR's full checkpoint keys to our model's nested backbone.swinViT
            if clean_k.startswith("swinViT."):
                mapped_k = "backbone." + clean_k
                if mapped_k in model_dict and model_dict[mapped_k].shape == v.shape:
                    load_dict[mapped_k] = v
            else:
                mapped_k = "backbone.swinViT." + clean_k
                if mapped_k in model_dict and model_dict[mapped_k].shape == v.shape:
                    load_dict[mapped_k] = v
                    
        if len(load_dict) > 0:
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)
            print(f"=> Successfully loaded {len(load_dict)} SOTA Swin-Transformer encoder pre-trained layers!", flush=True)
        else:
            print("Warning: Pre-trained weight keys did not match model architecture. Training Swin from scratch.", flush=True)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights ({e}). Training Swin from scratch.", flush=True)

def train_and_evaluate(train_csv, valid_csv, train_images_dir, valid_images_dir, num_epochs=100, batch_size=8, lr=5e-4, log_dir="runs/ct_classifier", use_swin=True, ssl_weights_path=None, warmup_epochs=5, freeze_epochs=3, checkpoint_path=None, best_model_path=None, reset_scheduler=False, weak_class_gamma=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Auto-adjust batch size and accumulation steps for Swin VRAM safety
    accumulation_steps = 1
    if use_swin:
        print(f"=> Swin-Transformer mode enabled. Auto-adjusting batch size from {batch_size} to 1, and setting gradient accumulation steps to 8 for memory safety (effective batch size = 8).", flush=True)
        batch_size = 1
        accumulation_steps = 8
    
    # Dataloaders - Stage 3 high-res spacing & crop configured in data_loader
    train_loader, val_loader, train_data = get_dataloaders(train_csv, valid_csv, train_images_dir, valid_images_dir, batch_size=batch_size)
    if train_loader is None or val_loader is None:
        raise ValueError("Invalid dataloaders generated. Check CSV and image paths.")

    # Model Selection
    if use_swin:
        model = ChestCTSwinClassificationModel(num_classes=18).to(device)
        print("Initialized SOTA 3D Swin-Transformer backbone (SwinViT-3D).", flush=True)
        
        # Load self-supervised pre-trained weights if available
        load_ssl_weights(model, ssl_weights_path, device)

        # --- Two-Stage Unfreezing: Freeze backbone for first `freeze_epochs` epochs ---
        # Stage 1: Train only the classification head (fc + dropout) to warm it up
        # Stage 2: Unfreeze everything with differential LR
        print(f"=> Two-stage unfreezing: Backbone frozen for first {freeze_epochs} epochs.", flush=True)
        for name, param in model.named_parameters():
            if 'fc' not in name and 'dropout' not in name:
                param.requires_grad_(False)
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"=> Stage 1: {frozen_count} backbone params frozen. Training head only.", flush=True)
    else:
        model = ChestCTClassificationModel(num_classes=18).to(device)
        print("Initialized legacy DenseNet201 CNN backbone.", flush=True)

    # --- Differential LR: backbone gets 10x lower LR than the classification head ---
    if use_swin:
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and 'dropout' not in n]
        head_params = [p for n, p in model.named_parameters() if 'fc' in n or 'dropout' in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},   # 1e-5 for backbone
            {'params': head_params,     'lr': lr}           # 1e-4 for head
        ], weight_decay=1e-4)
        print(f"=> Differential LR: backbone={lr*0.1:.0e}, head={lr:.0e}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # CosineAnnealingLR: decays LR from initial value to eta_min over T_max epochs.
    # This prevents the LR plateau issue observed at epoch 18+ with flat warmup-only scheduling.
    # backbone LR: 1e-5 → 1e-7 | head LR: 1e-4 → 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )
    print(f"=> CosineAnnealingLR scheduler: T_max={num_epochs}, eta_min=1e-7", flush=True)

    # --- Per-class ASL gamma_neg overrides for weak classes ---
    # Default: gamma_neg=4.0 global (strong hard-negative suppression)
    # Weak classes (Lung nodule=9, Pulmonary fibrotic sequela=11, Bronchiectasis=16)
    # get a lower gamma_neg so their rare positives aren't drowned out by easy negatives.
    class_gamma_neg_tensor = None
    if weak_class_gamma is not None and use_swin:
        class_gamma_neg_tensor = torch.ones(18) * 4.0  # default for all classes
        weak_indices = [9, 11, 16]  # Lung nodule, Fibrotic sequela, Bronchiectasis
        class_gamma_neg_tensor[weak_indices] = weak_class_gamma
        print(f"=> Per-class ASL: global gamma_neg=4.0, weak classes {weak_indices} -> gamma_neg={weak_class_gamma}", flush=True)

    # Upgraded Loss: Asymmetric Loss (ASL) + Label Smoothing
    criterion = AsymmetricLossWithLogits(
        gamma_neg=4.0, gamma_pos=0.0, clip=0.05, label_smoothing=0.1,
        class_gamma_neg=class_gamma_neg_tensor,
    ).to(device)
    
    # Metrics — macro average + per-class for detailed diagnostics
    auroc = MultilabelAUROC(num_labels=18, average="macro").to(device)
    f1_score = MultilabelF1Score(num_labels=18, average="macro").to(device)
    auroc_per_class = MultilabelAUROC(num_labels=18, average=None).to(device)
    f1_per_class = MultilabelF1Score(num_labels=18, average=None).to(device)
    
    start_epoch = 0
    best_val_score = 0.0
    
    # Separate checkpoint paths to prevent model parameter collisions
    if checkpoint_path is None:
        checkpoint_path = "best_model_checkpoint_swin.pth" if use_swin else "best_model_checkpoint.pth"
    if best_model_path is None:
        best_model_path = "best_model_swin.pth" if use_swin else "best_model.pth"
    
    # Checkpoint Loading: Seamlessly resume Slurm jobs if interrupted
    # Uses a staged approach: model weights are loaded first (always safe),
    # then optimizer/scheduler are loaded separately and skipped if incompatible
    # (e.g. after switching from 1-group to 2-group optimizer).
    if os.path.exists(checkpoint_path):
        print(f"=> Loading training checkpoint '{checkpoint_path}'...", flush=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Stage 1: Always load model weights — this is the most important part
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_score = checkpoint.get('best_val_score', 0.0)
            print(f"=> Model weights loaded. Resuming from Epoch {start_epoch + 1}", flush=True)

            # Stage 2: Try to restore optimizer and scheduler state (optional — may fail if
            # optimizer structure changed e.g. 1 param group → 2 param groups).
            # Pass --reset_scheduler to skip scheduler restore for a cosine warm-restart.
            if reset_scheduler:
                print(f"=> --reset_scheduler active: loading optimizer state only; cosine schedule restarted from T_max.", flush=True)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"=> Optimizer state restored. LR schedule restarting fresh.", flush=True)
                except Exception as opt_e:
                    print(f"=> Note: Optimizer state not restored ({opt_e}). Starting fresh.", flush=True)
            else:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"=> Optimizer and scheduler state restored successfully.", flush=True)
                except Exception as opt_e:
                    print(f"=> Note: Optimizer/scheduler state not restored ({opt_e}).", flush=True)
                    print(f"=> Model weights ARE loaded. Optimizer restarting fresh (expected after architecture changes).", flush=True)

        except Exception as e:
            print(f"Warning: Failed to load checkpoint entirely ({e}). Starting from scratch.", flush=True)
            
    # Speed Optimization: Mixed precision and cuDNN auto-benchmarking
    scaler = torch.amp.GradScaler('cuda')
    torch.backends.cudnn.benchmark = True

    # torch.compile: fuses ops and reduces kernel launch overhead on A100 (~10% speedup).
    # Applied AFTER checkpoint loading so compiled graph uses restored weights.
    # IMPORTANT: Must use mode='default' — NOT 'reduce-overhead' or 'max-autotune'.
    # Those modes use CUDA Graphs, which crash with MONAI's PatchEmbedding (in-place proj
    # overwrites a captured tensor between graph replays → RuntimeError: CUDAGraphs overwritten).
    # 'default' mode does op fusion + kernel optimisation without CUDA Graphs — safe with MONAI.
    # Falls back gracefully if PyTorch version < 2.0.
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='default')
            print("=> torch.compile enabled (default mode — CUDA-Graph-free, MONAI-safe). First epoch will be slower due to compilation.", flush=True)
        except Exception as compile_e:
            print(f"=> torch.compile skipped ({compile_e}).", flush=True)
    
    # VRAM Safety: Gradient accumulation is dynamically set at the beginning of train_and_evaluate
    
    print("Starting training loop...", flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        # --- Two-Stage Unfreezing: Unfreeze backbone after freeze_epochs ---
        if use_swin and epoch >= freeze_epochs:
            # Check if any backbone parameter is still frozen; if so, unfreeze them all
            backbone_frozen = any(not p.requires_grad for n, p in model.named_parameters() if 'fc' not in n and 'dropout' not in n)
            if backbone_frozen:
                print(f"\n=> Stage 2: Unfreezing full Swin backbone at epoch {epoch+1}!", flush=True)
                for param in model.parameters():
                    param.requires_grad_(True)
                alive = sum(1 for p in model.parameters() if p.requires_grad)
                print(f"=> All {alive} params now trainable with differential LR.", flush=True)

        model.train()
        train_loss = 0.0
        
        print(f"--- Starting Epoch {epoch+1} ---", flush=True)
        import time
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            step_start = time.time()
            
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass in 16-bit Mixed Precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss = loss / accumulation_steps
                
            # Backward pass scaled to prevent underflow
            scaler.scale(loss).backward()
            
            # Perform optimizer step at accumulation intervals
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Accurate gradient check before optimizer.zero_grad()
                if epoch == start_epoch and step == (accumulation_steps - 1):
                    frozen = []
                    alive = []
                    for name, param in model.named_parameters():
                        if param.grad is None or param.grad.abs().sum() == 0:
                            frozen.append(name)
                        else:
                            alive.append(name)
                    print(f"==> GRADIENT CHECK (at step {step}): {len(alive)} params with gradients, {len(frozen)} frozen params", flush=True)
                    if frozen and use_swin and epoch < freeze_epochs:
                        # Intentional: backbone is frozen in Stage 1 by design
                        print(f"==> INFO: {len(frozen)} backbone params intentionally frozen (Stage 1 of 2). "
                              f"Will unfreeze at epoch {freeze_epochs + 1}. This is correct.", flush=True)
                    elif frozen:
                        # Unexpected: should not happen after Stage 2 unfreezing
                        print(f"==> WARNING: Unexpected frozen params detected: {frozen[:10]}{'...' if len(frozen) > 10 else ''}", flush=True)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            # Log step loss
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Train/Loss_step", loss.item() * accumulation_steps, global_step)
            
            step_time = time.time() - step_start
            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item() * accumulation_steps:.4f} ({step_time:.2f} sec/batch)", flush=True)
                
        train_loss /= len(train_loader)
        writer.add_scalar("Train/Loss_epoch", train_loss, epoch)

        # Validation — run every val_freq epochs to save GPU time (~5-6 min per validation pass).
        # scheduler.step() always runs to keep LR cosine curve correct regardless of validation.
        val_freq = 2
        run_val = ((epoch + 1) % val_freq == 0) or (epoch == start_epoch) or (epoch + 1 == num_epochs)

        if run_val:
            model.eval()
            val_loss = 0.0
            auroc.reset()
            f1_score.reset()
            auroc_per_class.reset()
            f1_per_class.reset()
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Standard validation forward pass (TTA is disabled during training to save 24+ hours of GPU time)
                    out = model(inputs)
                    outputs_prob = torch.sigmoid(out)
                    
                    loss = criterion(out, labels.float())
                    val_loss += loss.item()
                    
                    auroc.update(outputs_prob, labels.long())
                    f1_score.update(outputs_prob, labels.long())
                    auroc_per_class.update(outputs_prob, labels.long())
                    f1_per_class.update(outputs_prob, labels.long())
                    
            val_loss /= len(val_loader)
            val_auroc = auroc.compute().item()
            val_f1 = f1_score.compute().item()
            per_class_auroc = auroc_per_class.compute().cpu().numpy()
            per_class_f1 = f1_per_class.compute().cpu().numpy()

        scheduler.step()
        
        # Log macro validation metrics (always log — use last known values on skipped epochs)
        writer.add_scalar("Val/Loss_epoch", val_loss, epoch)
        writer.add_scalar("Val/AUROC_macro", val_auroc, epoch)
        writer.add_scalar("Val/F1_macro", val_f1, epoch)
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], epoch)

        if run_val:
            # Log per-class metrics to TensorBoard
            for i, cname in enumerate(CLASS_NAMES):
                writer.add_scalar(f"Val/AUROC_{cname.replace(' ', '_')}", per_class_auroc[i], epoch)
                writer.add_scalar(f"Val/F1_{cname.replace(' ', '_')}", per_class_f1[i], epoch)

            # Print per-class table every 10 epochs (was 5 — halved since val is now every 2 epochs)
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                print(f"\n{'Class':<40} {'AUROC':>7} {'F1':>7}", flush=True)
                print("-" * 56, flush=True)
                for i, cname in enumerate(CLASS_NAMES):
                    print(f"{cname:<40} {per_class_auroc[i]:>7.4f} {per_class_f1[i]:>7.4f}", flush=True)
                print("-" * 56, flush=True)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} Val AUROC: {val_auroc:.4f} Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} "
                  f"[val skipped — runs every {val_freq} epochs]")
        
        score = val_auroc + val_f1
        if score > best_val_score:
            best_val_score = score
            torch.save(model.state_dict(), best_model_path)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_score': best_val_score
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"=> Saved new best model and training checkpoint to {best_model_path} and {checkpoint_path}")
            
    writer.close()

if __name__ == "__main__":
    import sys
    import argparse
    print("DEBUG: Entered train.py script", flush=True)

    parser = argparse.ArgumentParser(description="Train CT multi-abnormality classification pipeline.")
    parser.add_argument("--use_swin", action="store_true", default=True, help="Use SOTA 3D Swin-Transformer model backbone.")
    parser.add_argument("--ssl_weights_path", type=str, default="/scratch/25208443/pretrain_weights/model_swinvit.pt", help="Path to pre-trained SSL weights.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Custom path to save/load checkpoint.")
    parser.add_argument("--best_model_path", type=str, default=None, help="Custom path to save best model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head (backbone gets 0.1x of this).")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--reset_scheduler", action="store_true", default=False,
                        help="Skip loading scheduler state on resume — restarts cosine LR schedule from T_max (warm restart).")
    parser.add_argument("--weak_class_gamma", type=float, default=None,
                        help="Per-class ASL gamma_neg for weak classes (Lung nodule=9, Fibrotic sequela=11, Bronchiectasis=16). "
                             "Default None = global gamma_neg=4.0 for all classes. Recommended: 2.0.")
    args = parser.parse_args()

    # HPC Dataset Paths
    train_csv_path = "/scratch/25208443/dataset/dataset/multi_abnormality_labels/train_predicted_labels.csv"
    valid_csv_path = "/scratch/25208443/dataset/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    train_images_dir = "/scratch/25208443/data_volumes/dataset/train/"
    valid_images_dir = "/scratch/25208443/data_volumes/dataset/valid/"
    
    print(f"Initializing CT-RATE classification pipeline on HPC...", flush=True)
    
    # Pre-check paths
    for p in [train_csv_path, valid_csv_path, train_images_dir, valid_images_dir]:
        if not os.path.exists(p):
            print(f"CRITICAL ERROR: Path does not exist: {p}", flush=True)
            sys.exit(1)
            
    print(f"DEBUG: All paths verified. Starting recursive indexing...", flush=True)
    
    try:
        # Run Swin-Transformer training pipeline
        train_and_evaluate(
            train_csv_path,
            valid_csv_path,
            train_images_dir,
            valid_images_dir,
            num_epochs=args.num_epochs,
            use_swin=args.use_swin,
            lr=args.lr,
            warmup_epochs=5,
            freeze_epochs=3,
            ssl_weights_path=args.ssl_weights_path,
            checkpoint_path=args.checkpoint_path,
            best_model_path=args.best_model_path,
            reset_scheduler=args.reset_scheduler,
            weak_class_gamma=args.weak_class_gamma,
        )
    except Exception as e:
        print(f"CRITICAL EXCEPTION during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
