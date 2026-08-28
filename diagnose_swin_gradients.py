import torch
import os
import sys
from model import ChestCTSwinClassificationModel

def diagnose():
    print("=== Swin-Transformer Gradient Diagnostic ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate the model
    model = ChestCTSwinClassificationModel(num_classes=18).to(device)
    print("Model instantiated successfully.")
    
    # 2. Check if any parameter has requires_grad=False
    total_params = 0
    grad_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            grad_params += 1
        else:
            print(f"Parameter {name} has requires_grad = False!")
            
    print(f"Total parameters: {total_params}")
    print(f"Parameters with requires_grad=True: {grad_params}")
    
    # 3. Load pre-trained weights (mirroring train.py logic)
    ssl_weights_path = "/scratch/25208443/pretrain_weights/model_swinvit.pt"
    if os.path.exists(ssl_weights_path):
        print(f"Loading weights from {ssl_weights_path}...")
        try:
            weights = torch.load(ssl_weights_path, map_location=device, weights_only=False)
            state_dict = weights.get("state_dict", weights.get("net", weights))
            model_dict = model.state_dict()
            load_dict = {}
            for k, v in state_dict.items():
                clean_k = k[7:] if k.startswith("module.") else k
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
                print(f"Successfully loaded {len(load_dict)} encoder pre-trained layers.")
            else:
                print("No weights loaded because of mismatch.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(f"No weights file found at {ssl_weights_path}. Running unit test with scratch weights.")

    # 4. Perform a forward and backward pass
    model.train()
    
    # SwinUNETR input shape: (B, C, H, W, D)
    # 160x160x112 is the crop size used in Stage 3/4
    dummy_input = torch.randn(2, 1, 160, 160, 112).to(device)
    dummy_label = torch.randint(0, 2, (2, 18)).float().to(device)
    
    print(f"Running forward pass with input shape {dummy_input.shape}...")
    
    # Use torch.autocast to mirror AMP setting in train.py
    # Since we use torch.amp.GradScaler in training, let's test both standard and mixed precision
    print("\n--- Test 1: Standard FP32 Backward Pass ---")
    model.zero_grad()
    outputs = model(dummy_input)
    # Simple loss
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(outputs, dummy_label)
    print(f"Loss computed: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass completed.")
    
    # Check gradients
    no_grad_fp32 = []
    zero_grad_fp32 = []
    has_grad_fp32 = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_fp32.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grad_fp32.append(name)
        else:
            has_grad_fp32.append(name)
            
    print(f"FP32 Result: {len(has_grad_fp32)} params with non-zero gradients.")
    print(f"FP32 Result: {len(no_grad_fp32)} params with None gradients.")
    print(f"FP32 Result: {len(zero_grad_fp32)} params with exactly zero gradients.")
    
    if len(no_grad_fp32) > 0 or len(zero_grad_fp32) > 0:
        print("\nWARNING: Some parameters have zero/None gradients in FP32!")
        print("First 5 params with None grad:", no_grad_fp32[:5])
        print("First 5 params with Zero grad:", zero_grad_fp32[:5])

    print("\n--- Test 2: Mixed Precision (AMP float16) Backward Pass ---")
    model.zero_grad()
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_label)
        
    print(f"AMP Loss computed: {loss.item():.4f}")
    
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
        
    print("AMP Backward pass completed.")
    
    no_grad_amp = []
    zero_grad_amp = []
    has_grad_amp = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_amp.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grad_amp.append(name)
        else:
            has_grad_amp.append(name)
            
    print(f"AMP Result: {len(has_grad_amp)} params with non-zero gradients.")
    print(f"AMP Result: {len(no_grad_amp)} params with None gradients.")
    print(f"AMP Result: {len(zero_grad_amp)} params with exactly zero gradients.")
    
    if len(no_grad_amp) > 0 or len(zero_grad_amp) > 0:
        print("\nWARNING: Some parameters have zero/None gradients in AMP!")
        print("First 5 params with None grad in AMP:", no_grad_amp[:5])
        print("First 5 params with Zero grad in AMP:", zero_grad_amp[:5])
        
    print("\n=== Diagnosis Complete ===")

if __name__ == "__main__":
    diagnose()
