import os
import torch
from model import ChestCTSwinClassificationModel

def verify_suprem(weights_path="/scratch/25208443/pretrain_weights/supervised_suprem_swinunetr_2100.pth"):
    device = torch.device("cpu")
    print(f"=== SuPreM Pre-trained Weights Verification ===")
    print(f"Loading weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Checkpoint file not found at {weights_path}")
        print("Please download it first using:")
        print(f"wget -P {os.path.dirname(weights_path)} https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_swinunetr_2100.pth")
        return False

    try:
        weights = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = weights.get("state_dict", weights.get("net", weights))
        print("Loaded state dict successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint file: {e}")
        return False

    model = ChestCTSwinClassificationModel(num_classes=18)
    model_dict = model.state_dict()
    
    load_dict = {}
    mismatched_shapes = []
    
    print("Mapping keys...")
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        
        # Map SwinUNETR's encoder keys (swinViT) to our model's nested backbone.swinViT
        if clean_k.startswith("swinViT."):
            mapped_k = "backbone." + clean_k
        else:
            mapped_k = "backbone.swinViT." + clean_k
            
        if mapped_k in model_dict:
            if model_dict[mapped_k].shape == v.shape:
                load_dict[mapped_k] = v
            else:
                mismatched_shapes.append((mapped_k, model_dict[mapped_k].shape, v.shape))

    print("-" * 50)
    print(f"Total keys in SuPreM checkpoint: {len(state_dict)}")
    print(f"Successfully matched and mapped keys: {len(load_dict)} / {len(model_dict)}")
    
    if mismatched_shapes:
        print(f"\nWARNING: {len(mismatched_shapes)} keys matched by name but had shape mismatches:")
        for name, expected, actual in mismatched_shapes[:5]:
            print(f"  - {name}: Expected {expected}, got {actual}")
        if len(mismatched_shapes) > 5:
            print(f"  - ... and {len(mismatched_shapes)-5} more.")
            
    if len(load_dict) > 0:
        print("\nSUCCESS: SuPreM weights are compatible and can be loaded successfully!")
        return True
    else:
        print("\nERROR: No keys matched. SuPreM weights are incompatible with current model definition.")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify SuPreM weights compatibility.")
    parser.add_argument("--weights_path", type=str, default="/scratch/25208443/pretrain_weights/supervised_suprem_swinunetr_2100.pth", help="Path to downloaded SuPreM weights.")
    args = parser.parse_args()
    verify_suprem(args.weights_path)
