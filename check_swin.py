import torch
try:
    from monai.networks.nets import SwinUNETR
    print("Successfully imported SwinUNETR")
    model = SwinUNETR(img_size=(160, 160, 112), in_channels=1, out_channels=18)
    print("Successfully instantiated SwinUNETR")
except Exception as e:
    print(f"SwinUNETR check failed: {e}")

try:
    from monai.networks.nets import ViT
    print("Successfully imported ViT")
except Exception as e:
    print(f"ViT check failed: {e}")
