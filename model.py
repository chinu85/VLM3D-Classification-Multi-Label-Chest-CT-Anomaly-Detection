import torch
import torch.nn as nn
from monai.networks.nets import DenseNet201, SwinUNETR

class ChestCTClassificationModel(nn.Module):
    """
    Legacy Stage 3 CNN Backbone (DenseNet201)
    """
    def __init__(self, num_classes=18, spatial_dims=3, n_input_channels=1):
        super(ChestCTClassificationModel, self).__init__()
        self.backbone = DenseNet201(
            spatial_dims=spatial_dims,
            in_channels=n_input_channels,
            out_channels=num_classes,
            dropout_prob=0.1
        )
    
    def forward(self, x):
        return self.backbone(x)

class ChestCTSwinClassificationModel(nn.Module):
    """
    Stage 4 SOTA 3D Swin-Transformer Classification Model (SwinViT-3D)
    Utilizes the encoder from MONAI's SwinUNETR, pooled dynamically for classification.
    """
    def __init__(self, num_classes=18, in_channels=1, feature_size=48):
        super(ChestCTSwinClassificationModel, self).__init__()
        # SwinUNETR contains the SOTA SwinViT 3D transformer backbone
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,  # Dummy segmentation target, only using encoder
            feature_size=feature_size,  # Default 48 yields 768 projection dims
            use_checkpoint=False       # Disabled: reentrant checkpointing silently blocks encoder gradients under AMP
        )
        # Bottleneck projection features = feature_size * 16 = 768
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(feature_size * 16, num_classes)

    def forward(self, x):
        # Extract hierarchical encoder feature maps from Swin ViT
        features = self.backbone.swinViT(x)
        # Bottleneck is the last feature map (index 4) of shape: (B, 768, H/32, W/32, D/32)
        bottleneck = features[4]
        pooled = self.pool(bottleneck).squeeze(-1).squeeze(-1).squeeze(-1)  # Shape: (B, 768)
        logits = self.fc(self.dropout(pooled))
        return logits

if __name__ == "__main__":
    print("Testing legacy CNN model...")
    model_cnn = ChestCTClassificationModel()
    dummy_input = torch.randn(1, 1, 64, 64, 32)
    output_cnn = model_cnn(dummy_input)
    print(f"CNN Output shape: {output_cnn.shape}")

    print("\nTesting new Swin-Transformer model...")
    # SwinUNETR expects spatial sizes that are divisible by 32 (like 64, 64, 32)
    model_swin = ChestCTSwinClassificationModel()
    output_swin = model_swin(dummy_input)
    print(f"Swin Output shape: {output_swin.shape}")
