import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, encoder_name="tu-swin_tiny_patch4_window7_224"):
        super().__init__()
        # Using a timm-universal Swin encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        # Swin Transformer (V1) natively expects fixed 224x224 inputs due to windowed attention
        # We downsample the input, pass it through the model, and upsample back to 512x512
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.model(x_resized)
        out_resized = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return out_resized
