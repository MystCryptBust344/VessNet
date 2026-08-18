import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, encoder_name="resnet34"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        return self.model(x)
