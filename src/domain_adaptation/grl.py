import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses the gradient sign during backprop to fool the domain discriminator."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Classifies encoder features as source (0) or target (1).
    Operates on the bottleneck feature map from the U-Net++ encoder.
    Input: (B, in_channels, H, W) averaged to (B, in_channels) via global avg pool.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, features, alpha=1.0):
        # Gradient reversal so encoder learns domain-invariant features
        x = grad_reverse(features, alpha)
        x = self.net(x)
        return x  # raw logits, use BCEWithLogitsLoss
