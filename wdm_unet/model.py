import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DeformableConvBlock(nn.Module):
    """Paper-inspired DCE block: standard branch + deformable branch + residual fusion."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.std = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.offset = nn.Conv2d(in_ch, 18, 3, padding=1)
        self.dcn = DeformConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.dcn_bn = nn.BatchNorm2d(out_ch)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        y_std = self.std(x)
        off = self.offset(x)
        y_dcn = self.dcn_bn(self.dcn(x, off))
        y = y_std + y_dcn
        y = self.proj(self.relu(y))
        return self.relu(y)


class HaarDWT(nn.Module):
    def forward(self, x):
        # Pad odd dimensions so 2x2 decimation is safe.
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        f00 = x[..., 0::2, 0::2]
        f01 = x[..., 0::2, 1::2]
        f10 = x[..., 1::2, 0::2]
        f11 = x[..., 1::2, 1::2]
        ll = (f00 + f01 + f10 + f11) * 0.25
        lh = (f00 - f01 + f10 - f11) * 0.25
        hl = (f00 + f01 - f10 - f11) * 0.25
        hh = (f00 - f01 - f10 + f11) * 0.25
        return ll, lh, hl, hh, (h, w)


class HaarIDWT(nn.Module):
    def forward(self, ll, lh, hl, hh, original_hw):
        n, c, h, w = ll.shape
        out = ll.new_zeros((n, c, h * 2, w * 2))
        out[..., 0::2, 0::2] = ll + lh + hl + hh
        out[..., 0::2, 1::2] = ll - lh + hl - hh
        out[..., 1::2, 0::2] = ll + lh - hl - hh
        out[..., 1::2, 1::2] = ll - lh - hl + hh
        oh, ow = original_hw
        return out[..., :oh, :ow]


class WaveletConvBlock(nn.Module):
    """Paper-inspired WCE: Conv-BN-ReLU -> Haar DWT -> process LL -> IDWT -> refine."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pre = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.dwt = HaarDWT()
        self.ll_proc = ConvBNReLU(out_ch, out_ch, 3, 1, 1)
        self.idwt = HaarIDWT()
        self.refine = ConvBNReLU(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        f = self.pre(x)
        ll, lh, hl, hh, hw = self.dwt(f)
        ll = self.ll_proc(ll)
        y = self.idwt(ll, lh, hl, hh, hw)
        y = F.relu(y, inplace=True)
        return self.refine(y)


class FusionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, a, b):
        return self.mix(torch.cat([a, b], dim=1))


class GatedFusionTransformer(nn.Module):
    """
    Shape-safe approximation of the paper's GFT equations:
      - patchify DCE and WCE features
      - cross-gate keys
      - two attention branches
      - MLP fusion + residual
    """
    def __init__(self, channels, patch_size=2, heads=4, mlp_ratio=2.0):
        super().__init__()
        assert channels % heads == 0
        self.channels = channels
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.patch_embed_d = nn.Conv2d(channels, channels, patch_size, patch_size, bias=False)
        self.patch_embed_w = nn.Conv2d(channels, channels, patch_size, patch_size, bias=False)

        self.q_d = nn.Linear(channels, channels)
        self.k_d = nn.Linear(channels, channels)
        self.v_d = nn.Linear(channels, channels)
        self.q_w = nn.Linear(channels, channels)
        self.k_w = nn.Linear(channels, channels)
        self.v_w = nn.Linear(channels, channels)

        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, channels),
        )
        self.norm_d = nn.LayerNorm(channels)
        self.norm_w = nn.LayerNorm(channels)
        self.norm_out = nn.LayerNorm(channels)
        self.residual = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def _split_heads(self, x):
        b, n, c = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, dce, wce):
        b, c, h, w = dce.shape
        # Ensure divisibility by patch size.
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        if ph or pw:
            dce_p = F.pad(dce, (0, pw, 0, ph), mode="replicate")
            wce_p = F.pad(wce, (0, pw, 0, ph), mode="replicate")
        else:
            dce_p, wce_p = dce, wce

        td = self.patch_embed_d(dce_p).flatten(2).transpose(1, 2)
        tw = self.patch_embed_w(wce_p).flatten(2).transpose(1, 2)
        td = self.norm_d(td)
        tw = self.norm_w(tw)

        qd, kd, vd = map(self._split_heads, (self.q_d(td), self.k_d(td), self.v_d(td)))
        qw, kw, vw = map(self._split_heads, (self.q_w(tw), self.k_w(tw), self.v_w(tw)))

        # Cross-gated keys; close to Eq. (12), but symmetric values are used for stable fusion.
        gated_kw = kw * torch.sigmoid(kd) + kw
        gated_kd = kd * torch.sigmoid(kw) + kd

        attn_w = torch.softmax((qw @ gated_kw.transpose(-2, -1)) * self.scale, dim=-1)
        attn_d = torch.softmax((qd @ gated_kd.transpose(-2, -1)) * self.scale, dim=-1)
        zw = self._merge_heads(attn_w @ vw)
        zd = self._merge_heads(attn_d @ vd)

        z = self.mlp(torch.cat([zw, zd], dim=-1))
        z = self.norm_out(z)

        hp, wp = dce_p.shape[-2] // self.patch_size, dce_p.shape[-1] // self.patch_size
        z = z.transpose(1, 2).reshape(b, c, hp, wp)
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        res = self.residual(torch.cat([dce, wce], dim=1))
        return z + res


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.pre = DepthwiseSeparableConv(in_ch, out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.pre(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class WDMUNet(nn.Module):
    """
    Reproducible paper-faithful WDM-UNet approximation.

    The paper gives module equations and diagrams but not complete source code/channel table.
    This implementation preserves the four core components and U-shaped multi-scale fusion.
    """
    def __init__(self, in_channels=1, base=32, patch_size=2, heads=4):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]

        self.dce1 = DeformableConvBlock(in_channels, chs[0])
        self.wce1 = WaveletConvBlock(in_channels, chs[0])
        self.fuse1 = FusionBlock(chs[0])

        self.dce2 = DeformableConvBlock(chs[0], chs[1])
        self.wce2 = WaveletConvBlock(chs[0], chs[1])
        self.fuse2 = FusionBlock(chs[1])

        self.dce3 = DeformableConvBlock(chs[1], chs[2])
        self.wce3 = WaveletConvBlock(chs[1], chs[2])
        self.fuse3 = FusionBlock(chs[2])

        self.dce4 = DeformableConvBlock(chs[2], chs[3])
        self.wce4 = WaveletConvBlock(chs[2], chs[3])
        self.gft = GatedFusionTransformer(chs[3], patch_size=patch_size, heads=heads)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = DecoderBlock(chs[3], chs[2], chs[2])
        self.dec2 = DecoderBlock(chs[2], chs[1], chs[1])
        self.dec1 = DecoderBlock(chs[1], chs[0], chs[0])
        self.final_refine = DepthwiseSeparableConv(chs[0], chs[0])
        self.out = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        d1, w1 = self.dce1(x), self.wce1(x)
        s1 = self.fuse1(d1, w1)
        p1 = self.pool(s1)

        d2, w2 = self.dce2(p1), self.wce2(p1)
        s2 = self.fuse2(d2, w2)
        p2 = self.pool(s2)

        d3, w3 = self.dce3(p2), self.wce3(p2)
        s3 = self.fuse3(d3, w3)
        p3 = self.pool(s3)

        d4, w4 = self.dce4(p3), self.wce4(p3)
        bottleneck = self.gft(d4, w4)

        x = self.dec3(bottleneck, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.final_refine(x)
        return self.out(x)  # logits


if __name__ == "__main__":
    model = WDMUNet(in_channels=1, base=32)
    x = torch.randn(2, 1, 565, 565)
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape, "output:", y.shape)
    print("parameters(M):", sum(p.numel() for p in model.parameters()) / 1e6)
