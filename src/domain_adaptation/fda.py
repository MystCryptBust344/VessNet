import torch


def fda_transfer(source_img: torch.Tensor, target_img: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """
    Replaces the low-frequency amplitude of source_img with that of target_img.
    Args:
        source_img: (B, C, H, W) float tensor in [0, 1]
        target_img: (B, C, H, W) float tensor in [0, 1]
        beta: controls size of low-freq window (paper default: 0.01)
    Returns:
        adapted source image (B, C, H, W) clamped to [0, 1]
    """
    B, C, H, W = source_img.shape

    src_fft = torch.fft.fft2(source_img, dim=(-2, -1))
    tgt_fft = torch.fft.fft2(target_img, dim=(-2, -1))

    src_fft_shift = torch.fft.fftshift(src_fft, dim=(-2, -1))
    tgt_fft_shift = torch.fft.fftshift(tgt_fft, dim=(-2, -1))

    src_amp = torch.abs(src_fft_shift)
    src_pha = torch.angle(src_fft_shift)
    tgt_amp = torch.abs(tgt_fft_shift)

    h_half, w_half = int(beta * H), int(beta * W)
    cy, cx = H // 2, W // 2

    mask = torch.zeros(1, 1, H, W, device=source_img.device, dtype=torch.bool)
    mask[..., cy - h_half:cy + h_half, cx - w_half:cx + w_half] = True

    new_amp = src_amp.clone()
    new_amp[mask.expand(B, C, H, W)] = tgt_amp[mask.expand(B, C, H, W)]

    new_fft_shift = new_amp * torch.exp(1j * src_pha)

    new_fft = torch.fft.ifftshift(new_fft_shift, dim=(-2, -1))
    adapted = torch.fft.ifft2(new_fft, dim=(-2, -1)).real

    return adapted.clamp(0.0, 1.0)
