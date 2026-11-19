"""逐图 RGB PSNR、局部 Gaussian SSIM，以及活动物理分支误码计数。"""
import torch
import torch.nn.functional as F


def image_scores(original, reconstructed):
    """返回各图 PSNR(dB) 与 SSIM。动态范围 1，完全相同的图像 PSNR 为正无穷。"""
    mse = (original - reconstructed).square().mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse)
    window = min(11, *original.shape[-2:])
    window -= 1 - window % 2
    coords = torch.arange(window, device=original.device, dtype=original.dtype) - window // 2
    gaussian = torch.softmax(-coords.square() / (2 * 1.5 ** 2), dim=0)
    kernel = torch.outer(gaussian, gaussian).expand(3, 1, window, window).contiguous()
    fields = torch.cat((original, reconstructed, original.square(), reconstructed.square(),
                        original * reconstructed))
    mx, my, ex2, ey2, exy = F.conv2d(fields, kernel, groups=3).chunk(5)
    luminance = (2 * mx * my + 0.01 ** 2) / (mx.square() + my.square() + 0.01 ** 2)
    structure = (2 * (exy - mx * my) + 0.03 ** 2) / (
        (ex2 - mx.square()).clamp_min(0) + (ey2 - my.square()).clamp_min(0) + 0.03 ** 2)
    return psnr, (luminance * structure).mean(dim=(1, 2, 3))


def bit_counts(sent, received):
    """每色原始错误数和比较位数；关闭颜色两项均记 0，BER 留空。"""
    active = (received.weights > 0).long()
    errors = (sent != received.bits).sum(dim=(0, 2)) * active
    totals = active * (sent.shape[0] * sent.shape[2])
    return errors.tolist(), totals.tolist()
