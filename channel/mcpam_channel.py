"""RGB 无滤光片、等效响应度为 1 的单 PD 链路。分支轴固定按 R、B、G 排列。"""
import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Reception:
    bits: torch.Tensor          # [N,3,S]，关闭分支输出 0
    weights: torch.Tensor       # 活动光功率比例，总和为 1
    optical_mean: torch.Tensor  # [N]，LED 模型后实际发送的平均光强


class MCPAMChannel(nn.Module):
    """每个信息字发送原码与反码两个时隙，保证每帧平均光功率严格一致。"""

    def __init__(self, ratios=(6., 2., 1.), compression=0.8, balanced_pairs=True):
        super().__init__()
        if len(ratios) != 3 or not all(math.isfinite(x) for x in ratios):
            raise ValueError("需要三个有限的 R/B/G 权重")
        r, b, g = ratios
        if not (r > b + g and b > g > 0):
            raise ValueError("逐层判决要求 R > B + G 且 B > G > 0")
        if not math.isfinite(compression) or compression < 0:
            raise ValueError("LED 压缩系数必须有限且非负")
        self.register_buffer("ratios", torch.tensor(ratios, dtype=torch.float32))
        self.compression = compression
        self.slots_per_word = 2 if balanced_pairs else 1

    def weights(self, mode):
        """将本阶活动颜色的峰值光功率之和归一到 1。"""
        if mode not in (2, 4, 8):
            raise ValueError("mode 取 2、4 或 8")
        weights = self.ratios.clone()
        weights[int(math.log2(mode)):] = 0
        return weights / weights.sum()

    def led(self, x):
        """等效公共驱动的中点对称软限幅，f(0)=0、f(1)=1、f(1-x)=1-f(x)。"""
        k = self.compression
        if k == 0:
            return x
        return 0.5 + torch.tanh(k * (2 * x - 1)) * (0.5 / math.tanh(k))

    @staticmethod
    def noise_sigma(snr_db):
        """以单位响应度和平均光强 0.5 为参考，返回单时隙电流噪声标准差。"""
        return 0.5 * 10.0 ** (-snr_db / 20)

    def sic(self, received, weights, levels):
        """在 PD 电流域比较相邻子星座边界，在发射幅度域累计已判分量。"""
        prefix = torch.zeros_like(received)
        residual = received
        base = self.led(prefix)
        decoded = []
        for index in range(levels):
            weight = weights[index]
            tail = weights[index + 1:].sum()
            # 当低位取全 1 或全 0 时，两簇相邻端点经 LED 映射后取中点。
            threshold = (self.led(prefix + tail) + self.led(prefix + weight)) * 0.5 - base
            bit = (residual >= threshold).to(received.dtype)
            decoded.append(bit)
            prefix = prefix + weight * bit
            # 线性时这一步就是 residual -= weight * bit；软限幅时扣除标定的光强差。
            next_base = self.led(prefix)
            residual = residual - (next_base - base)
            base = next_base
        decoded.extend(torch.zeros_like(received) for _ in range(3 - levels))
        return torch.stack(decoded, dim=1)

    def forward(self, bits, snr_db, mode=8, noise=None):
        """输入浮点二进制 [N,3,S]，snr_db 为标量。noise 可传标准正态 [N,时隙数,S]。"""
        if bits.ndim != 3 or bits.shape[1] != 3 or not bits.is_floating_point():
            raise ValueError("输入形状须为 [批量,3,每色比特数]，元素为浮点 0/1")
        weights = self.weights(mode)
        x = (bits * weights.view(1, 3, 1)).sum(dim=1)
        pair = self.slots_per_word == 2
        clean = self.led(torch.stack((x, 1 - x), dim=1) if pair else x.unsqueeze(1))
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("噪声必须与实际发送时隙的形状一致")
        received = clean + self.noise_sigma(snr_db) * noise
        combined = (received[:, 0] + 1 - received[:, 1]) * 0.5 if pair else received[:, 0]
        hard = self.sic(combined, weights, int(math.log2(mode)))
        if self.training:
            # 按数据流作硬判决 STE。前向保留真实 SIC 错误，反向对活动位用恒等近似。
            # 直接微分离散 SIC 会给出零梯度；停用颜色不能接收梯度。
            surrogate = bits * (weights > 0).view(1, 3, 1)
            hard = surrogate + (hard - surrogate).detach()
        return Reception(hard, weights, clean.mean(dim=(1, 2)))
