"""全卷积语义编解码器。通信分支 R/B/G 与输入图像 RGB 像素轴互相独立。"""
import torch
import torch.nn.functional as F
from torch import nn
from models.classifier import Classifier


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(1, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x):
        return F.silu(x + self.body(x))


class SemanticNet(nn.Module):
    """每色占 C×ceil(H/4)×ceil(W/4) 个信息位，解码后裁回原尺寸。"""

    def __init__(self, channels=16, width=32):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.SiLU(), nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(), ResidualBlock(width),
            nn.Conv2d(width, 2 * channels, 1),
        )
        # R 使用干净分类器的卷积结构，训练入口用同一训练划分的权重初始化。
        self.core_encoder = Classifier().features[:-1]
        self.core_bits = nn.Conv2d(96, channels, 1)
        self.task_head = nn.Sequential(
            nn.Conv2d(channels, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(384, 10))
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * channels + 4, width, 3, padding=1),
            nn.SiLU(), ResidualBlock(width),
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, image):
        """输入 [N,3,H,W] 且像素在 [0,1]。返回 R/B/G 位流及特征空间尺寸。"""
        h, w = image.shape[-2:]
        padded = F.pad(image, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        core = self.core_bits(self.core_encoder(padded))
        logits = torch.cat((core, self.encoder(padded)), dim=1)
        soft = torch.sigmoid(logits)
        hard = (logits >= 0).to(logits.dtype)
        quantized = soft + (hard - soft).detach() if self.training else hard
        return quantized.reshape(image.size(0), 3, -1), logits.shape[-2:]

    def classify(self, received, latent_size):
        """仅从恢复逻辑顺序后的 R 核心位预测十类 logits。"""
        red = received[:, 0].reshape(received.size(0), self.channels, *latent_size)
        return self.task_head(red)

    def decode(self, bits, latent_size, output_size, mode, snr_db):
        """解码时显式屏蔽停用色，附加三路可用性与归一化 SNR 四张条件图。"""
        h, w = output_size
        active = (1., float(mode >= 4), float(mode == 8))
        mask = bits.new_tensor(active).view(1, 3, 1)
        quality = (min(30., max(-5., snr_db)) + 5.) / 35.
        condition = bits.new_tensor((*active, quality)).view(1, 4, 1, 1)
        features = (bits * mask).reshape(bits.size(0), 3 * self.channels, *latent_size)
        features = torch.cat((features, condition.expand(bits.size(0), -1, *latent_size)), dim=1)
        return self.decoder(features)[:, :, :h, :w]

    def forward(self, image, link, snr_db, mode=8, noise=None):
        bits, shape = self.encode(image)
        received = link(bits, snr_db, mode, noise)
        reconstruction = self.decode(received.bits, shape, image.shape[-2:], mode, snr_db)
        return reconstruction, bits, received
