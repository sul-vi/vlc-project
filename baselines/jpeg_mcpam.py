"""预算内的完整 JPEG 码流。像素按 RGB，连续比特按 R、B、G 轮流装入信息字。"""
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class JPEGFrames:
    bits: torch.Tensor
    sizes: list        # 含 JPEG 头部的实际字节数，0 表示发端预算不足


def pack(images, symbols, qualities):
    """symbols 是每张图的信息字数 S，预算 floor(3S/8) 字节，空位补零。"""
    arrays = images.detach().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    frames = np.zeros((len(arrays), symbols * 3), dtype=np.uint8)
    sizes = []
    for index, array in enumerate(arrays):
        image = Image.fromarray(array)
        size = 0
        for quality in qualities:
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=quality, subsampling=2, optimize=True)
            payload = stream.getvalue()
            if len(payload) * 8 <= symbols * 3:
                size = len(payload)
                frames[index, :size * 8] = np.unpackbits(np.frombuffer(payload, np.uint8))
                break
        sizes.append(size)
    bits = torch.from_numpy(frames.reshape(-1, symbols, 3)).transpose(1, 2)
    return JPEGFrames(bits.to(device=images.device, dtype=images.dtype), sizes)


def unpack(bits, image_size):
    """接收端只使用固定帧预算和约定尺寸。无法解码时输出中灰，并计入图像指标。"""
    h, w = image_size
    flat = bits.transpose(1, 2).contiguous().byte().cpu().numpy().reshape(bits.shape[0], -1)
    payloads = np.packbits(flat[:, :flat.shape[1] // 8 * 8], axis=1)
    images = []
    failures = 0
    for payload in payloads:
        try:
            with Image.open(io.BytesIO(payload.tobytes()), formats=("JPEG",)) as image:
                if image.size != (w, h):
                    raise ValueError("帧尺寸错误")
                decoded = np.array(image.convert("RGB"), dtype=np.float32) / 255
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            decoded = np.full((h, w, 3), 0.5, dtype=np.float32)
            failures += 1
        images.append(decoded)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(bits.device), failures
