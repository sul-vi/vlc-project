"""等时隙的 JPEG 保护传输。完整 JPEG 经无损 DEFLATE，再用 RM(1,4) 分组纠错。"""
import io
import zlib
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from baselines.jpeg_mcpam import JPEGFrames

# 32 个 5 位消息代表 GF(2)^4 上的 32 个仿射函数，真值表即 16 位码字。
CODE = np.array([[(int(m & x).bit_count() + (m >> 4)) % 2 for x in range(16)]
                 for m in range(32)], dtype=np.uint8)
POW5 = 1 << np.arange(4, -1, -1)


def jpeg_bytes(image, quality):
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality, optimize=True, subsampling=2)
    return stream.getvalue()


@lru_cache(maxsize=1)
def dictionary():
    """收发两端固定的无损压缩字典，只使用纯黑 32×32 合成图，没有样本相关旁信息。"""
    blank = Image.new("RGB", (32, 32))
    return b"".join(jpeg_bytes(blank, q) for q in (95,90,85,80,75,70,60,50,40,30,20,10,5,1))[-32768:]


@lru_cache(maxsize=8)
def permutation(length):
    """打散同一 RM 码字的位置，降低同一 SIC 字内的相关错误。"""
    return np.random.default_rng(314159).permutation(length)


def rm_encode(data, capacity):
    """输入 [N,信息字节]，返回每张图恰好 capacity 个物理分支比特，尾部补零。"""
    blocks = capacity // 16
    payload = np.zeros((len(data), blocks*5), np.uint8)
    payload[:, :data.shape[1]*8] = np.unpackbits(data, axis=1)
    messages = payload.reshape(len(data), blocks, 5) @ POW5
    coded = np.zeros((len(data), capacity), np.uint8)
    coded[:, :blocks*16] = CODE[messages].reshape(len(data), -1)
    return coded[:, permutation(capacity)]


def rm_decode(physical):
    """硬比特上按最小汉明距离解码。16 点 Walsh 变换同时比较全部仿射函数。"""
    n, capacity = physical.shape
    blocks = capacity // 16
    ordered = np.empty_like(physical)
    ordered[:, permutation(capacity)] = physical
    signs = 1 - 2*ordered[:, :blocks*16].reshape(n, blocks, 16).astype(np.int16)
    for half in (1, 2, 4, 8):
        pairs = signs.reshape(n, blocks, -1, 2, half)
        left = pairs[..., 0, :].copy()
        right = pairs[..., 1, :].copy()
        pairs[..., 0, :] = left + right
        pairs[..., 1, :] = left - right
    slope = np.abs(signs).argmax(axis=-1)
    peak = np.take_along_axis(signs, slope[..., None], axis=-1)[..., 0]
    messages = slope + 16*(peak < 0)
    bits = ((messages[..., None] & POW5) != 0).reshape(n, -1)
    return np.packbits(bits[:, :blocks*5//8*8], axis=1)


def pack(images, symbols, qualities):
    """完整文件和最终发射帧均受 3S 位约束。384 字节帧有 120 信息字节（含 4 字节 CRC）。"""
    capacity = 3*symbols
    data = np.zeros((len(images), capacity//16*5//8), np.uint8)
    pixels = images.detach().clamp(0,1).mul(255).round().byte().permute(0,2,3,1).cpu().numpy()
    sizes = []
    for i, array in enumerate(pixels):
        image = Image.fromarray(array)
        size = 0
        for q in qualities:
            raw = jpeg_bytes(image, q)
            if len(raw)*8 > capacity:
                continue
            encoder = zlib.compressobj(9, zlib.DEFLATED, -15, zdict=dictionary())
            packed = encoder.compress(raw) + encoder.flush()
            if len(packed)+4 > data.shape[1]:
                continue
            size = len(raw)
            data[i, :len(packed)] = np.frombuffer(packed, np.uint8)
            crc = zlib.crc32(data[i, :-4]).to_bytes(4, "big")
            data[i, -4:] = np.frombuffer(crc, np.uint8)
            break
        sizes.append(size)
    coded = rm_encode(data, capacity)
    bits = torch.from_numpy(coded.reshape(len(images), symbols, 3)).transpose(1, 2)
    return JPEGFrames(bits.to(images), sizes)


def unpack(bits, image_size):
    """执行 RM 纠错、CRC 校验、限长解压和 JPEG 解码。丢帧按中灰图统计。"""
    h, w = image_size
    physical = bits.transpose(1,2).contiguous().byte().cpu().numpy().reshape(len(bits), -1)
    decoded = rm_decode(physical) if physical.shape[1] >= 16 else [b""]*len(bits)
    limit = physical.shape[1]//8
    images, failures = [], 0
    for row in decoded:
        try:
            data = bytes(row)
            if len(data)<5 or zlib.crc32(data[:-4]) != int.from_bytes(data[-4:], "big"):
                raise ValueError("CRC 校验失败")
            stream = zlib.decompressobj(-15, zdict=dictionary())
            raw = stream.decompress(data[:-4], limit+1)
            if not stream.eof or len(raw)>limit:
                raise ValueError("压缩载荷越界")
            with Image.open(io.BytesIO(raw), formats=("JPEG",)) as img:
                if img.size != (w,h):
                    raise ValueError("JPEG 尺寸错误")
                array = np.array(img.convert("RGB"), dtype=np.float32)/255
        except (ValueError, OSError, SyntaxError, zlib.error, Image.DecompressionBombError):
            array = np.full((h,w,3), .5, np.float32)
            failures += 1
        images.append(array)
    return torch.from_numpy(np.stack(images)).permute(0,3,1,2).to(bits.device), failures
