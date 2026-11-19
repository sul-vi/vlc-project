"""图像源与固定数据划分。CIFAR-10 直接读取官方二进制归档，无需反序列化 pickle。"""
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset

CIFAR_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cifar_archive(root):
    """仅在官方 MD5 匹配后原子发布归档。中断下载不会变成有效数据集。"""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cifar-10-binary.tar.gz"
    if path.exists():
        if checksum(path) != CIFAR_MD5:
            raise ValueError(f"CIFAR-10 校验失败，请检查或移走 {path}")
        return path
    print(f"Downloading {CIFAR_URL}", flush=True)
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
        temporary = Path(output.name)
    try:
        with urllib.request.urlopen(CIFAR_URL, timeout=120) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        if checksum(temporary) != CIFAR_MD5:
            raise ValueError("下载的 CIFAR-10 与官方 MD5 不一致")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class CIFARImages(Dataset):
    def __init__(self, path, training, labeled=False):
        self.labeled = labeled
        names = [f"data_batch_{i}.bin" for i in range(1, 6)] if training else ["test_batch.bin"]
        batches = {}
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if name in names and member.isfile():
                    records = np.frombuffer(archive.extractfile(member).read(), np.uint8)
                    batches[name] = records.reshape(10000, 3073)
        records = np.concatenate([batches[name] for name in names])
        self.labels = records[:, 0]
        self.images = records[:, 1:].reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index]).float().div_(255)
        return (image, int(self.labels[index])) if self.labeled else image


class FolderImages(Dataset):
    """从划分目录递归读图。size>0 等比例缩放并中心裁剪，size=0 保持原始尺寸。"""
    def __init__(self, root, size):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions and p.is_file())
        if not self.files:
            raise ValueError(f"未找到图片：{root}")
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            rgb = image.convert("RGB")
            if self.size:
                rgb = ImageOps.fit(rgb, (self.size, self.size), method=Image.Resampling.BICUBIC)
            return torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div_(255)


class SyntheticImages(Dataset):
    """仅作离线冒烟测试，每个样本独立生成低频色块，不参与 CIFAR-10 结论。"""
    def __init__(self, count, seed):
        self.count, self.seed = count, seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(self.seed + index)
        small = torch.rand(1, 3, 5, 5, generator=gen)
        return torch.nn.functional.interpolate(small, size=(32, 32), mode="bilinear",
                                               align_corners=False)[0]


def datasets(cfg, training, labeled=False):
    """返回训练/验证二元组，或单独的测试集；先固定划分，再按 limit 截断。"""
    root = Path(cfg.data_dir)
    splits = ("train", "val") if training else ("test",)
    if cfg.dataset == "cifar10":
        images = CIFARImages(cifar_archive(root), training, labeled)
        if training:
            order = torch.randperm(len(images), generator=torch.Generator().manual_seed(cfg.seed))
            k = cfg.validation_holdout
            groups = [Subset(images, order[k:].tolist()), Subset(images, order[:k].tolist())]
        else:
            groups = [images]
    elif cfg.dataset == "folder":
        groups = [FolderImages(root / split, cfg.image_size) for split in splits]
    else:
        groups = [SyntheticImages(128, cfg.seed + 10000 * ("train", "val", "test").index(split))
                  for split in splits]
    limits = (cfg.train_limit, cfg.val_limit) if training else (cfg.test_limit,)
    result = []
    for group, limit, split in zip(groups, limits, splits):
        if limit and limit < len(group):
            gen = torch.Generator().manual_seed(cfg.seed + ("train", "val", "test").index(split))
            group = Subset(group, torch.randperm(len(group), generator=gen)[:limit].tolist())
        result.append(group)
    return tuple(result) if training else result[0]
