"""实验参数、命令行入口与随机性控制。路径均相对于项目目录。"""
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
COLORS = ("R", "B", "G")
JPEG, UNORDERED, ORDERED, FIXED, ADAPTIVE = (
    "JPEG + FEC 4:2:1", "Unordered semantic 4:2:1", "Ordered semantic 4:2:1",
    "Proposed 6:2:1 fixed", "Proposed 6:2:1 adaptive")
SCHEMES = (JPEG, UNORDERED, ORDERED, FIXED, ADAPTIVE)
RAW_JPEG = "JPEG uncoded 4:2:1"


@dataclass(frozen=True)
class Config:
    dataset: str = "cifar10"
    data_dir: str = str(ROOT / "data")
    image_size: int = 32       # 文件夹数据集的正方形裁剪边长，0 保持原分辨率
    uniform: tuple = (4., 2., 1.)
    proposed: tuple = (6., 2., 1.)
    channels: int = 16        # 每色 16×8×8 = 1024 位
    width: int = 32
    compression: float = 0.8  # 中点对称 tanh 的强度，0 对应线性响应
    balanced_pairs: bool = True  # True 保证逐帧平均光功率相同，False 运行原始单时隙 M-CPAM
    candidates: tuple = ((6., 2., 1.),)
    classifier_epochs: int = 14
    task_weight: float = 0.5
    snrs: tuple = (-5, 0, 5, 10, 15, 20, 25)
    train_snr: tuple = (-5., 25.)
    mode_probs: tuple = (0.25, 0.25, 0.50)  # 分别对应 2、4、8 阶
    val_points: tuple = ((2, 5.), (4, 10.), (8, 5.), (8, 20.))
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 64
    validation_holdout: int = 5000
    train_limit: int = 15000  # 0 表示全部，三个 limit 只截取各自的集合
    val_limit: int = 1024
    test_limit: int = 0
    seed: int = 2026
    device: str = "auto"
    run: str = "v3"
    output_dir: str = ""      # 可指定独立评估输出目录，方便同一权重评估不同数据集
    eval_ratio: str = ""      # 留空时使用验证集选出的比例，例如 6:2:1 可固定候选
    jpeg_qualities: tuple = (95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10, 5, 1)
    probe_symbols: int = 262144  # BER 对照中的独立三比特字数

    def directory(self, name):
        return ROOT / name / self.run


def parameters(description, training):
    """读取公共参数与当前入口支持的覆盖项。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=("cifar10", "folder", "synthetic"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--run", help="checkpoints 和 results 下的实验名")
    parser.add_argument("--data-dir")
    if not training:
        parser.add_argument("--output-dir", help="评估输出位置，默认 results/<run>")
        parser.add_argument("--eval-ratio", help="固定一个受训候选，例如 5:2:1 或 6:2:1")
    keys = ["batch_size", "seed", "image_size"]
    keys += ["epochs", "train_limit", "val_limit"] if training else ["test_limit"]
    for key in keys:
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    cfg = Config(**args)
    if cfg.run and (Path(cfg.run).name != cfg.run or cfg.run in (".", "..")):
        parser.error("--run 必须是单个目录名")
    if min(cfg.batch_size, cfg.epochs, cfg.channels, cfg.width) < 1:
        parser.error("batch size、轮数和网络宽度必须为正数")
    if min(cfg.train_limit, cfg.val_limit, cfg.test_limit, cfg.image_size, cfg.seed) < 0:
        parser.error("样本上限、图像尺寸和种子须为非负数")
    if cfg.dataset == "folder" and cfg.image_size == 0 and cfg.batch_size != 1:
        parser.error("文件夹原尺寸模式请使用 --batch-size 1，以支持不同尺寸")
    return cfg


def initialize(cfg):
    """设置随机源和设备。相同软件、设备和参数下可复现实验。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    return torch.device(device)
