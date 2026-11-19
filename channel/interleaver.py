"""共享伪随机交织。每个逻辑层分布到三种物理颜色的位数最多相差一位。"""
import torch


def color_indices(bits, generator):
    """逐帧打乱轮转次序，返回 [批量,物理颜色,字序号] 对应的逻辑颜色索引。"""
    n, _, words = bits.shape
    shifts = torch.rand(n, words, generator=generator).argsort(dim=1).to(bits.device)
    colors = torch.arange(3, device=bits.device).view(1, 3, 1)
    return (colors + shifts.unsqueeze(1)) % 3


def restore(physical, indices):
    """按发端的索引解交织；每个信息字内的三色索引恰好是一个排列。"""
    return torch.zeros_like(physical).scatter(1, indices, physical)
