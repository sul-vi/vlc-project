"""仅用训练留出集标定非均匀权重与模式切换边界。"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from channel.mcpam_channel import MCPAMChannel
from data import datasets


def choose_policy(rows, snrs, metric="psnr"):
    """最大化指定验证指标，允许降阶和回切，平分时优先较低阶。"""
    scores = {(r["snr_db"], r["mode"]): r[metric] for r in rows}
    modes = (2, 4, 8)
    path = [max(modes, key=lambda m: scores[db, m]) for db in snrs]
    switches = []
    current = path[0]
    for lo, hi in zip(snrs, snrs[1:]):
        # 三条线段的全部交点把区间切成若干段，每段内的优劣排序不变。
        cuts = {0., 1.}
        for i, a in enumerate(modes):
            for b in modes[i+1:]:
                left = scores[lo, a] - scores[lo, b]
                right = scores[hi, a] - scores[hi, b]
                if left * right < 0:
                    cuts.add(left / (left - right))
        knots = sorted(cuts)
        for start, end in zip(knots, knots[1:]):
            t = (start + end) / 2
            best = max(modes, key=lambda m: (1-t)*scores[lo, m] + t*scores[hi, m])
            if best != current:
                switches.append(dict(threshold_db=lo + start*(hi-lo), mode=best))
                current = best
    return dict(initial_mode=path[0], switches=switches, grid=list(zip(snrs, path)),
                metric=metric, mean_score=sum(scores[db, m] for db, m in zip(snrs, path))/len(path))


def active_mode(policy, db):
    for sample, selected in policy["grid"]:
        if db == sample:
            return selected
    mode = policy["initial_mode"]
    for switch in policy["switches"]:
        if db >= switch["threshold_db"]:
            mode = switch["mode"]
    return mode


@torch.inference_mode()
def calibrate(model, cfg, device):
    """有标签时按负联合损失选模式，同时保存 v2 的 PSNR 策略供对照。"""
    labeled = cfg.dataset == "cifar10"
    _, validation = datasets(cfg, True, labeled=labeled)
    links = [MCPAMChannel(r, cfg.compression, cfg.balanced_pairs).eval().to(device)
             for r in cfg.candidates]
    stats = {(i, db, mode): [0., 0, 0.] for i in range(len(links))
             for db in cfg.snrs for mode in (2, 4, 8)}
    gen = torch.Generator().manual_seed(cfg.seed + 73)
    for batch in DataLoader(validation, cfg.batch_size):
        x, labels = batch if labeled else (batch, None)
        x = x.to(device)
        if labeled:
            labels = labels.to(device)
        bits, shape = model.encode(x)
        noise = torch.randn(len(x), links[0].slots_per_word, bits.size(2), generator=gen).to(device)
        for (i, db, mode), totals in stats.items():
            rx = links[i](bits, db, mode, noise)
            y = model.decode(rx.bits, shape, x.shape[-2:], mode, db)
            totals[0] += (-10 * (x-y).square().mean((1,2,3)).log10()).sum().item()
            if labeled:
                logits = model.classify(rx.bits, shape)
                totals[1] += (logits.argmax(1) == labels).sum().item()
                cost = F.mse_loss(y, x) + cfg.task_weight * F.cross_entropy(logits, labels)
                totals[2] -= cost.item() * len(x)
    reports = []
    for i, ratio in enumerate(cfg.candidates):
        rows = [dict(snr_db=db, mode=mode, psnr=v[0]/len(validation),
                     acc=v[1]/len(validation) if labeled else None,
                     neg_dual_loss=v[2]/len(validation) if labeled else None)
                for (idx, db, mode), v in stats.items() if idx == i]
        reports.append(dict(ratio=ratio, rows=rows, psnr_policy=choose_policy(rows, cfg.snrs),
                            policy=choose_policy(rows, cfg.snrs, "neg_dual_loss" if labeled else "psnr")))
    return reports, max(reports, key=lambda r: r["policy"]["mean_score"])
