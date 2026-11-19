"""共享双任务编解码器训练。所有功率和交织对照共享源编码、接收网络和参数。"""
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from channel.interleaver import color_indices, restore
from channel.mcpam_channel import MCPAMChannel
from checkpoints_io import save_best
from config import initialize, parameters
from data import datasets
from models.classifier import frozen
from models.semantic_net import SemanticNet


def transmit(bits, link, mode, db, indices=None, noise=None):
    """交织只在全开消融中使用，解码器总是收到恢复逻辑顺序的特征。"""
    sent = bits if indices is None else bits.gather(1, indices)
    physical = link(sent, db, mode, noise).bits
    return physical if indices is None else restore(physical, indices)


def dual_loss(model, x, label, logical, shape, mode, db, weight):
    y = model.decode(logical, shape, x.shape[-2:], mode, db)
    mse = F.mse_loss(y, x)
    if label is None:
        return mse
    return mse + weight * F.cross_entropy(model.classify(logical, shape), label)


@torch.inference_mode()
def validate(model, links, loader, cfg, device):
    """按验证集两个任务的训练目标选权重，交织与有序固定模式均参与。"""
    model.eval()
    rng = torch.Generator().manual_seed(cfg.seed + 91)
    total = count = 0
    for item in loader:
        x, label = item if cfg.dataset == "cifar10" else (item, None)
        x = x.to(device)
        if label is not None:
            label = label.to(device)
        bits, shape = model.encode(x)
        index = color_indices(bits, rng)
        noise = torch.randn(len(x), links[0].slots_per_word, bits.size(2), generator=rng).to(device)
        for mode, db in cfg.val_points:
            for link, mapping in ((links[0], index), (links[0], None), (links[1], None)):
                operating_mode = 8 if link is links[0] else mode
                rx = transmit(bits, link.eval(), operating_mode, db, mapping, noise)
                loss = dual_loss(model, x, label, rx, shape, operating_mode, db, cfg.task_weight)
                total += loss.item() * len(x)
                count += len(x)
    return total / count


def fit(cfg):
    device = initialize(cfg)
    labeled = cfg.dataset == "cifar10"
    classifier, info = frozen(cfg, device, True) if labeled else (None, None)
    if info and (info["seed"], info["train_limit"], info["holdout"]) != (
            cfg.seed, cfg.train_limit, cfg.validation_holdout):
        raise ValueError("分类器与训练划分不一致，请使用独立 --run")
    initialize(cfg)
    train, val = datasets(cfg, True, labeled=labeled)
    batches = DataLoader(train, cfg.batch_size, shuffle=True)
    model = SemanticNet(cfg.channels, cfg.width).to(device)
    if classifier is not None:
        model.core_encoder.load_state_dict(classifier.features[:-1].state_dict())
    links = [MCPAMChannel(r, cfg.compression, cfg.balanced_pairs).to(device)
             for r in (cfg.uniform, cfg.proposed)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=.03*cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    interleave_rng = torch.Generator().manual_seed(cfg.seed + 13)
    best = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.
        for step, item in enumerate(batches):
            x, label = item if labeled else (item, None)
            x = x.to(device)
            if label is not None:
                label = label.to(device)
            if rng.random() < .5:
                x = x.flip((-1,))
            if step % 3 == 0:
                order = rng.permutation(3)
            plan = order[step % 3]
            # 三组轮换：无序 4:2:1、有序 4:2:1、有序 6:2:1。降阶仅用于拟部署的自适应组。
            link = links[int(plan == 2)].train()
            mode = int(rng.choice((2, 4, 8), p=cfg.mode_probs)) if plan == 2 else 8
            db = float(rng.uniform(*cfg.train_snr))
            if rng.random() < .1:
                mode, db = 8, float("inf")
            bits, shape = model.encode(x)
            mapping = color_indices(bits, interleave_rng) if plan == 0 else None
            logical = transmit(bits, link, mode, db, mapping)
            loss = dual_loss(model, x, label, logical, shape, mode, db, cfg.task_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x)
        score = validate(model, links, DataLoader(val, cfg.batch_size), cfg, device)
        history.append(dict(epoch=epoch, loss=loss_sum / len(train), val_loss=score,
                            lr=optimizer.param_groups[0]["lr"]))
        scheduler.step()
        print(f"epoch {epoch}/{cfg.epochs} validation dual loss={score:.4f}", flush=True)
        if score < best:
            best = score
            save_best(cfg.directory("checkpoints") / "codec.pt",
                      dict(model=model.state_dict(), config=asdict(cfg), epoch=epoch,
                           val_loss=score, train_images=len(train), val_images=len(val), classifier=info))
        (cfg.directory("checkpoints") / "training.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    fit(parameters(__doc__, True))
