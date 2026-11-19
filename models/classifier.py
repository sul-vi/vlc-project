"""用干净 CIFAR-10 训练分类器，按验证 Top-1 存档，供冻结感知损失和评估复用。"""
import torch
from torch import nn
from torch.utils.data import DataLoader

from checkpoints_io import save_best
from config import initialize, parameters
from data import datasets


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 3
        for outgoing in (32, 64, 96):
            layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)))
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(384, 10))

    def forward(self, x):
        return self.head(self.features(x))


def frozen(cfg, device, train_if_missing=False):
    """加载冻结分类器。首次训练仅使用与编解码器相同的训练划分。"""
    path = cfg.directory("checkpoints") / "classifier.pt"
    if not path.exists() and train_if_missing:
        train_classifier(cfg, device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Classifier().to(device)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False), state["info"]


def train_classifier(cfg, device):
    if cfg.dataset != "cifar10":
        raise ValueError("分类器训练需要带真实标签的 CIFAR-10")
    train, val = datasets(cfg, True, labeled=True)
    model = Classifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.classifier_epochs)
    loader = DataLoader(train, cfg.batch_size, shuffle=True)
    best = -1.
    for epoch in range(1, cfg.classifier_epochs + 1):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if torch.rand(()) < .5:
                images = images.flip((-1,))
            loss = nn.functional.cross_entropy(model(images), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schedule.step()
        model.eval()
        hits = 0
        with torch.inference_mode():
            for images, labels in DataLoader(val, cfg.batch_size):
                hits += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
        score = hits / len(val)
        print(f"classifier {epoch} val top1={score:.3f}", flush=True)
        if score > best:
            best = score
            info = dict(epoch=epoch, val_accuracy=score, seed=cfg.seed, train_limit=cfg.train_limit,
                        holdout=cfg.validation_holdout, train_samples=len(train))
            save_best(cfg.directory("checkpoints") / "classifier.pt", dict(model=model.state_dict(), info=info))


if __name__ == "__main__":
    cfg = parameters(__doc__, True)
    train_classifier(cfg, initialize(cfg))
