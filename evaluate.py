"""配对评估 JPEG、无序/有序均匀方案及固定/动态非均匀方案。"""
import csv
import json
import platform
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import matplotlib
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader

from baselines import jpeg_fec, jpeg_mcpam
from channel.mode_policy import active_mode, calibrate
from channel.interleaver import color_indices, restore
from models.classifier import frozen
from channel.mcpam_channel import MCPAMChannel
from config import ADAPTIVE, COLORS, FIXED, JPEG, ORDERED, RAW_JPEG, SCHEMES, UNORDERED, initialize, parameters
from data import datasets
from metrics import bit_counts, image_scores
from models.semantic_net import SemanticNet
from plots import save_modes, save_ber, save_examples, save_quality


def counts_as_dict(errors, totals):
    fields = {}
    for i, color in enumerate(COLORS):
        fields[f"errors_{color}"] = int(errors[i])
        fields[f"bits_{color}"] = int(totals[i])
        fields[f"ber_{color}"] = errors[i] / totals[i] if totals[i] else None
    return fields


@dataclass
class Accumulator:
    images: int = 0
    slots: int = 0
    psnr: float = 0.
    ssim: float = 0.
    optical: float = 0.
    errors: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    totals: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    jpeg_failures: int = 0
    jpeg_overflows: int = 0
    correct: int = 0
    task_hits: int = 0
    task_samples: int = 0
    core_errors: int = 0
    core_total: int = 0
    labeled: int = 0
    jpeg_bytes: int = 0

    def add(self, original, decoded, bits, received, slots_per_word, classifier, labels,
            logits=None, logical_core=None, decoded_core=None):
        """物理 BER 按发射颜色统计，逻辑 R 的 BER 在解交织后单独统计。"""
        if logical_core is not None:
            self.core_errors += (logical_core != decoded_core).sum().item()
            self.core_total += logical_core.numel()
        if labels is not None:
            self.correct += (classifier(decoded).argmax(1).cpu() == labels).sum().item()
            self.labeled += len(labels)
            if logits is not None:
                self.task_hits += (logits.argmax(1).cpu() == labels).sum().item()
                self.task_samples += len(labels)
        psnr, ssim = image_scores(original, decoded)
        errors, totals = bit_counts(bits, received)
        self.errors += errors
        self.totals += totals
        self.psnr += psnr.sum().item()
        self.ssim += ssim.sum().item()
        self.images += len(original)
        slots_per_image = slots_per_word * bits.size(2)
        self.slots += len(original) * slots_per_image
        self.optical += received.optical_mean.sum().item() * slots_per_image

    def row(self, scheme, snr, mode):
        jpeg = scheme in (JPEG, RAW_JPEG)
        task_acc = self.task_hits / self.task_samples if self.task_samples else None
        recon_acc = self.correct / self.labeled if self.labeled else None
        return dict(scheme=scheme, snr_db=snr, mode=mode, images=self.images,
                    physical_slots=self.slots, mean_optical_power=self.optical / self.slots,
                    acc=recon_acc if jpeg else task_acc, task_acc=task_acc, recon_acc=recon_acc,
                    task_correct=self.task_hits if self.task_samples else None,
                    recon_correct=self.correct if self.labeled else None,
                    core_errors=self.core_errors if self.core_total else None, core_bits=self.core_total,
                    core_ber=self.core_errors/self.core_total if self.core_total else None,
                    psnr=self.psnr / self.images, ssim=self.ssim / self.images,
                    jpeg_decode_failures=self.jpeg_failures if jpeg else None,
                    jpeg_success_rate=1-self.jpeg_failures/self.images if jpeg else None,
                    jpeg_budget_failures=self.jpeg_overflows if jpeg else None,
                    jpeg_mean_bytes=self.jpeg_bytes / self.images if jpeg else None,
                    **counts_as_dict(self.errors, self.totals))


def probe_ber(cfg, links, device):
    """使用同一组三路 IID 位流和噪声，独立比较权重本身的误码影响。"""
    gen = torch.Generator().manual_seed(cfg.seed + 23)
    shape = (1, 3, cfg.probe_symbols)
    bits = torch.randint(2, shape, generator=gen).float().to(device)
    noise = torch.randn(1, links["uniform"].slots_per_word, cfg.probe_symbols, generator=gen).to(device)
    rows = []
    for snr in cfg.snrs:
        for name, scheme in (("uniform", ORDERED), ("proposed", FIXED)):
            received = links[name](bits, snr, 8, noise)
            rows.append(dict(snr_db=snr, scheme=scheme, **counts_as_dict(*bit_counts(bits, received))))
    return rows



def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(cfg):
    device = initialize(cfg)
    path = cfg.directory("checkpoints") / "codec.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    training = type(cfg)(**{f.name: state["config"][f.name] for f in fields(cfg) if f.name in state["config"]})
    model = SemanticNet(training.channels, training.width).to(device).eval()
    model.load_state_dict(state.pop("model"))
    classifier = None
    if training.dataset == "cifar10" or cfg.dataset == "cifar10":
        classifier, info = frozen(cfg, device)
        if state["classifier"] is not None and info != state["classifier"]:
            raise ValueError("冻结分类器与训练记录不一致")
    calibration_dir = cfg.data_dir if training.dataset == cfg.dataset else training.data_dir
    reports, selected = calibrate(model, replace(training, snrs=cfg.snrs, data_dir=calibration_dir), device)
    if cfg.eval_ratio:
        wanted = tuple(float(x) for x in cfg.eval_ratio.split(":"))
        matches = [r for r in reports if tuple(r["ratio"]) == wanted]
        if not matches:
            raise ValueError(f"指定比例须属于训练候选 {training.candidates}")
        selected = matches[0]
    policy = selected["policy"]
    cfg = replace(cfg, proposed=selected["ratio"], uniform=training.uniform,
                  balanced_pairs=training.balanced_pairs, compression=training.compression)
    links = {name: MCPAMChannel(r, cfg.compression, cfg.balanced_pairs).eval().to(device)
             for name, r in (("uniform", cfg.uniform), ("proposed", cfg.proposed))}
    labeled = cfg.dataset == "cifar10"
    test = datasets(cfg, False, labeled=labeled)
    gen = torch.Generator().manual_seed(cfg.seed + 17)
    mapping_rng = torch.Generator().manual_seed(cfg.seed + 117)
    modes = (2, 4, 8)
    keys = (JPEG, RAW_JPEG, UNORDERED, ORDERED, *modes)
    sums = {(s, db): Accumulator() for s in keys for db in cfg.snrs}
    clean = {key: Accumulator() for key in (JPEG, RAW_JPEG, UNORDERED, ORDERED, 8)}
    originals_correct = 0
    comparisons = ((ORDERED, UNORDERED), (FIXED, ORDERED), (ADAPTIVE, UNORDERED))
    paired = {(db, a, b): [0, 0] for db in cfg.snrs for a, b in comparisons}
    examples = {}
    k = links["uniform"].slots_per_word
    for batch, item in enumerate(DataLoader(test, cfg.batch_size)):
        images, labels = item if labeled else (item, None)
        images = images.to(device)
        if labeled:
            originals_correct += (classifier(images).argmax(1).cpu() == labels).sum().item()
        bits, latent = model.encode(images)
        indices = color_indices(bits, mapping_rng)
        interleaved = bits.gather(1, indices)
        noise = torch.randn(len(images), k, bits.size(2), generator=gen).to(device)
        jpeg_codecs = {JPEG: jpeg_fec, RAW_JPEG: jpeg_mcpam}
        frames = {name: codec.pack(images, bits.size(2), cfg.jpeg_qualities)
                  for name, codec in jpeg_codecs.items()}
        for db in (*cfg.snrs, float("inf")):
            noiseless = db == float("inf")
            preview = {}
            hits = {}
            for key in (clean if noiseless else keys):
                acc = clean[key] if noiseless else sums[key, db]
                mode = key if isinstance(key, int) else 8
                sent = frames[key].bits if key in frames else interleaved if key == UNORDERED else bits
                link = links["proposed" if isinstance(key, int) else "uniform"]
                received = link(sent, db, mode, noise)
                if key in frames:
                    decoded, failures = jpeg_codecs[key].unpack(received.bits, images.shape[-2:])
                    acc.jpeg_failures += failures
                    acc.jpeg_overflows += frames[key].sizes.count(0)
                    acc.jpeg_bytes += sum(frames[key].sizes)
                    acc.add(images, decoded, sent, received, k, classifier, labels)
                else:
                    logical = restore(received.bits, indices) if key == UNORDERED else received.bits
                    decoded = model.decode(logical, latent, images.shape[-2:], mode, db)
                    logits = model.classify(logical, latent) if labeled else None
                    if labeled and not noiseless:
                        hits[key] = logits.argmax(1).cpu() == labels
                    acc.add(images, decoded, sent, received, k, classifier, labels,
                            logits, bits[:, 0], logical[:, 0])
                if batch == 0 and db in (5, 15):
                    preview[key] = decoded[0].cpu()
            if preview:
                examples[db] = [images[0].cpu(), preview[JPEG], preview[UNORDERED], preview[ORDERED],
                                preview[8], preview[active_mode(policy, db)]]
            if hits:
                hits[FIXED] = hits[8]
                hits[ADAPTIVE] = hits[active_mode(policy, db)]
                for a, b in comparisons:
                    # 同一测试图在两个任务头中的对错差异，保留配对胜负原始计数。
                    paired[db, a, b][0] += (hits[a] & ~hits[b]).sum().item()
                    paired[db, a, b][1] += (~hits[a] & hits[b]).sum().item()
    # 整个 SNR 点采用共同策略；复用已计算的同一模式结果，保持逐图逐噪声严格配对。
    rows, mode_rows = [], []
    for db in cfg.snrs:
        mode = active_mode(policy, db)
        for scheme, key in zip(SCHEMES, (JPEG, UNORDERED, ORDERED, 8, mode)):
            rows.append(sums[key, db].row(scheme, db, key if isinstance(key, int) else 8))
        mode_rows.extend(sums[m, db].row("fixed", db, m) for m in modes)
    if len({r["physical_slots"] for r in rows}) != 1 or (cfg.balanced_pairs and any(
            abs(r["mean_optical_power"] - .5) > 1e-6 for r in rows)):
        raise RuntimeError("物理时隙或平均功率不一致")
    probe = probe_ber(cfg, links, device)
    directory = Path(cfg.output_dir) if cfg.output_dir else cfg.directory("results")
    if cfg.eval_ratio and not cfg.output_dir:
        directory = directory / ("ratio_" + "_".join(f"{x:g}" for x in selected["ratio"]))
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "metrics.csv", rows)
    write_rows(directory / "jpeg_uncoded.csv", [sums[RAW_JPEG, db].row(RAW_JPEG, db, 8) for db in cfg.snrs])
    write_rows(directory / "mode_metrics.csv", mode_rows)
    record = dict(config=asdict(cfg), checkpoint=state, device=str(device), torch=str(torch.__version__),
                  python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__,
                  pillow=PIL.__version__, test_images=len(test), validation_candidates=reports,
                  policy=policy, clean_original_top1=originals_correct/len(test) if labeled else None,
                  noiseless={name: clean[key].row(name, "noiseless", 8) for key, name in
                             ((JPEG, JPEG), (RAW_JPEG, RAW_JPEG), (UNORDERED, UNORDERED), (ORDERED, ORDERED), (8, FIXED))},
                  jpeg_transport=dict(source="complete JPEG", packing="raw DEFLATE, public black-image dictionary",
                                      correction="interleaved RM(1,4) [16,5,8]", checksum="CRC32",
                                      reference="jpeg_uncoded.csv"),
                  interleaver_seed=cfg.seed+117, iid_ber=probe,
                  paired_task_comparisons=[dict(snr_db=db, first=a, second=b,
                      first_only_correct=v[0], second_only_correct=v[1])
                      for (db, a, b), v in paired.items()] if labeled else [])
    (directory / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    save_quality(rows, directory, cfg)
    save_ber(rows, directory)
    save_modes(rows, mode_rows, directory, selected["psnr_policy"])
    if examples:
        save_examples(examples, directory)
    print(f"ratio={cfg.proposed}, policy={policy}, test={len(test)} -> {directory}", flush=True)


if __name__ == "__main__":
    evaluate(parameters(__doc__, training=False))
