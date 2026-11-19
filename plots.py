"""将逐点指标绘制为可用于论文排版的 320 dpi PNG。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import ADAPTIVE, COLORS, SCHEMES

PALETTE = ("#545b69", "#25758c", "#a44628", "#64509b", "#ad7510")
MARKERS = ("o", "s", "D", "^", "v")


STYLES = (":", "--", "-.", "-", "--")


def series(ax, points, field, index, name):
    ax.plot([r["snr_db"] for r in points], [r[field] for r in points], color=PALETTE[index],
            marker=MARKERS[index], markerfacecolor="none", linestyle=STYLES[index], label=name)


def save_quality(rows, directory, cfg):
    """准确率左图比较任务输出，右图让所有重建图通过同一个冻结分类器。"""
    panels = {"psnr": [("psnr", "PSNR (dB)")], "ssim": [("ssim", "SSIM")]}
    if rows[0]["recon_acc"] is not None:
        panels["acc"] = [("acc", "End-to-end top-1 accuracy"), ("recon_acc", "Reconstruction top-1 accuracy")]
    for filename, fields in panels.items():
        fig, axes = plt.subplots(1, len(fields), figsize=(6.2 * len(fields), 4.2),
                                 squeeze=False, layout="constrained")
        for ax, (field, label) in zip(axes[0], fields):
            for i, name in enumerate(SCHEMES):
                points = [r for r in rows if r["scheme"] == name]
                legend = name + " (image classifier)" if i == 0 and field == "acc" else name
                series(ax, points, field, i, legend)
            ax.set(xlabel="Reference electrical SNR (dB)", ylabel=label)
            ax.grid(alpha=.23)
            ax.legend(frameon=False, fontsize=8)
        fig.savefig(directory / f"{filename}_vs_snr.png", dpi=320)
        plt.close(fig)


def save_modes(rows, mode_rows, directory, psnr_policy):
    """联合策略与仅按 PSNR 标定的策略均使用验证集；上界来自测试单模式结果。"""
    metrics = [("psnr", "PSNR (dB)")]
    if rows[0]["acc"] is not None:
        metrics.append(("acc", "Core task top-1 accuracy"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.1),
                             squeeze=False, layout="constrained")
    snrs = sorted({r["snr_db"] for r in rows})
    for ax, (metric, label) in zip(axes[0], metrics):
        for i, mode in enumerate((2, 4, 8)):
            data = [r[metric] for r in mode_rows if r["mode"] == mode]
            ax.plot(snrs, data, color=PALETTE[i], marker=MARKERS[i], label=f"Fixed {mode}")
        oracle = [max(r[metric] for r in mode_rows if r["snr_db"] == db) for db in snrs]
        ax.scatter(snrs, oracle, s=100, facecolors="none", edgecolors="black", label="Test upper bound")
        adaptive = [r[metric] for r in rows if r["scheme"] == ADAPTIVE]
        ax.plot(snrs, adaptive, "--", color=PALETTE[3], lw=2.1, label="Adaptive (validation joint loss)")
        lookup = {(r["snr_db"], r["mode"]): r[metric] for r in mode_rows}
        reference = [lookup[db, mode] for db, mode in psnr_policy["grid"]]
        ax.plot(snrs, reference, ":", color=PALETTE[4], lw=2, label="Validation PSNR policy")
        ax.set(xlabel="Reference electrical SNR (dB)", ylabel=label)
        ax.grid(alpha=.23)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(directory / "mode_curves.jpg", dpi=320)
    plt.close(fig)


def save_ber(rows, directory):
    """物理 RGB 每色独立子图。零误码以 0.5/N 位置的空心倒三角表示，停用颜色留空。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True, layout="constrained")
    for ax, color in zip(axes, COLORS):
        for i, name in enumerate(SCHEMES):
            points = [r for r in rows if r["scheme"] == name]
            x = np.array([p["snr_db"] for p in points])
            ber = np.array([p[f"ber_{color}"] for p in points], dtype=float)
            count = np.array([p[f"bits_{color}"] for p in points])
            zero = ber == 0
            ax.plot(x, np.where(zero, np.nan, ber), color=PALETTE[i], linestyle=STYLES[i],
                    marker=MARKERS[i], markersize=4, label=name)
            ax.scatter(x[zero], .5/count[zero], facecolors="none", edgecolors=PALETTE[i], marker="v")
        ax.set(title=f"Physical {color}", xlabel="Reference electrical SNR (dB)", yscale="log")
        ax.grid(which="both", alpha=.2)
    axes[0].set_ylabel("Bit error rate")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], ls="", marker="v", mfc="none", color="black"))
    fig.legend(handles, labels + ["0 errors at 0.5/N"], loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(directory / "ber_vs_snr.jpg", dpi=320)
    plt.close(fig)


def save_examples(examples, directory):
    snrs = sorted(examples)
    fig, axes = plt.subplots(len(snrs), 6, figsize=(12.5, 2.4 * len(snrs)),
                             squeeze=False, layout="constrained")
    titles = ("Original", "JPEG", "Unordered 4:2:1", "Ordered 4:2:1", "Fixed 8-CPAM", "Adaptive")
    for i, snr in enumerate(snrs):
        for j, sample in enumerate(examples[snr]):
            ax = axes[i, j]
            ax.imshow(sample.permute(1, 2, 0).numpy().clip(0, 1), interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            if i == 0:
                ax.set_title(titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{snr} dB")
    fig.savefig(directory / "visual_comparison.png", dpi=320)
    plt.close(fig)
