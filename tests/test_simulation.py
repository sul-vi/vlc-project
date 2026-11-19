"""针对信道标定、功率/时隙预算、STE、JPEG 和指标的可重复回归测试。"""
import itertools
import unittest

import torch

from baselines.jpeg_mcpam import pack, unpack
from channel.mcpam_channel import MCPAMChannel
from config import Config
from metrics import bit_counts, image_scores
from models.semantic_net import SemanticNet


class SimulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        torch.set_num_threads(2)

    def test_noiseless_all_words_and_modes(self):
        bits = torch.tensor(list(itertools.product((0., 1.), repeat=3))).unsqueeze(-1)
        for k, ratio, mode in itertools.product((0., 0.8, 1.8), ((4, 2, 1), (6, 2, 1)), (2, 4, 8)):
            link = MCPAMChannel(ratio, k).eval()
            rx = link(bits, 10, mode, torch.zeros(8, 2, 1))
            mask = (rx.weights > 0).view(1, 3, 1)
            torch.testing.assert_close(rx.bits, bits * mask)
            torch.testing.assert_close(rx.optical_mean, torch.full((8,), 0.5))
            self.assertAlmostEqual(rx.weights.sum().item(), 1., places=6)
            errors, totals = bit_counts(bits, rx)
            self.assertEqual(errors, [0, 0, 0])
            self.assertEqual(totals, [8 if w > 0 else 0 for w in rx.weights])

    def test_sigma_and_ook_theory(self):
        # OOK 互补合并后的误码率是 Q(sqrt(2)*10^(SNR/20))，用它端到端检查噪声定标。
        import math
        link = MCPAMChannel().eval()
        for db in (0, 5, 25):
            self.assertAlmostEqual(link.noise_sigma(db) ** 2, .25 * 10 ** (-db / 10))
        bits = torch.randint(2, (1, 3, 150000)).float()
        rx = link(bits, 0, 2)
        errors, totals = bit_counts(bits, rx)
        self.assertAlmostEqual(errors[0] / totals[0], math.erfc(1.) / 2, delta=.003)

    def test_raw_single_slot_levels(self):
        bits = torch.tensor(list(itertools.product((0., 1.), repeat=3))).unsqueeze(-1)
        link = MCPAMChannel((4, 2, 1), compression=0, balanced_pairs=False).eval()
        rx = link(bits, 0, noise=torch.zeros(8, 1, 1))
        torch.testing.assert_close(rx.optical_mean, torch.arange(8) / 7.)
        torch.testing.assert_close(rx.bits, bits)

    def test_ste_and_inactive_gradients(self):
        bits = torch.ones(1, 3, 8, requires_grad=True)
        rx = MCPAMChannel().train()(bits, 0, 4)
        rx.bits.sum().backward()
        torch.testing.assert_close(bits.grad[:, :2], torch.ones_like(bits[:, :2]))
        self.assertEqual(bits.grad[:, 2].count_nonzero().item(), 0)
        net = SemanticNet(4, 8)
        x = torch.rand(2, 3, 31, 35)
        y, sent, _ = net(x, MCPAMChannel(), 5, 2)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(sent.size(-1), 4 * 8 * 9)
        (y - x).square().mean().backward()
        grad = net.core_encoder[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)

    def test_interleaver_and_task_head(self):
        from channel.interleaver import color_indices, restore
        model = SemanticNet().eval()
        bits, shape = model.encode(torch.rand(4, 3, 32, 32))
        self.assertEqual(bits.shape, (4, 3, 1024))
        index = color_indices(bits, torch.Generator().manual_seed(7))
        physical = bits.gather(1, index)
        for logical_color in range(3):
            counts = (index == logical_color).sum(-1)
            self.assertTrue(((counts >= 341) & (counts <= 342)).all())
        received = MCPAMChannel((4, 2, 1)).eval()(physical, float("inf"))
        restored = restore(received.bits, index)
        torch.testing.assert_close(restored, bits)
        logits = model.classify(restored, shape)
        altered = restored.clone()
        altered[:, 1:] = 1 - altered[:, 1:]
        torch.testing.assert_close(model.classify(altered, shape), logits)
        model.train()
        bits, shape = model.encode(torch.rand(4, 3, 32, 32))
        model.classify(bits, shape).sum().backward()
        self.assertTrue(model.core_bits.weight.grad.abs().sum() > 0)
        self.assertTrue(model.encoder[0].weight.grad is None or model.encoder[0].weight.grad.count_nonzero() == 0)

    def test_condition_masks_and_policy(self):
        from channel.mode_policy import active_mode, choose_policy
        model = SemanticNet(8, 8).eval()
        bits, shape = model.encode(torch.rand(1, 3, 32, 32))
        self.assertEqual(bits.shape, (1, 3, 512))
        y = model.decode(bits, shape, (32, 32), 2, -5)
        bits[:, 1:] = 1 - bits[:, 1:]
        torch.testing.assert_close(y, model.decode(bits, shape, (32, 32), 2, -5))
        snrs = (-5, 0, 5)
        rows = [dict(snr_db=db, mode=m, psnr=v) for db, scores in
                zip(snrs, ((3, 2, 1), (4, 5, 3), (5, 6, 8))) for m, v in zip((2, 4, 8), scores)]
        p = choose_policy(rows, snrs)
        self.assertEqual([active_mode(p, db) for db in snrs], [2, 4, 8])
        for row in rows:
            row["psnr"] = float(row["mode"] == (2 if row["snr_db"] == 0 else 8))
        p = choose_policy(rows, snrs)
        self.assertEqual([active_mode(p, db) for db in snrs], [8, 2, 8])

    def test_jpeg_budget_and_failure(self):
        x = torch.rand(2, 3, 32, 32)
        frames = pack(x, 2048, Config().jpeg_qualities)
        self.assertEqual(frames.bits.shape, (2, 3, 2048))
        self.assertTrue(all(0 < size <= 768 for size in frames.sizes))
        y, failures = unpack(frames.bits, (32, 32))
        self.assertEqual(failures, 0)
        self.assertEqual(y.shape, x.shape)
        tiny = pack(x, 32, Config().jpeg_qualities)
        self.assertEqual(tiny.sizes, [0, 0])
        grey, failures = unpack(tiny.bits, (32, 32))
        self.assertEqual(failures, 2)
        torch.testing.assert_close(grey, torch.full_like(x, .5))

    def test_protected_jpeg(self):
        import numpy as np
        from baselines import jpeg_fec
        data = np.random.default_rng(11).integers(256, size=(4, 120), dtype=np.uint8)
        coded = jpeg_fec.rm_encode(data, 3072)
        # 每个 16 位码字翻转 3 位，达到保证纠正的半径。
        errors = np.zeros((4,192,16), np.uint8)
        errors[:, :, :3] = 1
        coded ^= errors.reshape(4,3072)[:, jpeg_fec.permutation(3072)]
        np.testing.assert_array_equal(jpeg_fec.rm_decode(coded), data)
        x = torch.rand(4, 3, 32, 32)
        frames = jpeg_fec.pack(x, 1024, Config().jpeg_qualities)
        self.assertEqual(frames.bits.shape, (4, 3, 1024))
        self.assertTrue(all(0 < size <= 384 for size in frames.sizes))
        self.assertEqual(jpeg_fec.unpack(frames.bits, (32,32))[1], 0)
        self.assertEqual(jpeg_fec.unpack(torch.zeros_like(frames.bits), (32,32))[1], 4)

    def test_ssim(self):
        x = torch.rand(2, 3, 32, 32)
        psnr, ssim = image_scores(x, x.clone())
        self.assertTrue(torch.isposinf(psnr).all())
        torch.testing.assert_close(ssim, torch.ones(2))
        zeros, ones = torch.zeros_like(x), torch.ones_like(x)
        psnr, ssim = image_scores(zeros, ones)
        torch.testing.assert_close(psnr, torch.zeros(2))
        torch.testing.assert_close(ssim, torch.full((2,), .0001 / 1.0001))


if __name__ == "__main__":
    unittest.main()
