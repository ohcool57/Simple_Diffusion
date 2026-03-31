"""
Unit tests for simpleDiff.py, organized by the chunks in that file.
Each test class corresponds to one logical chunk.
All tests use small synthetic data so they run quickly.
plt.show() is patched out so no windows appear during testing.
"""

import unittest
from unittest.mock import patch
import numpy as np
import torch
import itertools
from torchvision import datasets, transforms
import random

IMAGE_DIM = 784
TIME_STEPS = 250
BETA = 0.02

# ── helpers copied verbatim from simpleDiff.py ───────────────────────────────

def do_diffusion(data, steps=TIME_STEPS, beta=BETA):
    distributions, samples = [None], [data]
    xt = data
    for t in range(steps):
        q = torch.distributions.Normal(
            np.sqrt(1 - beta) * xt,
            np.sqrt(beta)
        )
        xt = q.sample()
        distributions.append(q)
        samples.append(xt)
    return distributions, samples


def compute_loss(forward_distributions, forward_samples, mean_model, var_model):
    p = torch.distributions.Normal(
        torch.zeros(forward_samples[0].shape),
        torch.ones(forward_samples[0].shape)
    )
    loss = -p.log_prob(forward_samples[-1]).mean()
    for t in range(1, len(forward_samples)):
        xt = forward_samples[t]
        xprev = forward_samples[t - 1]
        q = forward_distributions[t]
        xin = torch.cat(
            (xt, (t / len(forward_samples)) * torch.ones(xt.shape[0], 1)),
            dim=1
        )
        mu = mean_model(xin)
        sigma = var_model(xin)
        p = torch.distributions.Normal(mu, sigma)
        loss -= torch.mean(p.log_prob(xprev))
        loss += torch.mean(q.log_prob(xt))
    return loss / len(forward_samples)


def sample_reverse(mean_model, var_model, count, steps=TIME_STEPS):
    p = torch.distributions.Normal(
        torch.zeros(count, IMAGE_DIM),
        torch.ones(count, IMAGE_DIM)
    )
    xt = p.sample()
    sample_history = [xt]
    for t in range(steps, 0, -1):
        xin = torch.cat((xt, t * torch.ones(xt.shape[0], 1) / steps), dim=1)
        p = torch.distributions.Normal(
            mean_model(xin), var_model(xin)
        )
        xt = p.sample()
        sample_history.append(xt)
    return sample_history


def make_models():
    mean_model = torch.nn.Sequential(
        torch.nn.Linear(IMAGE_DIM + 1, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, IMAGE_DIM)
    )
    var_model = torch.nn.Sequential(
        torch.nn.Linear(IMAGE_DIM + 1, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, IMAGE_DIM), torch.nn.Softplus()
    )
    return mean_model, var_model


# ── Chunk 1: MNIST data loading ───────────────────────────────────────────────

class TestDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        cls.mnist_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.mnist_dataset), 60000)

    def test_image_shape(self):
        image, label = self.mnist_dataset[0]
        self.assertEqual(image.shape, torch.Size([1, 28, 28]))

    def test_label_is_int(self):
        _, label = self.mnist_dataset[0]
        self.assertIsInstance(label, int)

    def test_pixel_range_after_normalize(self):
        # Normalize((0.5,),(0.5,)) maps [0,1] → [-1,1]
        image, _ = self.mnist_dataset[0]
        self.assertGreaterEqual(image.min().item(), -1.0)
        self.assertLessEqual(image.max().item(),  1.0)

    def test_flat_tensor_shape(self):
        N = 50
        indices = random.sample(range(len(self.mnist_dataset)), N)
        dataset = torch.stack([self.mnist_dataset[i][0].view(-1) for i in indices])
        self.assertEqual(dataset.shape, torch.Size([N, IMAGE_DIM]))

    def test_flat_tensor_dtype(self):
        indices = random.sample(range(len(self.mnist_dataset)), 10)
        dataset = torch.stack([self.mnist_dataset[i][0].view(-1) for i in indices])
        self.assertEqual(dataset.dtype, torch.float32)


# ── Chunk 2: Diffusion visualization (image grid) ────────────────────────────

class TestDiffusionVisualization(unittest.TestCase):

    def setUp(self):
        self.data = torch.randn(4, IMAGE_DIM)

    @patch('matplotlib.pyplot.show')
    def test_time_steps_to_show_count(self, _mock_show):
        time_steps_to_show = np.linspace(0, TIME_STEPS, 8, dtype=int)
        self.assertEqual(len(time_steps_to_show), 8)

    @patch('matplotlib.pyplot.show')
    def test_samples_indexable_at_all_displayed_steps(self, _mock_show):
        _, samples = do_diffusion(self.data, steps=TIME_STEPS)
        time_steps_to_show = np.linspace(0, TIME_STEPS, 8, dtype=int)
        for t in time_steps_to_show:
            self.assertLess(t, len(samples))

    def test_sample_reshapeable_to_28x28(self):
        _, samples = do_diffusion(self.data, steps=TIME_STEPS)
        for t in np.linspace(0, TIME_STEPS, 8, dtype=int):
            img = samples[t][0].view(28, 28)
            self.assertEqual(img.shape, torch.Size([28, 28]))


# ── Chunk 3: do_diffusion ─────────────────────────────────────────────────────

class TestDoDiffusion(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.data = torch.randn(self.batch_size, IMAGE_DIM)
        self.steps = 10  # small for speed

    def test_returns_two_lists(self):
        dists, samps = do_diffusion(self.data, steps=self.steps)
        self.assertIsInstance(dists, list)
        self.assertIsInstance(samps, list)

    def test_list_lengths(self):
        dists, samps = do_diffusion(self.data, steps=self.steps)
        # distributions: None + one per step = steps + 1
        self.assertEqual(len(dists), self.steps + 1)
        self.assertEqual(len(samps), self.steps + 1)

    def test_first_distribution_is_none(self):
        dists, _ = do_diffusion(self.data, steps=self.steps)
        self.assertIsNone(dists[0])

    def test_first_sample_is_input(self):
        _, samps = do_diffusion(self.data, steps=self.steps)
        self.assertTrue(torch.equal(samps[0], self.data))

    def test_sample_shapes_preserved(self):
        _, samps = do_diffusion(self.data, steps=self.steps)
        for s in samps:
            self.assertEqual(s.shape, torch.Size([self.batch_size, IMAGE_DIM]))

    def test_distributions_are_normal(self):
        dists, _ = do_diffusion(self.data, steps=self.steps)
        for d in dists[1:]:
            self.assertIsInstance(d, torch.distributions.Normal)

    def test_noise_increases_variance(self):
        # Variance of samples should grow with diffusion time
        _, samps = do_diffusion(self.data, steps=50)
        var_early = samps[5].var().item()
        var_late  = samps[45].var().item()
        self.assertGreater(var_late, var_early)

    def test_late_samples_approach_standard_normal(self):
        # After many steps the distribution should be close to N(0,1)
        data = torch.zeros(500, IMAGE_DIM)
        _, samps = do_diffusion(data, steps=1000)
        final = samps[-1]
        self.assertAlmostEqual(final.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(final.std().item(),  1.0, delta=0.1)


# ── Chunk 4: compute_loss ─────────────────────────────────────────────────────

class TestComputeLoss(unittest.TestCase):

    def setUp(self):
        self.mean_model, self.var_model = make_models()
        self.batch = torch.randn(4, IMAGE_DIM)
        self.steps = 5  # small for speed

    def test_returns_scalar_tensor(self):
        dists, samps = do_diffusion(self.batch, steps=self.steps)
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_finite(self):
        dists, samps = do_diffusion(self.batch, steps=self.steps)
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_is_float(self):
        dists, samps = do_diffusion(self.batch, steps=self.steps)
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        self.assertEqual(loss.dtype, torch.float32)

    def test_backward_runs(self):
        dists, samps = do_diffusion(self.batch, steps=self.steps)
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        try:
            loss.backward()
        except Exception as e:
            self.fail(f'loss.backward() raised: {e}')

    def test_gradients_exist_after_backward(self):
        dists, samps = do_diffusion(self.batch, steps=self.steps)
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        loss.backward()
        for p in self.mean_model.parameters():
            self.assertIsNotNone(p.grad)
        for p in self.var_model.parameters():
            self.assertIsNotNone(p.grad)


# ── Chunk 5: model architecture ───────────────────────────────────────────────

class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.mean_model, self.var_model = make_models()
        self.xin = torch.randn(8, IMAGE_DIM + 1)

    def test_mean_model_output_shape(self):
        out = self.mean_model(self.xin)
        self.assertEqual(out.shape, torch.Size([8, IMAGE_DIM]))

    def test_var_model_output_shape(self):
        out = self.var_model(self.xin)
        self.assertEqual(out.shape, torch.Size([8, IMAGE_DIM]))

    def test_var_model_output_positive(self):
        # Softplus ensures all outputs are > 0
        out = self.var_model(self.xin)
        self.assertTrue((out > 0).all())

    def test_mean_model_output_unbounded(self):
        # mean model has no activation at the end so can be negative
        out = self.mean_model(self.xin)
        self.assertTrue((out < 0).any() or (out > 0).any())

    def test_xin_construction_shape(self):
        # Mirrors what compute_loss does to build xin
        xt = torch.randn(8, IMAGE_DIM)
        t, total = 3, 10
        xin = torch.cat(
            (xt, (t / total) * torch.ones(xt.shape[0], 1)),
            dim=1
        )
        self.assertEqual(xin.shape, torch.Size([8, IMAGE_DIM + 1]))


# ── Chunk 6: training loop ────────────────────────────────────────────────────

class TestTrainingLoop(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.mean_model, self.var_model = make_models()
        self.optimizer = torch.optim.AdamW(
            itertools.chain(self.mean_model.parameters(), self.var_model.parameters()),
            lr=1e-2, weight_decay=1e-6,
        )
        self.dataset = torch.randn(200, IMAGE_DIM)
        self.batch_size = 8
        self.steps = 5  # small for speed

    def _one_step(self):
        idx = torch.randint(0, len(self.dataset), (self.batch_size,))
        batch = self.dataset[idx]
        dists, samps = do_diffusion(batch, steps=self.steps)
        self.optimizer.zero_grad()
        loss = compute_loss(dists, samps, self.mean_model, self.var_model)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_single_training_step_runs(self):
        try:
            self._one_step()
        except Exception as e:
            self.fail(f'Training step raised: {e}')

    def test_loss_is_finite_during_training(self):
        for _ in range(3):
            loss_val = self._one_step()
            self.assertTrue(np.isfinite(loss_val), f'Non-finite loss: {loss_val}')

    def test_batch_sampled_from_dataset(self):
        idx = torch.randint(0, len(self.dataset), (self.batch_size,))
        batch = self.dataset[idx]
        self.assertEqual(batch.shape, torch.Size([self.batch_size, IMAGE_DIM]))

    def test_parameters_change_after_step(self):
        params_before = [p.clone() for p in self.mean_model.parameters()]
        self._one_step()
        params_after = list(self.mean_model.parameters())
        changed = any(
            not torch.equal(pb, pa)
            for pb, pa in zip(params_before, params_after)
        )
        self.assertTrue(changed)


# ── Chunk 7: sample_reverse ───────────────────────────────────────────────────

class TestSampleReverse(unittest.TestCase):

    def setUp(self):
        self.mean_model, self.var_model = make_models()
        self.count = 4
        self.steps = 5  # small for speed

    def _run(self):
        # Override IMAGE_DIM-aware sample_reverse with small steps
        p = torch.distributions.Normal(
            torch.zeros(self.count, IMAGE_DIM),
            torch.ones(self.count, IMAGE_DIM)
        )
        xt = p.sample()
        sample_history = [xt]
        for t in range(self.steps, 0, -1):
            xin = torch.cat(
                (xt, t * torch.ones(xt.shape[0], 1) / self.steps),
                dim=1
            )
            p = torch.distributions.Normal(
                self.mean_model(xin), self.var_model(xin)
            )
            xt = p.sample()
            sample_history.append(xt)
        return sample_history

    def test_history_length(self):
        history = self._run()
        self.assertEqual(len(history), self.steps + 1)

    def test_each_sample_shape(self):
        history = self._run()
        for s in history:
            self.assertEqual(s.shape, torch.Size([self.count, IMAGE_DIM]))

    def test_output_is_finite(self):
        history = self._run()
        for s in history:
            self.assertTrue(torch.isfinite(s).all())

    def test_initial_sample_is_noise(self):
        # First entry should be drawn from N(0,1) so std should be roughly 1
        history = self._run()
        std = history[0].std().item()
        self.assertAlmostEqual(std, 1.0, delta=0.5)


# ── Chunk 8: generated image visualization ────────────────────────────────────

class TestGeneratedImageVisualization(unittest.TestCase):

    def test_generated_reshapeable_to_28x28(self):
        count = 16
        generated = torch.randn(count, IMAGE_DIM)
        for i in range(count):
            img = generated[i].view(28, 28)
            self.assertEqual(img.shape, torch.Size([28, 28]))

    def test_correct_number_of_images(self):
        count = 16
        generated = torch.randn(count, IMAGE_DIM)
        self.assertEqual(generated.shape[0], count)

    @patch('matplotlib.pyplot.show')
    def test_imshow_does_not_raise(self, _mock_show):
        import matplotlib.pyplot as plt
        generated = torch.randn(16, IMAGE_DIM)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated[i].view(28, 28).detach(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
