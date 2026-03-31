import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

### Data Loading -- Using MNIST Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data   = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)

# Visualise a few training images to confirm loading.
sample_batch, _ = next(iter(train_loader))
grid = make_grid(sample_batch[:16], nrow=4, normalize=True)
plt.figure(figsize=(4, 4))
plt.imshow(grid.permute(1, 2, 0), cmap='gray')
plt.axis('off')
plt.title('Training samples')
plt.show()

### Noise Schedule
TIME_STEPS = 1000
BETA_START = 1e-4
BETA_END   = 0.02

betas      = torch.linspace(BETA_START, BETA_END, TIME_STEPS)  # β_t
alphas     = 1.0 - betas                                         # 1 - β_t
alpha_bars = torch.cumprod(alphas, dim=0)                        # α_t = ∏(1-β_τ)

### Diffusion Kernel
def sample_zt(x, t):
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x)
    zt  = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * eps
    return zt, eps

### Noise Model
IMAGE_DIM = 28 * 28  # 784

class NoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMAGE_DIM + 1, 512), nn.ReLU(),
            nn.Linear(512, 512),           nn.ReLU(),
            nn.Linear(512, IMAGE_DIM)
        )

    def forward(self, x, t):
        # Flatten image and append normalised timestep as a scalar feature.
        x_flat = x.view(x.shape[0], -1)                        # (B, 784)
        t_norm = (t.float() / TIME_STEPS).unsqueeze(1)         # (B, 1)
        xin    = torch.cat([x_flat, t_norm], dim=1)            # (B, 785)
        return self.net(xin).view(x.shape)                      # (B, 1, 28, 28)

### Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

betas      = betas.to(device)
alphas     = alphas.to(device)
alpha_bars = alpha_bars.to(device)

noise_model = NoiseModel().to(device)
optim = torch.optim.AdamW(noise_model.parameters(), lr=2e-4, weight_decay=1e-4)

### Training Loop
EPOCHS = 10
loss_history = []

for epoch in range(EPOCHS):
    epoch_losses = []
    bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for x, _ in bar:
        x = x.to(device)
        t = torch.randint(0, TIME_STEPS, (x.shape[0],), device=device)
        zt, eps  = sample_zt(x, t)
        eps_pred = noise_model(zt, t)

        loss = F.mse_loss(eps_pred, eps)
        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_losses.append(loss.item())
        bar.set_postfix(loss=f'{loss.item():.4f}')

    mean_loss = float(np.mean(epoch_losses))
    loss_history.append(mean_loss)
    print(f'Epoch {epoch+1}: mean loss = {mean_loss:.4f}')

plt.plot(loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Training loss')
plt.show()

### Reverse Sampling
@torch.no_grad()
def sample_reverse(noise_model, count, steps=TIME_STEPS, device='cpu'):
    xt = torch.randn(count, 1, 28, 28, device=device)

    for t_idx in tqdm(range(steps - 1, -1, -1), desc='Sampling', leave=False):
        t_tensor    = torch.full((count,), t_idx, device=device, dtype=torch.long)
        eps_pred    = noise_model(xt, t_tensor)

        beta_t      = betas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]
        alpha_t     = alphas[t_idx]

        # Recover the posterior mean (Bishop Eq 20.16 rearranged for ε-prediction).
        mu = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        if t_idx > 0:
            xt = mu + torch.sqrt(beta_t) * torch.randn_like(xt)
        else:
            xt = mu

    return xt

### Visualise Generated Samples
samples = sample_reverse(noise_model, 16, device=device)
samples = (samples.clamp(-1, 1) + 1) / 2  # map [-1, 1] → [0, 1] for display

grid = make_grid(samples, nrow=4)
plt.figure(figsize=(5, 5))
plt.imshow(grid.permute(1, 2, 0).cpu(), cmap='gray')
plt.axis('off')
plt.title('Generated MNIST samples')
plt.show()