# -*- coding: utf-8 -*-
"""Part 1 — Variational Autoencoder (VAE) on MNIST.

Production-grade implementation of a fully-connected VAE with:
- Log-variance parameterization for training stability
- KL annealing (β-VAE schedule) to prevent posterior collapse
- Mixed-precision training (AMP) for efficient GPU utilization
- Comprehensive output pipeline: loss curves, reconstructions,
  generated samples, latent space PCA, and latent interpolation

Author : Samay Mehar (B22AI048)
Hardware: NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
"""

# ──────────────────────────────────────────────────────────────────────
# 0. Cache & environment overrides — MUST come before any torch import
# ──────────────────────────────────────────────────────────────────────
import os

os.environ["TORCH_HOME"] = r"D:\Projects2.0\Last Days Work\AML\Assignment3\torch_cache"
os.environ["XDG_CACHE_HOME"] = r"D:\Projects2.0\Last Days Work\AML\Assignment3\cache"

# ──────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────────────
# 2. Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ──────────────────────────────────────────────────────────────────────
# 3. Hyperparameters
# ──────────────────────────────────────────────────────────────────────
INPUT_DIM: int = 784        # 28 × 28 flattened
H_DIM: int = 400            # hidden layer width
Z_DIM: int = 20             # latent dimension (good disentanglement)
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 120
LR: float = 1e-3
KL_ANNEAL_EPOCHS: int = 10  # linearly ramp β from 0 → 1 over first N epochs
EARLY_STOP_PATIENCE: int = 10  # stop if no improvement for N consecutive epochs

# ──────────────────────────────────────────────────────────────────────
# 4. Paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = r"D:\Projects2.0\Last Days Work\AML\Assignment3"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CKPT_DIR = os.path.join(BASE_DIR, "outputs", "part1_vae", "checkpoints")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "part1_vae", "plots")
GEN_DIR = os.path.join(BASE_DIR, "outputs", "part1_vae", "generated")

for d in [DATASET_DIR, CKPT_DIR, PLOT_DIR, GEN_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 5. Device
# ──────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ──────────────────────────────────────────────────────────────────────
# 6. Data
# ──────────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=DATASET_DIR, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=DATASET_DIR, train=False, transform=transform, download=True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)


# ──────────────────────────────────────────────────────────────────────
# 7. Model
# ──────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    """Variational Autoencoder with log-variance parameterisation.

    Architecture (linear layers only, per assignment spec):
        Encoder: 784 → H_DIM (ReLU) → μ (Z_DIM), log_var (Z_DIM)
        Decoder: Z_DIM → H_DIM (ReLU) → 784 (raw logits; sigmoid applied at inference)
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 h_dim: int = H_DIM, z_dim: int = Z_DIM) -> None:
        """Initialise encoder and decoder layers.

        Args:
            input_dim: Flattened image dimensionality (784 for MNIST).
            h_dim: Hidden layer width.
            z_dim: Latent space dimensionality.
        """
        super().__init__()
        # Encoder
        self.enc_hidden = nn.Linear(input_dim, h_dim)
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_log_var = nn.Linear(h_dim, z_dim)
        # Decoder
        self.dec_hidden = nn.Linear(z_dim, h_dim)
        self.dec_out = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Flattened input images, shape (B, 784).

        Returns:
            mu: Mean of approximate posterior, shape (B, Z_DIM).
            log_var: Log-variance of approximate posterior, shape (B, Z_DIM).
        """
        h = self.relu(self.enc_hidden(x))
        mu = self.enc_mu(h)
        log_var = self.enc_log_var(h)
        return mu, log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ · ε, where ε ~ N(0, I).

        Args:
            mu: Mean of approximate posterior.
            log_var: Log-variance of approximate posterior.

        Returns:
            Sampled latent vector z.
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to raw logits (pre-sigmoid).

        Args:
            z: Latent vector, shape (B, Z_DIM).

        Returns:
            Logits, shape (B, 784). Apply sigmoid for pixel values.
        """
        h = self.relu(self.dec_hidden(z))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Args:
            x: Flattened input images, shape (B, 784).

        Returns:
            x_hat_logits: Reconstructed image logits, shape (B, 784).
            mu: Posterior mean.
            log_var: Posterior log-variance.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat_logits = self.decode(z)
        return x_hat_logits, mu, log_var


# ──────────────────────────────────────────────────────────────────────
# 8. Loss function
# ──────────────────────────────────────────────────────────────────────
# BCEWithLogitsLoss = sigmoid + BCE fused — autocast-safe & numerically stable
bce_logits_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")


def vae_loss(x: torch.Tensor, x_hat_logits: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE ELBO loss = Reconstruction + β · KL divergence.

    Uses BCEWithLogitsLoss (fused sigmoid+BCE) which is safe under AMP
    autocast and numerically more stable than separate sigmoid → BCELoss.

    Args:
        x: Original images (B, 784).
        x_hat_logits: Decoder output logits — pre-sigmoid (B, 784).
        mu: Posterior mean (B, Z_DIM).
        log_var: Posterior log-variance (B, Z_DIM).
        beta: KL annealing weight.

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors).
    """
    batch_size = x.size(0)
    recon = bce_logits_loss_fn(x_hat_logits, x) / batch_size
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    total = recon + beta * kl
    return total, recon, kl


# ──────────────────────────────────────────────────────────────────────
# 9. Training
# ──────────────────────────────────────────────────────────────────────
def train() -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the VAE with mixed-precision, KL annealing, early stopping, and checkpointing.

    Returns:
        model: Trained VAE model (best checkpoint loaded).
        history: Dictionary of per-epoch loss lists.
    """
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    history: Dict[str, List[float]] = {
        "total": [], "recon": [], "kl": []
    }
    best_loss = float("inf")
    patience_counter: int = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        # KL annealing: linearly ramp β from 0 → 1 over first KL_ANNEAL_EPOCHS
        beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}",
                     leave=False)
        for i, (x, _) in enumerate(loop):
            try:
                x = x.view(-1, INPUT_DIM).to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    x_hat, mu, log_var = model(x)
                    loss, recon, kl = vae_loss(x, x_hat, mu, log_var, beta)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_total += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
                num_batches += 1

                loop.set_postfix(
                    loss=f"{loss.item():.2f}",
                    recon=f"{recon.item():.2f}",
                    kl=f"{kl.item():.2f}",
                    beta=f"{beta:.2f}"
                )

                del x, x_hat, mu, log_var, loss, recon, kl
                if i % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"[WARN] OOM caught at batch {i} — skipping")
                    continue
                raise e

        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        print(f"  → Epoch {epoch:02d}  |  Loss: {avg_total:.4f}  "
              f"Recon: {avg_recon:.4f}  KL: {avg_kl:.4f}  β: {beta:.2f}")

        # Checkpointing + early stopping
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last_model.pt"))
        
        # Only track early stopping AFTER KL annealing finishes
        if epoch > KL_ANNEAL_EPOCHS:
            if avg_total < best_loss:
                best_loss = avg_total
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
                print(f"     ✓ Best model saved (loss={best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n[EARLY STOP] No improvement for {EARLY_STOP_PATIENCE} "
                          f"epochs. Stopping at epoch {epoch}.")
                    break
        else:
            # During annealing, always update best model and loss to start tracking properly after
            best_loss = avg_total
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))

        torch.cuda.empty_cache()

    # Load best checkpoint before returning
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                                     weights_only=True))
    print(f"[INFO] Loaded best model (loss={best_loss:.4f})")
    return model, history


# ──────────────────────────────────────────────────────────────────────
# 10. Visualization helpers
# ──────────────────────────────────────────────────────────────────────
def plot_loss_curves(history: Dict[str, List[float]]) -> None:
    """Save a three-panel loss curve figure.

    Args:
        history: Dictionary with keys 'total', 'recon', 'kl'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["total"]) + 1)
    titles = ["Total Loss", "Reconstruction Loss", "KL Divergence"]
    keys = ["total", "recon", "kl"]
    colors = ["#6C5CE7", "#00B894", "#E17055"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        ax.plot(epochs, history[key], color=color, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("VAE Training Curves", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_reconstructions(model: nn.Module) -> None:
    """Save a grid of original vs. reconstructed test images.

    Args:
        model: Trained VAE model.
    """
    model.eval()
    # Grab first batch from test set
    x_test, _ = next(iter(test_loader))
    x_test = x_test[:8].to(device)
    x_flat = x_test.view(-1, INPUT_DIM)

    with torch.no_grad():
        x_hat_logits, _, _ = model(x_flat)
    x_hat = torch.sigmoid(x_hat_logits).view(-1, 1, 28, 28)

    # Top row: originals, bottom row: reconstructions
    comparison = torch.cat([x_test, x_hat], dim=0).cpu()
    grid = make_grid(comparison, nrow=8, padding=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title("Top: Original  |  Bottom: Reconstructed", fontsize=13,
                 fontweight="bold")
    ax.axis("off")
    path = os.path.join(PLOT_DIR, "reconstructions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_generated_samples(model: nn.Module) -> None:
    """Generate 64 samples from the prior and save as 8×8 grid.

    Args:
        model: Trained VAE model.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, Z_DIM, device=device)
        samples = torch.sigmoid(model.decode(z)).view(-1, 1, 28, 28).cpu()

    grid = make_grid(samples, nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title("Generated Samples (z ~ N(0, I))", fontsize=13,
                 fontweight="bold")
    ax.axis("off")
    path = os.path.join(GEN_DIR, "samples_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_latent_space_pca(model: nn.Module) -> None:
    """Encode entire test set and visualize μ projected to 2-D via PCA.

    Args:
        model: Trained VAE model.
    """
    model.eval()
    all_mu: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, INPUT_DIM).to(device)
            mu, _ = model.encode(x)
            all_mu.append(mu.cpu())
            all_labels.append(y)

    mu_cat = torch.cat(all_mu, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()

    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu_cat)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        mu_2d[:, 0], mu_2d[:, 1],
        c=labels_cat, cmap="tab10", s=4, alpha=0.6
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=12)
    ax.set_title("Latent Space (μ) — PCA Projection", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = os.path.join(PLOT_DIR, "latent_space_pca.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_latent_interpolation(model: nn.Module) -> None:
    """Interpolate between two random test images of different digits.

    Args:
        model: Trained VAE model.
    """
    model.eval()
    # Find two images of different digits
    imgs, labels = [], []
    for x, y in test_dataset:
        imgs.append(x)
        labels.append(y)
        if len(imgs) >= 1000:
            break

    # Pick two different-class samples
    idx_a, idx_b = None, None
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                idx_a, idx_b = i, j
                break
        if idx_a is not None:
            break

    img_a = imgs[idx_a].view(1, INPUT_DIM).to(device)
    img_b = imgs[idx_b].view(1, INPUT_DIM).to(device)

    with torch.no_grad():
        mu_a, _ = model.encode(img_a)
        mu_b, _ = model.encode(img_b)

    num_steps = 10
    interp_imgs = []
    with torch.no_grad():
        for alpha in np.linspace(0, 1, num_steps):
            z_interp = (1 - alpha) * mu_a + alpha * mu_b
            decoded = torch.sigmoid(model.decode(z_interp)).view(1, 1, 28, 28).cpu()
            interp_imgs.append(decoded)

    strip = torch.cat(interp_imgs, dim=0)
    grid = make_grid(strip, nrow=num_steps, padding=2)

    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title(
        f"Latent Interpolation: digit {labels[idx_a]} → digit {labels[idx_b]}",
        fontsize=13, fontweight="bold"
    )
    ax.axis("off")
    path = os.path.join(GEN_DIR, "interpolation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────
# 11. Main entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run full VAE pipeline: train → evaluate → generate outputs."""
    print("=" * 60)
    print("  PART 1 — Variational Autoencoder (VAE) on MNIST")
    print("=" * 60)
    print(f"[INFO] Using device: {torch.cuda.get_device_name(device)}")

    # Train
    model, history = train()

    # Generate all outputs
    print("\n[INFO] Generating outputs …")
    plot_loss_curves(history)
    save_reconstructions(model)
    save_generated_samples(model)
    save_latent_space_pca(model)
    save_latent_interpolation(model)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    best_loss = min(history['total'][KL_ANNEAL_EPOCHS:]) if len(history['total']) > KL_ANNEAL_EPOCHS else history['total'][-1]
    print(f"  Best Model Total Loss : {best_loss:.4f} (loaded for inference)")
    print(f"  Final Epoch Recon Loss : {history['recon'][-1]:.4f}")
    print(f"  Final Epoch KL Loss    : {history['kl'][-1]:.4f}")
    print(f"\n  Saved files:")
    for root, _, files in os.walk(os.path.join(BASE_DIR, "outputs", "part1_vae")):
        for f in files:
            print(f"    • {os.path.join(root, f)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
