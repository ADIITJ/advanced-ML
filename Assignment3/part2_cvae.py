# -*- coding: utf-8 -*-
"""Part 2 — Conditional Variational Autoencoder (CVAE) on MNIST.

Production-grade implementation of a fully-connected CVAE that conditions
on class labels via one-hot concatenation. Features:
- Log-variance parameterization for training stability
- KL annealing (β-VAE schedule) to prevent posterior collapse
- Mixed-precision training (AMP) for efficient GPU utilization
- Conditional generation: produce specific digits on demand
- Comprehensive output pipeline: loss curves, reconstructions,
  conditional grids, and digit-to-digit latent interpolation

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
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

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
INPUT_DIM: int = 784          # 28 × 28 flattened
NUM_CLASSES: int = 10         # MNIST digits 0-9
H_DIM: int = 400              # hidden layer width
Z_DIM: int = 20               # latent dimension
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 120
LR: float = 1e-3
KL_ANNEAL_EPOCHS: int = 10    # linearly ramp β from 0 → 1
EARLY_STOP_PATIENCE: int = 10  # stop if no improvement for N consecutive epochs

# ──────────────────────────────────────────────────────────────────────
# 4. Paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = r"D:\Projects2.0\Last Days Work\AML\Assignment3"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CKPT_DIR = os.path.join(BASE_DIR, "outputs", "part2_cvae", "checkpoints")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "part2_cvae", "plots")
GEN_DIR = os.path.join(BASE_DIR, "outputs", "part2_cvae", "generated")

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
# 7. Utilities
# ──────────────────────────────────────────────────────────────────────
def one_hot(labels: torch.Tensor, num_classes: int = NUM_CLASSES,
            device: str = "cuda") -> torch.Tensor:
    """Create one-hot encoded vectors from integer labels.

    Args:
        labels: Integer class labels, shape (B,).
        num_classes: Total number of classes.
        device: Target device for the output tensor.

    Returns:
        One-hot tensor of shape (B, num_classes).
    """
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(
        1, labels.unsqueeze(1), 1
    )


# ──────────────────────────────────────────────────────────────────────
# 8. Model
# ──────────────────────────────────────────────────────────────────────
class CVAE(nn.Module):
    """Conditional Variational Autoencoder with log-variance parameterisation.

    Conditions on class labels by concatenating a one-hot vector to both
    the encoder input (x || y) and the decoder input (z || y).

    Architecture (linear layers only, per assignment spec):
        Encoder: (784 + 10) → H_DIM (ReLU) → μ (Z_DIM), log_var (Z_DIM)
        Decoder: (Z_DIM + 10) → H_DIM (ReLU) → 784 (raw logits; sigmoid applied at inference)
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 num_classes: int = NUM_CLASSES,
                 h_dim: int = H_DIM, z_dim: int = Z_DIM) -> None:
        """Initialise encoder and decoder layers.

        Args:
            input_dim: Flattened image dimensionality (784 for MNIST).
            num_classes: Number of conditioning classes (10 for MNIST).
            h_dim: Hidden layer width.
            z_dim: Latent space dimensionality.
        """
        super().__init__()
        # Encoder: input = x (784) + one_hot(y) (10) = 794
        self.enc_hidden = nn.Linear(input_dim + num_classes, h_dim)
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_log_var = nn.Linear(h_dim, z_dim)
        # Decoder: input = z (Z_DIM) + one_hot(y) (10)
        self.dec_hidden = nn.Linear(z_dim + num_classes, h_dim)
        self.dec_out = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor,
               y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and label to latent distribution parameters.

        Args:
            x: Flattened input images, shape (B, 784).
            y_onehot: One-hot encoded labels, shape (B, 10).

        Returns:
            mu: Mean of approximate posterior, shape (B, Z_DIM).
            log_var: Log-variance of approximate posterior, shape (B, Z_DIM).
        """
        inp = torch.cat([x, y_onehot], dim=1)  # (B, 794)
        h = self.relu(self.enc_hidden(inp))
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

    def decode(self, z: torch.Tensor,
               y_onehot: torch.Tensor) -> torch.Tensor:
        """Decode latent vector conditioned on class label to raw logits.

        Args:
            z: Latent vector, shape (B, Z_DIM).
            y_onehot: One-hot encoded labels, shape (B, 10).

        Returns:
            Logits, shape (B, 784). Apply sigmoid for pixel values.
        """
        inp = torch.cat([z, y_onehot], dim=1)  # (B, Z_DIM + 10)
        h = self.relu(self.dec_hidden(inp))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor,
                y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Args:
            x: Flattened input images, shape (B, 784).
            y_onehot: One-hot encoded labels, shape (B, 10).

        Returns:
            x_hat_logits: Reconstructed image logits, shape (B, 784).
            mu: Posterior mean.
            log_var: Posterior log-variance.
        """
        mu, log_var = self.encode(x, y_onehot)
        z = self.reparameterize(mu, log_var)
        x_hat_logits = self.decode(z, y_onehot)
        return x_hat_logits, mu, log_var


# ──────────────────────────────────────────────────────────────────────
# 9. Loss function
# ──────────────────────────────────────────────────────────────────────
# BCEWithLogitsLoss = sigmoid + BCE fused — autocast-safe & numerically stable
bce_logits_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")


def cvae_loss(x: torch.Tensor, x_hat_logits: torch.Tensor,
              mu: torch.Tensor, log_var: torch.Tensor,
              beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CVAE ELBO loss = Reconstruction + β · KL divergence.

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
# 10. Training
# ──────────────────────────────────────────────────────────────────────
def train() -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the CVAE with mixed-precision, KL annealing, early stopping, and checkpointing.

    Returns:
        model: Trained CVAE model (best checkpoint loaded).
        history: Dictionary of per-epoch loss lists.
    """
    model = CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    history: Dict[str, List[float]] = {
        "total": [], "recon": [], "kl": []
    }
    best_loss = float("inf")
    patience_counter: int = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}",
                     leave=False)
        for i, (x, y) in enumerate(loop):
            try:
                x = x.view(-1, INPUT_DIM).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_onehot = one_hot(y, device=device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    x_hat, mu, log_var = model(x, y_onehot)
                    loss, recon, kl = cvae_loss(x, x_hat, mu, log_var, beta)

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

                del x, y, y_onehot, x_hat, mu, log_var, loss, recon, kl
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
# 11. Visualization helpers
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

    fig.suptitle("CVAE Training Curves", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_reconstructions(model: nn.Module) -> None:
    """Save a grid of original vs. reconstructed test images.

    Args:
        model: Trained CVAE model.
    """
    model.eval()
    x_test, y_test = next(iter(test_loader))
    x_test = x_test[:8].to(device)
    y_test = y_test[:8].to(device)
    x_flat = x_test.view(-1, INPUT_DIM)
    y_onehot = one_hot(y_test, device=device)

    with torch.no_grad():
        x_hat_logits, _, _ = model(x_flat, y_onehot)
    x_hat = torch.sigmoid(x_hat_logits).view(-1, 1, 28, 28)

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


def save_conditional_grid(model: nn.Module) -> None:
    """Generate 8 samples per digit (0-9) and save as a 10×8 grid.

    Args:
        model: Trained CVAE model.
    """
    model.eval()
    num_samples_per_digit = 8
    all_samples = []

    with torch.no_grad():
        for digit in range(NUM_CLASSES):
            z = torch.randn(num_samples_per_digit, Z_DIM, device=device)
            labels = torch.full((num_samples_per_digit,), digit,
                                dtype=torch.long, device=device)
            y_onehot = one_hot(labels, device=device)
            samples = torch.sigmoid(model.decode(z, y_onehot)).view(-1, 1, 28, 28)
            all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0).cpu()  # (80, 1, 28, 28)
    grid = make_grid(all_samples, nrow=num_samples_per_digit, padding=2)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title("Conditional Generation (rows = digits 0–9, cols = samples)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add digit labels on the left
    img_h = 28 + 2  # image height + padding
    for digit in range(NUM_CLASSES):
        y_pos = 2 + digit * img_h + img_h / 2
        ax.text(-5, y_pos, str(digit), fontsize=12, fontweight="bold",
                ha="right", va="center", color="#333333")

    path = os.path.join(GEN_DIR, "conditional_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def save_interpolation_1_to_7(model: nn.Module) -> None:
    """Interpolate in latent space between digit 1 and digit 7.

    Uses fixed class conditioning to show smooth morphing while
    transitioning the label from 1 → 7.

    Args:
        model: Trained CVAE model.
    """
    model.eval()

    # Find one example of digit 1 and one of digit 7 from the test set
    sample_1, sample_7 = None, None
    for x, y in test_dataset:
        if y == 1 and sample_1 is None:
            sample_1 = x
        elif y == 7 and sample_7 is None:
            sample_7 = x
        if sample_1 is not None and sample_7 is not None:
            break

    img_1 = sample_1.view(1, INPUT_DIM).to(device)
    img_7 = sample_7.view(1, INPUT_DIM).to(device)

    label_1 = torch.tensor([1], device=device)
    label_7 = torch.tensor([7], device=device)
    y_oh_1 = one_hot(label_1, device=device)
    y_oh_7 = one_hot(label_7, device=device)

    with torch.no_grad():
        mu_1, _ = model.encode(img_1, y_oh_1)
        mu_7, _ = model.encode(img_7, y_oh_7)

    num_steps = 10
    interp_imgs = []
    with torch.no_grad():
        for step, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z_interp = (1 - alpha) * mu_1 + alpha * mu_7
            # Interpolate the conditioning label as well
            y_interp = (1 - alpha) * y_oh_1 + alpha * y_oh_7
            decoded = torch.sigmoid(model.decode(z_interp, y_interp)).view(1, 1, 28, 28).cpu()
            interp_imgs.append(decoded)

    strip = torch.cat(interp_imgs, dim=0)
    grid = make_grid(strip, nrow=num_steps, padding=2)

    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.set_title("Latent Interpolation: digit 1 → digit 7",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    path = os.path.join(GEN_DIR, "interpolation_1_to_7.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────
# 12. Main entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run full CVAE pipeline: train → evaluate → generate outputs."""
    print("=" * 60)
    print("  PART 2 — Conditional VAE (CVAE) on MNIST")
    print("=" * 60)
    print(f"[INFO] Using device: {torch.cuda.get_device_name(device)}")

    # Train
    model, history = train()

    # Generate all outputs
    print("\n[INFO] Generating outputs …")
    plot_loss_curves(history)
    save_reconstructions(model)
    save_conditional_grid(model)
    save_interpolation_1_to_7(model)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    best_loss = min(history['total'][KL_ANNEAL_EPOCHS:]) if len(history['total']) > KL_ANNEAL_EPOCHS else history['total'][-1]
    print(f"  Best Model Total Loss : {best_loss:.4f} (loaded for inference)")
    print(f"  Final Epoch Recon Loss : {history['recon'][-1]:.4f}")
    print(f"  Final Epoch KL Loss    : {history['kl'][-1]:.4f}")
    print(f"\n  Saved files:")
    for root, _, files in os.walk(os.path.join(BASE_DIR, "outputs", "part2_cvae")):
        for f in files:
            print(f"    • {os.path.join(root, f)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
