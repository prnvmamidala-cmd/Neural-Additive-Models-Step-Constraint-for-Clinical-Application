#!/usr/bin/env python3
"""
Vanilla NAM (Tanh) vs Threshold-Constrained NAM with HARD structural Step Activation
on synthetic Shock Index risk data.

NO derivative penalties, NO custom calculus losses — pure BCE + Adam.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Paths / hypers
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SEED = 42
N = 2000
EPOCHS = 150
BATCH = 64
LR = 1e-3
THRESHOLDS = (0.7, 1.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def generate_data(n: int = N, seed: int = SEED):
    """
    Shock Index ~ Uniform(0.4, 1.4)
    Age ~ Uniform(18, 80)
    p = clip(0.10 + 0.40*1{SI>=0.7} + 0.35*1{SI>=1.0} + 0.0015*(Age-49) + N(0,0.05^2), 0.01, 0.99)
    y ~ Bernoulli(p)
    """
    rng = np.random.default_rng(seed)
    si = rng.uniform(0.4, 1.4, size=n).astype(np.float32)
    age = rng.uniform(18.0, 80.0, size=n).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=n).astype(np.float32)
    p = (
        0.10
        + 0.40 * (si >= 0.7).astype(np.float32)
        + 0.35 * (si >= 1.0).astype(np.float32)
        + 0.0015 * (age - 49.0)
        + noise
    )
    p = np.clip(p, 0.01, 0.99)
    y = rng.binomial(1, p).astype(np.float32)
    return si, age, y, p


DATA_FORMULA = (
    "SI~U(0.4,1.4); Age~U(18,80); "
    "p=clip(0.10 + 0.40*1{SI>=0.7} + 0.35*1{SI>=1.0} + 0.0015*(Age-49) + N(0,0.05^2), 0.01, 0.99); "
    "y~Bernoulli(p)"
)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class FeatureNet(nn.Module):
    """Vanilla feature subnet: Linear(1,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HardStepActivation(nn.Module):
    """
    Hard structural step activation with three learnable scalar levels.
    Boolean masks only — NO soft approximations.

        f(x) = b0  if x < 0.7
             = b1  if 0.7 <= x < 1.0
             = b2  if x >= 1.0
    """

    def __init__(self, t0: float = 0.7, t1: float = 1.0):
        super().__init__()
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.b0 = nn.Parameter(torch.zeros(()))
        self.b1 = nn.Parameter(torch.zeros(()))
        self.b2 = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,) or (batch, 1)
        x = x.view(-1)
        mask0 = x < self.t0
        mask1 = (x >= self.t0) & (x < self.t1)
        mask2 = x >= self.t1
        out = torch.zeros_like(x)
        out = torch.where(mask0, self.b0.expand_as(x), out)
        out = torch.where(mask1, self.b1.expand_as(x), out)
        out = torch.where(mask2, self.b2.expand_as(x), out)
        return out


class VanillaNAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_si = FeatureNet()
        self.f_age = FeatureNet()

    def forward(self, si: torch.Tensor, age: torch.Tensor):
        # return logits (pre-sigmoid) for BCEWithLogitsLoss
        return self.f_si(si) + self.f_age(age)

    def contributions(self, si: torch.Tensor, age: torch.Tensor):
        return self.f_si(si), self.f_age(age)


class ThresholdConstrainedNAM(nn.Module):
    """Age: Tanh MLP. Shock Index: hard piecewise-constant step module."""

    def __init__(self):
        super().__init__()
        self.f_si = HardStepActivation(t0=THRESHOLDS[0], t1=THRESHOLDS[1])
        self.f_age = FeatureNet()

    def forward(self, si: torch.Tensor, age: torch.Tensor):
        return self.f_si(si) + self.f_age(age)

    def contributions(self, si: torch.Tensor, age: torch.Tensor):
        return self.f_si(si), self.f_age(age)


# -----------------------------------------------------------------------------
# Training / eval
# -----------------------------------------------------------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_si, val_age, val_y, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss for stability; documented as BCE
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_bce": [], "val_bce": [], "val_auroc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, n_seen = 0.0, 0
        for si_b, age_b, y_b in train_loader:
            si_b = si_b.to(DEVICE)
            age_b = age_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            opt.zero_grad()
            logits = model(si_b, age_b)
            loss = criterion(logits, y_b)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y_b.size(0)
            n_seen += y_b.size(0)
        train_bce = total_loss / n_seen

        model.eval()
        with torch.no_grad():
            val_logits = model(val_si.to(DEVICE), val_age.to(DEVICE))
            val_bce = criterion(val_logits, val_y.to(DEVICE)).item()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            try:
                val_auroc = float(roc_auc_score(val_y.numpy(), val_prob))
            except ValueError:
                val_auroc = float("nan")

        history["train_bce"].append(train_bce)
        history["val_bce"].append(val_bce)
        history["val_auroc"].append(val_auroc)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1:3d}/{epochs}  "
                f"train_bce={train_bce:.4f}  val_bce={val_bce:.4f}  val_auroc={val_auroc:.4f}"
            )

    return history


@torch.no_grad()
def evaluate(model: nn.Module, si, age, y):
    model.eval()
    logits = model(si.to(DEVICE), age.to(DEVICE))
    bce = nn.BCEWithLogitsLoss()(logits, y.to(DEVICE)).item()
    prob = torch.sigmoid(logits).cpu().numpy()
    auroc = float(roc_auc_score(y.numpy(), prob))
    return {"auroc": auroc, "bce": bce, "prob": prob, "logits": logits.cpu()}


def center_contribution(vals: np.ndarray) -> np.ndarray:
    """Center subnetwork contribution (mean-zero) for interpretability plots."""
    return vals - vals.mean()


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_main_figure(
    vanilla: VanillaNAM,
    constrained: ThresholdConstrainedNAM,
    metrics_v: dict,
    metrics_c: dict,
    out_path: Path,
):
    """
    Publication figure: SI on x, centered subnetwork contribution on y;
    vanilla smooth vs constrained hard steps; dashed lines at 0.7 and 1.0;
    AUROC annotation. 300 DPI.
    """
    si_grid = np.linspace(0.4, 1.4, 500).astype(np.float32)
    si_t = torch.from_numpy(si_grid).view(-1, 1)
    # age dummy (not used for SI contribution)
    age_dummy = torch.zeros_like(si_t)

    vanilla.eval()
    constrained.eval()
    with torch.no_grad():
        f_v, _ = vanilla.contributions(si_t.to(DEVICE), age_dummy.to(DEVICE))
        f_c, _ = constrained.contributions(si_t.to(DEVICE), age_dummy.to(DEVICE))
        f_v = center_contribution(f_v.cpu().numpy())
        f_c = center_contribution(f_c.cpu().numpy())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(si_grid, f_v, color="#1f77b4", lw=2.2, label=f"Vanilla NAM (Tanh)  AUROC={metrics_v['auroc']:.3f}")
    ax.plot(
        si_grid,
        f_c,
        color="#d62728",
        lw=2.4,
        label=f"Threshold-Constrained NAM (Hard Step)  AUROC={metrics_c['auroc']:.3f}",
    )
    ax.axvline(0.7, color="gray", ls="--", lw=1.3, label="Clinical thresholds (0.7, 1.0)")
    ax.axvline(1.0, color="gray", ls="--", lw=1.3)

    b0 = float(constrained.f_si.b0.detach().cpu())
    b1 = float(constrained.f_si.b1.detach().cpu())
    b2 = float(constrained.f_si.b2.detach().cpu())
    # centered step levels for annotation
    levels = np.array([b0, b1, b2], dtype=np.float64)
    levels_c = levels - levels.mean()  # approximate: actual plot centers over dense grid density

    ax.set_xlabel("Shock Index (SI)", fontsize=12)
    ax.set_ylabel("Centered SI subnetwork contribution  $f_{SI}(x)$", fontsize=12)
    ax.set_title(
        "Vanilla NAM vs Hard Step-Activation NAM\n"
        "(BCE only — no derivative / TV penalties)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.4)

    text = (
        f"Learned step levels (raw):  b0={b0:.3f},  b1={b1:.3f},  b2={b2:.3f}\n"
        f"Val AUROC — Vanilla: {metrics_v['auroc']:.4f}   Constrained: {metrics_c['auroc']:.4f}"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="0.7"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_roc(y_true, prob_v, prob_c, auroc_v, auroc_c, out_path: Path):
    fpr_v, tpr_v, _ = roc_curve(y_true, prob_v)
    fpr_c, tpr_c, _ = roc_curve(y_true, prob_c)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_v, tpr_v, lw=2, label=f"Vanilla NAM (AUROC={auroc_v:.3f})")
    ax.plot(fpr_c, tpr_c, lw=2, label=f"Hard Step NAM (AUROC={auroc_c:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_training_curves(hist_v, hist_c, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(hist_v["train_bce"]) + 1)
    axes[0].plot(epochs, hist_v["val_bce"], label="Vanilla val BCE")
    axes[0].plot(epochs, hist_c["val_bce"], label="Constrained val BCE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val BCE")
    axes[0].set_title("Validation BCE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist_v["val_auroc"], label="Vanilla AUROC")
    axes[1].plot(epochs, hist_c["val_auroc"], label="Constrained AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_event_rate(si, y, out_path: Path):
    bins = np.linspace(0.4, 1.4, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rates = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (si >= lo) & (si < hi)
        rates.append(y[m].mean() if m.any() else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, rates, width=(bins[1] - bins[0]) * 0.9, color="#4c72b0", alpha=0.85, edgecolor="k", lw=0.4)
    ax.axvline(0.7, color="gray", ls="--", lw=1.3)
    ax.axvline(1.0, color="gray", ls="--", lw=1.3)
    ax.set_xlabel("Shock Index")
    ax.set_ylabel("Empirical event rate")
    ax.set_title("Data: event rate by Shock Index bin")
    ax.set_xlim(0.4, 1.4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    set_seed(SEED)

    si, age, y, p = generate_data(N, SEED)
    print(f"Data: n={N}, event_rate={y.mean():.3f}")

    # 80/20 stratified split
    idx = np.arange(N)
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=y.astype(int)
    )

    def to_t(a):
        return torch.from_numpy(a).float()

    train_si = to_t(si[train_idx]).view(-1, 1)
    train_age = to_t(age[train_idx]).view(-1, 1)
    train_y = to_t(y[train_idx])
    val_si = to_t(si[val_idx]).view(-1, 1)
    val_age = to_t(age[val_idx]).view(-1, 1)
    val_y = to_t(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(train_si, train_age, train_y),
        batch_size=BATCH,
        shuffle=True,
        drop_last=False,
    )

    # --- Model 1: Vanilla NAM ---
    print("\n=== Training Vanilla NAM (Tanh) ===")
    set_seed(SEED)
    vanilla = VanillaNAM()
    hist_v = train_model(vanilla, train_loader, val_si, val_age, val_y)

    # --- Model 2: Threshold-Constrained NAM ---
    print("\n=== Training Threshold-Constrained NAM (Hard Step) ===")
    set_seed(SEED)
    constrained = ThresholdConstrainedNAM()
    hist_c = train_model(constrained, train_loader, val_si, val_age, val_y)

    # Evaluate
    metrics_v = evaluate(vanilla, val_si, val_age, val_y)
    metrics_c = evaluate(constrained, val_si, val_age, val_y)

    b0 = float(constrained.f_si.b0.detach().cpu())
    b1 = float(constrained.f_si.b1.detach().cpu())
    b2 = float(constrained.f_si.b2.detach().cpu())

    print("\n=== Results ===")
    print(f"Vanilla AUROC:     {metrics_v['auroc']:.4f}  val BCE: {metrics_v['bce']:.4f}")
    print(f"Constrained AUROC: {metrics_c['auroc']:.4f}  val BCE: {metrics_c['bce']:.4f}")
    print(f"Learned steps: b0={b0:.4f}, b1={b1:.4f}, b2={b2:.4f}")
    print(f"AUROC delta: {abs(metrics_v['auroc'] - metrics_c['auroc']):.4f} (expect within ~0.05)")

    # Verify hard piecewise shape
    si_grid = np.linspace(0.4, 1.4, 1000).astype(np.float32)
    with torch.no_grad():
        f_grid = constrained.f_si(torch.from_numpy(si_grid).view(-1, 1).to(DEVICE)).cpu().numpy()
    # Unique values should be exactly {b0,b1,b2}
    unique = np.unique(np.round(f_grid, 6))
    print(f"Constrained unique f_SI values (rounded): {unique.tolist()}")
    # Check constancy within regions
    r0 = f_grid[si_grid < 0.7]
    r1 = f_grid[(si_grid >= 0.7) & (si_grid < 1.0)]
    r2 = f_grid[si_grid >= 1.0]
    assert np.allclose(r0, b0), "Region <0.7 not constant at b0"
    assert np.allclose(r1, b1), "Region [0.7,1.0) not constant at b1"
    assert np.allclose(r2, b2), "Region >=1.0 not constant at b2"
    print("Verified: constrained f_SI is EXACTLY piecewise constant with jumps only at 0.7 and 1.0")

    # Figures
    plot_main_figure(
        vanilla,
        constrained,
        metrics_v,
        metrics_c,
        ROOT / "clinical_step_activation_results.png",
    )
    plot_roc(
        val_y.numpy(),
        metrics_v["prob"],
        metrics_c["prob"],
        metrics_v["auroc"],
        metrics_c["auroc"],
        ROOT / "fig_roc.png",
    )
    plot_training_curves(hist_v, hist_c, ROOT / "fig_training_curves.png")
    plot_event_rate(si, y, ROOT / "fig_data_event_rate.png")

    # metrics.json
    metrics = {
        "hypers": {
            "seed": SEED,
            "n": N,
            "split": "80/20 stratified",
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": LR,
            "optimizer": "Adam",
            "loss": "BCE (BCEWithLogitsLoss)",
            "derivative_penalty": None,
            "lambda": None,
            "tv_loss": None,
            "device": str(DEVICE),
        },
        "data_formula": DATA_FORMULA,
        "vanilla_nam": {
            "architecture": "FeatureNet(SI)+FeatureNet(Age); each Linear(1,64)-Tanh-Linear(64,64)-Tanh-Linear(64,1); pred=sigmoid(sum)",
            "val_auroc": metrics_v["auroc"],
            "val_bce": metrics_v["bce"],
            "final_train_bce": hist_v["train_bce"][-1],
            "final_val_auroc_history": hist_v["val_auroc"][-1],
        },
        "threshold_constrained_nam": {
            "architecture": "HardStepActivation(SI; b0,b1,b2 at 0.7/1.0) + FeatureNet(Age); pred=sigmoid(sum)",
            "val_auroc": metrics_c["auroc"],
            "val_bce": metrics_c["bce"],
            "final_train_bce": hist_c["train_bce"][-1],
            "final_val_auroc_history": hist_c["val_auroc"][-1],
            "learned_step_levels": {"b0": b0, "b1": b1, "b2": b2},
            "thresholds": list(THRESHOLDS),
            "piecewise_verified": True,
        },
        "auroc_delta": abs(metrics_v["auroc"] - metrics_c["auroc"]),
        "auroc_similar_within_0_05": abs(metrics_v["auroc"] - metrics_c["auroc"]) <= 0.05,
    }

    metrics_path = ROOT / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
