"""
Heat Equation PINN — Extension Experiment
==========================================
Demonstrates PINN's generality by solving the 1-D heat (diffusion) equation:

        ∂u/∂t = α ∂²u/∂x²   for x ∈ [0,1],  t ∈ [0,1]

Boundary conditions (Dirichlet):
        u(0, t) = u(1, t) = 0

Initial condition:
        u(x, 0) = sin(πx)

Analytical solution:
        u(x, t) = sin(πx) · exp(-α π² t)

The PINN learns to both fit the PDE *and* discover the unknown thermal
diffusivity α from sparse, noisy observations — an *inverse problem*.

Run:
    python experiments/heat_equation_pinn.py
"""

import sys
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HeatEquationPINN(nn.Module):
    """
    Physics-Informed Neural Network for the 1-D Heat Equation.

    Network maps (x, t) → u(x, t) (temperature / concentration).
    Learns thermal diffusivity α as a latent physical parameter.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 device: str = "cpu") -> None:
        super().__init__()
        self._device = device

        # Build MLP: (x, t) → u
        layers = []
        in_dim = 2
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        # Learnable thermal diffusivity α (constrained positive via softplus later)
        self.log_alpha = nn.Parameter(torch.tensor([np.log(0.5)], device=device))

        # Xavier init
        for name, p in self.network.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(p)

    @property
    def alpha(self) -> torch.Tensor:
        """Thermal diffusivity α — guaranteed positive via softplus."""
        return torch.nn.functional.softplus(self.log_alpha)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.network(xt)

    # ── PDE residual ─────────────────────────────────────────────────────────

    def pde_residual(self, x: torch.Tensor,
                     t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute heat-equation residual via automatic differentiation:

            R = ∂u/∂t − α ∂²u/∂x²    (should be ≈ 0 everywhere)
        """
        x = x.unsqueeze(1) if x.dim() == 1 else x
        t = t.unsqueeze(1) if t.dim() == 1 else t

        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)

        u = self.forward(xt)

        # First-order derivatives
        grads = torch.autograd.grad(
            u, xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True,
        )[0]
        u_x = grads[:, 0:1]   # ∂u/∂x
        u_t = grads[:, 1:2]   # ∂u/∂t

        # Second-order spatial derivative ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x, xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0][:, 0:1]

        residual = u_t - self.alpha * u_xx
        return residual, u


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_heat_data(
    alpha_true: float = 0.1,
    num_data: int = 3_000,
    num_physics: int = 8_000,
    noise_std: float = 0.002,
) -> dict:
    """
    Generate synthetic training data from the analytical solution.

    Args:
        alpha_true  : True thermal diffusivity [m²/s].
        num_data    : Number of supervised observation points.
        num_physics : Number of collocation points for PDE constraint.
        noise_std   : Gaussian noise added to observations.

    Returns:
        Dictionary of torch.Tensor training arrays.
    """
    # ── Observations (sparse + noisy) ────────────────────────────────────────
    x_d = np.random.uniform(0, 1, num_data)
    t_d = np.random.uniform(0, 1, num_data)
    u_exact = np.sin(np.pi * x_d) * np.exp(-alpha_true * np.pi**2 * t_d)
    u_noisy = u_exact + np.random.normal(0, noise_std, u_exact.shape)

    # ── Physics collocation points ────────────────────────────────────────────
    x_p = np.random.uniform(0, 1, num_physics)
    t_p = np.random.uniform(0, 1, num_physics)

    # ── Boundary conditions: u(0,t) = u(1,t) = 0 ────────────────────────────
    t_bc      = np.random.uniform(0, 1, 1_000)
    x_bc_left = np.zeros_like(t_bc)    # x = 0
    x_bc_right= np.ones_like(t_bc)     # x = 1

    # ── Initial condition: u(x,0) = sin(πx) ─────────────────────────────────
    x_ic = np.linspace(0, 1, 1_000)
    u_ic = np.sin(np.pi * x_ic)

    def _t(arr):
        return torch.from_numpy(arr.astype(np.float32))

    return {
        "x_data":      _t(x_d),
        "t_data":      _t(t_d),
        "u_data":      _t(u_noisy),
        "x_physics":   _t(x_p),
        "t_physics":   _t(t_p),
        "t_bc":        _t(t_bc),
        "x_bc_left":   _t(x_bc_left),
        "x_bc_right":  _t(x_bc_right),
        "x_ic":        _t(x_ic),
        "u_ic":        _t(u_ic),
        "alpha_true":  alpha_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_heat_pinn(
    model: HeatEquationPINN,
    data: dict,
    num_epochs:     int   = 4_000,
    device:         str   = "cpu",
    lambda_physics: float = 1.0,
    lambda_data:    float = 10.0,
) -> tuple[list, list]:
    """
    Train the Heat-Equation PINN with composite loss.

        L = λ_phys · ‖R‖² + λ_data · ‖u − u_obs‖² + ‖u_BC‖² + ‖u_IC − sin(πx)‖²

    Returns:
        loss_history  : Total loss per epoch.
        alpha_history : Learned α per epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.85)

    # Move to device
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(device)

    loss_history  = []
    alpha_history = []

    BAR = "=" * 60
    print(f"\n{BAR}")
    print("  Heat Equation PINN — Inverse Problem")
    print(f"  Target α (thermal diffusivity) = {data['alpha_true']:.4f} m²/s")
    print(f"  Device : {device}  |  Epochs : {num_epochs}")
    print(f"{BAR}\n")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # ── 1. Physics loss ───────────────────────────────────────────────
        residual, _ = model.pde_residual(data["x_physics"], data["t_physics"])
        loss_physics = torch.mean(residual**2)

        # ── 2. Data loss ──────────────────────────────────────────────────
        xt_d   = torch.stack([data["x_data"], data["t_data"]], dim=1)
        u_pred = model(xt_d)
        loss_data = torch.mean((u_pred - data["u_data"].unsqueeze(1))**2)

        # ── 3. Boundary condition loss: u(0,t)=0 and u(1,t)=0 ────────────
        xt_left  = torch.stack([data["x_bc_left"],  data["t_bc"]], dim=1)
        xt_right = torch.stack([data["x_bc_right"], data["t_bc"]], dim=1)
        loss_bc  = torch.mean(model(xt_left)**2) + torch.mean(model(xt_right)**2)

        # ── 4. Initial condition loss: u(x,0) = sin(πx) ──────────────────
        t_zero   = torch.zeros_like(data["x_ic"])
        xt_ic    = torch.stack([data["x_ic"], t_zero], dim=1)
        u_ic_pred = model(xt_ic)
        loss_ic  = torch.mean((u_ic_pred - data["u_ic"].unsqueeze(1))**2)

        # ── Total ─────────────────────────────────────────────────────────
        loss_total = (lambda_physics * loss_physics
                    + lambda_data    * loss_data
                    + loss_bc + loss_ic)

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        alpha_learned = model.alpha.item()
        loss_history.append(loss_total.item())
        alpha_history.append(alpha_learned)

        if (epoch + 1) % 400 == 0 or epoch == 0:
            pct_err = abs(alpha_learned - data["alpha_true"]) / data["alpha_true"] * 100
            print(
                f"  Epoch {epoch+1:5d}/{num_epochs} | "
                f"Loss: {loss_total.item():.3e} | "
                f"α: {alpha_learned:.5f}  (true: {data['alpha_true']:.4f}, "
                f"err: {pct_err:.2f}%)"
            )

    print(f"\n{BAR}")
    print(f"  ✅ Training Complete!")
    print(f"  Learned  α = {model.alpha.item():.6f}")
    print(f"  True     α = {data['alpha_true']:.6f}")
    print(f"  Error      = {abs(model.alpha.item() - data['alpha_true'])*100:.3f}%")
    print(f"{BAR}\n")

    return loss_history, alpha_history


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_heat_results(
    model: HeatEquationPINN,
    data: dict,
    loss_history: list,
    alpha_history: list,
    device: str = "cpu",
) -> None:
    """6-panel figure: loss, α convergence, PINN prediction,
       analytical solution, absolute error, and spatial profiles."""

    alpha_true    = data["alpha_true"]
    alpha_learned = model.alpha.item()

    x_test = np.linspace(0, 1, 120)
    t_test = np.linspace(0, 1, 160)
    X, T   = np.meshgrid(x_test, t_test)

    x_torch = torch.from_numpy(X.flatten().astype(np.float32)).to(device)
    t_torch = torch.from_numpy(T.flatten().astype(np.float32)).to(device)
    xt_test = torch.stack([x_torch, t_torch], dim=1)

    with torch.no_grad():
        U_pinn = model(xt_test).cpu().numpy().reshape(X.shape)

    U_exact = np.sin(np.pi * X) * np.exp(-alpha_true * np.pi**2 * T)
    error   = np.abs(U_pinn - U_exact)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0d1b2a")
    for ax in axes.flat:
        ax.set_facecolor("#0a0e1a")
        ax.tick_params(colors="#90caf9")
        ax.xaxis.label.set_color("#4fc3f7")
        ax.yaxis.label.set_color("#4fc3f7")
        ax.title.set_color("#e8f0fe")
        for spine in ax.spines.values():
            spine.set_edgecolor("rgba(79,195,247,0.3)")

    CMAP_MAIN = "RdBu_r"
    CMAP_ERR  = "YlOrRd"

    # 1. Loss curve
    ax = axes[0, 0]
    ax.semilogy(loss_history, color="#4fc3f7", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.grid(alpha=0.2)

    # 2. α convergence
    ax = axes[0, 1]
    ax.plot(alpha_history, color="#7c4dff", lw=2, label=f"Learned α")
    ax.axhline(y=alpha_true, color="#ef9a9a", ls="--", lw=2,
               label=f"True α = {alpha_true}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Thermal diffusivity α [m²/s]")
    ax.set_title("Learned vs True α")
    legend = ax.legend(facecolor="#0d1b2a", labelcolor="#e8f0fe")
    ax.grid(alpha=0.2)

    # 3. PINN prediction
    ax = axes[0, 2]
    cs = ax.contourf(X, T, U_pinn, levels=25, cmap=CMAP_MAIN)
    plt.colorbar(cs, ax=ax, label="u")
    ax.set_xlabel("x [m]"); ax.set_ylabel("t [s]")
    ax.set_title(f"PINN Prediction  (α={alpha_learned:.4f})")

    # 4. Analytical solution
    ax = axes[1, 0]
    cs = ax.contourf(X, T, U_exact, levels=25, cmap=CMAP_MAIN)
    plt.colorbar(cs, ax=ax, label="u")
    ax.set_xlabel("x [m]"); ax.set_ylabel("t [s]")
    ax.set_title(f"Analytical Solution  (α={alpha_true})")

    # 5. Error map
    ax = axes[1, 1]
    cs = ax.contourf(X, T, error, levels=25, cmap=CMAP_ERR)
    plt.colorbar(cs, ax=ax, label="Error")
    ax.set_xlabel("x [m]"); ax.set_ylabel("t [s]")
    ax.set_title(f"Absolute Error  (MAE = {error.mean():.2e})")

    # 6. Spatial profiles at multiple times
    ax = axes[1, 2]
    times_show = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_grad  = plt.cm.plasma(np.linspace(0, 1, len(times_show)))
    for t_val, col in zip(times_show, cmap_grad):
        u_snap = np.sin(np.pi * x_test) * np.exp(-alpha_true * np.pi**2 * t_val)
        ax.plot(x_test, u_snap, color=col, lw=2, label=f"t={t_val:.1f}s")
    ax.set_xlabel("x [m]"); ax.set_ylabel("Temperature u")
    ax.set_title("Analytical Profiles vs Time")
    ax.legend(fontsize=7, facecolor="#0d1b2a", labelcolor="#e8f0fe")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"Heat Equation PINN  |  Learned α = {alpha_learned:.5f}  "
        f"(true: {alpha_true})  |  Error: {abs(alpha_learned-alpha_true)*100:.3f}%",
        fontsize=13, fontweight="bold", color="#e8f0fe",
    )
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heat_pinn_results_{ts}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Saved: {filename}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full pipeline: data → model → train → evaluate → visualize.

    This experiment shows that the *same* PINN framework used for
    the damped wave equation can solve a completely different PDE
    (parabolic vs. hyperbolic) just by swapping the residual function.
    """
    ALPHA_TRUE  = 0.1    # Target thermal diffusivity [m²/s]
    NUM_EPOCHS  = 4_000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    print(f"\n🌡️  Heat Equation:  ∂u/∂t = α ∂²u/∂x²")
    print(f"   Inverse problem: learn α = {ALPHA_TRUE} from noisy obs.")

    # Data
    data = generate_heat_data(alpha_true=ALPHA_TRUE,
                              num_data=3_000, num_physics=8_000)

    # Model
    model = HeatEquationPINN(hidden_dim=128, num_layers=4, device=device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters : {n_params:,}")

    # Train
    loss_history, alpha_history = train_heat_pinn(
        model, data,
        num_epochs=NUM_EPOCHS,
        device=device,
        lambda_physics=1.0,
        lambda_data=10.0,
    )

    # Visualize
    plot_heat_results(model, data, loss_history, alpha_history, device)

    # Save
    save_path = "heat_pinn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved: {save_path}")

    return model, loss_history


if __name__ == "__main__":
    main()
