"""
Unit Tests — Physics-Informed Neural Network
============================================
Run with:   pytest tests/test_pinn.py -v
Coverage:   pytest tests/test_pinn.py --cov=pinn_complete_starter --cov-report=term-missing
"""

import pytest
import torch
import numpy as np
import sys
import os

# ── Make parent directory importable ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinn_complete_starter import PhysicsInformedNN, generate_synthetic_data


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def small_model(device):
    """Lightweight model for fast unit tests."""
    model = PhysicsInformedNN(hidden_dim=32, num_layers=2, device=device)
    return model.to(device)


@pytest.fixture(scope="session")
def full_model(device):
    """Full-sized model as used in production training."""
    model = PhysicsInformedNN(hidden_dim=256, num_layers=5, device=device)
    return model.to(device)


@pytest.fixture(scope="session")
def small_data():
    """Minimal dataset — fast to generate, sufficient for shape checks."""
    return generate_synthetic_data(num_data=200, num_physics=400)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARCHITECTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestModelArchitecture:
    """Verify model construction, shapes, and parameter setup."""

    @pytest.mark.parametrize("hidden_dim,num_layers", [
        (32, 2), (64, 3), (128, 4), (256, 5),
    ])
    def test_model_instantiation(self, device, hidden_dim, num_layers):
        model = PhysicsInformedNN(hidden_dim=hidden_dim, num_layers=num_layers, device=device)
        assert model is not None

    def test_forward_pass_batch(self, small_model, device):
        batch_size = 64
        xt = torch.randn(batch_size, 2).to(device)
        u = small_model(xt)
        assert u.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {u.shape}"

    def test_forward_pass_single_point(self, small_model, device):
        xt = torch.tensor([[0.5, 1.0]]).to(device)
        u = small_model(xt)
        assert u.shape == (1, 1)
        assert not torch.isnan(u).any(), "Output should not be NaN"

    def test_learnable_physical_parameters(self, small_model):
        param_names = [n for n, _ in small_model.named_parameters()]
        assert "omega_0" in param_names, "omega_0 must be a learnable parameter"
        assert "zeta"    in param_names, "zeta must be a learnable parameter"

    def test_initial_omega0_value(self, small_model):
        assert abs(small_model.omega_0.item() - 1.0) < 1e-5

    def test_initial_zeta_value(self, small_model):
        assert abs(small_model.zeta.item() - 0.05) < 1e-5

    def test_output_always_finite(self, small_model, device):
        random_xt = torch.rand(200, 2).to(device)
        u = small_model(random_xt)
        assert torch.isfinite(u).all(), "Forward pass should never produce inf/nan"

    def test_full_model_parameter_count(self, full_model):
        n = sum(p.numel() for p in full_model.parameters())
        # 5-layer 256-dim network ≈ 330k–340k params (+2 physics params)
        assert 100_000 < n < 2_000_000, f"Unexpected parameter count: {n:,}"

    def test_output_changes_with_input(self, small_model, device):
        """Model must not output a constant for all inputs."""
        xt_a = torch.tensor([[0.1, 0.5], [0.9, 4.0]]).to(device)
        u_a = small_model(xt_a)
        assert not torch.allclose(u_a[0], u_a[1]), "Model output should vary with input"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PDE RESIDUAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPDEResidual:
    """Verify correctness of automatic-differentiation based PDE computation."""

    def test_residual_output_shape(self, small_model, device):
        n = 50
        x = torch.rand(n).to(device)
        t = torch.rand(n).to(device) * 5.0
        residual, u = small_model.pde_residual(x, t)
        assert residual.shape == (n, 1), f"Residual shape: expected ({n},1), got {residual.shape}"
        assert u.shape       == (n, 1), f"u shape: expected ({n},1), got {u.shape}"

    def test_residual_finite(self, small_model, device):
        x = torch.rand(30).to(device)
        t = torch.rand(30).to(device) * 5.0
        residual, u = small_model.pde_residual(x, t)
        assert torch.isfinite(residual).all(), "PDE residual must be finite"
        assert torch.isfinite(u).all(),        "Prediction u must be finite"

    def test_residual_scalar_input(self, small_model, device):
        """Single-point PDE computation must work."""
        x = torch.tensor([0.5]).to(device)
        t = torch.tensor([1.0]).to(device)
        residual, u = small_model.pde_residual(x, t)
        assert residual.shape == (1, 1)

    def test_residual_gradient_flows(self, small_model, device):
        """Loss computed from PDE residual must be differentiable."""
        x = torch.rand(20).to(device)
        t = torch.rand(20).to(device) * 5.0
        residual, _ = small_model.pde_residual(x, t)
        loss = torch.mean(residual ** 2)
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLossFunctions:
    """Verify boundary-condition and initial-condition loss behaviour."""

    def test_bc_loss_non_negative(self, small_model, device):
        t_bc = torch.linspace(0, 5, 50).to(device)
        x_bc = torch.zeros_like(t_bc)
        loss = small_model.boundary_condition_loss(x_bc, t_bc, bc_value=0.0)
        assert loss.item() >= 0.0

    def test_bc_loss_finite(self, small_model, device):
        t_bc = torch.rand(50).to(device) * 5.0
        x_bc = torch.zeros_like(t_bc)
        loss = small_model.boundary_condition_loss(x_bc, t_bc)
        assert torch.isfinite(loss)

    def test_ic_loss_non_negative(self, small_model, device):
        x_ic = torch.linspace(0, 1, 50).to(device)
        u_ic = 0.1 * torch.sin(torch.tensor(np.pi) * x_ic)
        loss = small_model.initial_condition_loss(x_ic, u_ic)
        assert loss.item() >= 0.0

    def test_ic_loss_finite(self, small_model, device):
        x_ic = torch.rand(50).to(device)
        u_ic = torch.rand(50).to(device)
        loss = small_model.initial_condition_loss(x_ic, u_ic)
        assert torch.isfinite(loss)

    def test_total_loss_backward(self, small_model, device):
        """Full multi-component loss must back-propagate."""
        # Zero previous grads
        for p in small_model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        x = torch.rand(20).to(device)
        t = torch.rand(20).to(device) * 5.0
        residual, _ = small_model.pde_residual(x, t)
        loss = torch.mean(residual ** 2)
        loss.backward()

        grads_exist = [
            p.grad is not None
            for p in small_model.parameters()
            if p.requires_grad
        ]
        assert all(grads_exist), "Not all parameters received gradients"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA GENERATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:
    """Validate synthetic dataset properties."""

    REQUIRED_KEYS = [
        "x_data", "t_data", "u_data",
        "x_physics", "t_physics",
        "x_bc", "t_bc",
        "x_ic", "u_ic",
    ]

    def test_required_keys_present(self, small_data):
        for key in self.REQUIRED_KEYS:
            assert key in small_data, f"Missing data key: '{key}'"

    def test_data_point_count(self, small_data):
        assert small_data["x_data"].shape[0]    == 200
        assert small_data["x_physics"].shape[0] == 400

    def test_spatial_range(self, small_data):
        assert small_data["x_data"].min().item() >= 0.0 - 1e-6
        assert small_data["x_data"].max().item() <= 1.0 + 1e-6

    def test_temporal_range(self, small_data):
        assert small_data["t_data"].min().item() >= 0.0 - 1e-6
        assert small_data["t_data"].max().item() <= 5.0 + 1e-6

    def test_ic_shape_consistency(self, small_data):
        assert small_data["x_ic"].shape == small_data["u_ic"].shape

    def test_all_tensors_finite(self, small_data):
        for key, val in small_data.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"Non-finite values in '{key}'"

    def test_ic_matches_analytical_at_t0(self, small_data):
        """u_ic must equal A·sin(πx) with A=0.1 everywhere on [0,1]."""
        x_ic = small_data["x_ic"].numpy()
        u_ic = small_data["u_ic"].numpy()
        expected = 0.1 * np.sin(np.pi * x_ic)
        np.testing.assert_allclose(u_ic, expected, atol=1e-5,
                                   err_msg="IC values don't match A·sin(πx)")

    def test_bc_x_is_zero(self, small_data):
        """Boundary condition points must be at x = 0."""
        x_bc = small_data["x_bc"].numpy()
        np.testing.assert_allclose(x_bc, 0.0, atol=1e-6,
                                   err_msg="BC x-values should all be 0")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PHYSICS VALIDATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsValidation:
    """Verify physical correctness of the governing equations."""

    def test_analytical_solution_satisfies_pde(self):
        """
        The analytical solution u(t) = A·exp(-ζω₀t)·cos(ω₀t) must satisfy
        the damped oscillator ODE numerically (central-difference check).
        """
        A, omega_0, zeta = 0.1, 1.0, 0.05
        t = np.linspace(0.05, 8.0, 2000)
        dt = t[1] - t[0]

        u    = A * np.exp(-zeta * omega_0 * t) * np.cos(omega_0 * t)
        du   = np.gradient(u,  dt)
        d2u  = np.gradient(du, dt)

        residual = d2u + 2 * zeta * omega_0 * du + omega_0**2 * u

        # Interior points only (gradient is less accurate at edges)
        max_res = np.abs(residual[20:-20]).max()
        assert max_res < 5e-3, (
            f"Analytical solution violates ODE (max residual = {max_res:.2e})"
        )

    def test_bc_x0_computation_runs(self, small_model, device):
        """BC loss at x=0 must return a valid scalar."""
        t_bc = torch.linspace(0, 5, 100).to(device)
        x_bc = torch.zeros_like(t_bc)
        xt   = torch.stack([x_bc, t_bc], dim=1)
        u    = small_model(xt)
        assert u.shape == (100, 1)
        assert torch.isfinite(u).all()

    def test_damping_ratio_range_valid(self, small_model):
        """Damping ratio must stay in a physically meaningful range."""
        zeta = small_model.zeta.item()
        # Initialised at 0.05; should not go negative or > 10 during valid training
        assert -0.5 < zeta < 10.0, f"zeta = {zeta} seems physically unreasonable"


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
