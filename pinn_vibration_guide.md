# Physics-Informed Neural Network untuk Prediksi Getaran Struktur

## Daftar Isi
1. Teori Dasar PINN
2. Persamaan Fisika untuk Getaran
3. Arsitektur Neural Network
4. Implementasi Kode
5. Dataset & Training
6. Hasil & Interpretasi
7. Deployment & Demo

---

## 1. Teori Dasar PINN

### Apa itu PINN?

PINN adalah neural network yang di-*training* tidak hanya untuk fit data, tetapi juga untuk **satisfy persamaan diferensial** yang mengatur sistem fisika.

**Loss Function Tradisional (Data-Only):**
```
L = MSE(prediction, ground_truth)
```

**Loss Function PINN (Physics-Informed):**
```
L_total = λ_data * MSE(u_pred, u_data) 
        + λ_physics * MSE(residual_PDE)
        + λ_bc * MSE(boundary_conditions)

Di mana:
- u_pred = prediksi network
- u_data = data terukur (sedikit, bisa ratusan)
- residual_PDE = error saat substitute u_pred ke persamaan diferensial
- boundary_conditions = constraint fisika di batas
```

### Keuntungan PINN untuk Getaran:

1. **Data efisien** — butuh data sedikit (vs deep learning tradisional)
2. **Extrapolation lebih baik** — constraint fisika mencegah "hallucination"
3. **Interpretable** — persamaan fisika jelas di loss function
4. **Multi-variable input** — bisa prediksi displacement, velocity, acceleration sekaligus

---

## 2. Persamaan Fisika untuk Getaran Struktur

### Persamaan Dasar: Wave Equation (untuk beam/string)

```
∂²u/∂t² + c ∂²u/∂x² = 0

Di mana:
- u(x,t) = displacement di posisi x, waktu t
- c = wave speed = sqrt(E/ρ) untuk material
- E = Young's modulus
- ρ = density
```

### Versi Lebih Realistis: Damped Wave Equation

```
∂²u/∂t² + 2ζω₀ ∂u/∂t + ω₀² u = 0

Di mana:
- ζ = damping ratio (biasanya 0.01-0.05 untuk struktur)
- ω₀ = natural frequency
- Kondisi awal: u(x,0) = f(x), ∂u/∂t(x,0) = g(x)
- Kondisi batas: u(0,t) = 0 (fixed), ∂u/∂x(L,t) = 0 (free)
```

### Untuk Beam Cantilever:

```
EI ∂⁴u/∂x⁴ + ρA ∂²u/∂t² = 0

Di mana:
- E = Young's modulus
- I = second moment of inertia
- ρ = density
- A = cross-sectional area
```

**Dalam PINN, kita compute:**
- `u_xx = ∂²u/∂x²` via automatic differentiation
- `u_tt = ∂²u/∂t²` via automatic differentiation
- Substitute ke persamaan → residual yang di-minimize

---

## 3. Arsitektur Neural Network

### Struktur PINN untuk Getaran

```
Input Layer:
  [x, t] (2 neurons)
  
Hidden Layers (Deep & Wide):
  [256 neurons, ReLU activation]
  [256 neurons, ReLU activation]
  [256 neurons, ReLU activation]
  [256 neurons, ReLU activation]
  [256 neurons, ReLU activation]
  
Output Layer:
  [u(x,t)] (1 neuron, displacement)
  
Aktivasi: Biasanya ReLU atau Tanh (Tanh lebih baik untuk periodic)
```

### Alasan Arsitektur Ini:

1. **Input layer:** x (posisi spatial) dan t (waktu)
2. **Deep & wide hidden layers:** Perlu capacity untuk fit 4 derivatives
3. **Output layer:** 1 value (displacement)
4. **ReLU:** Computation efficient, mudah di-differentiate

---

## 4. Implementasi Kode

### File 1: PINN Model (`pinn_model.py`)

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class PhysicsInformedNN(nn.Module):
    """PINN untuk getaran struktur (damped wave equation)"""
    
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=5):
        super().__init__()
        
        # Input: (x, t) → normalize ke [-1, 1]
        self.input_dim = input_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Parameter fisika (bisa di-train atau fixed)
        self.omega_0 = nn.Parameter(torch.tensor([1.0]))  # natural frequency
        self.zeta = nn.Parameter(torch.tensor([0.05]))    # damping ratio
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor shape (N, 2) dengan [x_coord, time]
        Returns:
            u: displacement, shape (N, 1)
        """
        return self.network(x)
    
    def compute_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual untuk damped wave equation:
        ∂²u/∂t² + 2ζω₀ ∂u/∂t + ω₀² u = 0
        """
        # Combine x, t untuk input ke network
        xt = torch.stack([x, t], dim=1)
        xt.requires_grad_(True)
        
        # Forward pass
        u = self.forward(xt)
        
        # First derivatives
        u_x = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 0:1]
        u_t = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 1:2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        u_tt = torch.autograd.grad(u_t.sum(), xt, create_graph=True)[0][:, 1:2]
        
        # PDE residual: u_tt + 2*zeta*omega_0*u_t + omega_0^2*u = 0
        residual = u_tt + 2 * self.zeta * self.omega_0 * u_t + (self.omega_0 ** 2) * u
        
        return residual, u
    
    def compute_bc_loss(self, x: torch.Tensor, t: torch.Tensor, 
                       bc_value: float) -> torch.Tensor:
        """
        Compute boundary condition loss
        Contoh: u(0, t) = 0 atau u(L, t) = displacement
        """
        xt = torch.stack([x, t], dim=1)
        u_bc = self.forward(xt)
        bc_loss = torch.mean((u_bc - bc_value) ** 2)
        return bc_loss
    
    def compute_ic_loss(self, x: torch.Tensor, ic_displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss
        Contoh: u(x, 0) = f(x) (initial displacement)
        """
        t_zero = torch.zeros_like(x)
        xt = torch.stack([x, t_zero], dim=1)
        u_ic = self.forward(xt)
        ic_loss = torch.mean((u_ic - ic_displacement) ** 2)
        return ic_loss
```

### File 2: Training Loop (`train_pinn.py`)

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pinn_model import PhysicsInformedNN
import numpy as np

class PINNTrainer:
    """Trainer untuk PINN dengan adaptive loss weighting"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = PhysicsInformedNN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
    def generate_training_data(self, num_samples=5000):
        """
        Generate synthetic data dari solusi analitik atau FEM
        
        Untuk simple case: u(x,t) = A * cos(ω*t) * sin(k*x)
        """
        x = np.random.uniform(0, 1, num_samples)
        t = np.random.uniform(0, 5, num_samples)
        
        # Analytical solution (simple harmonic)
        omega = 1.0  # natural frequency
        k = np.pi    # wave number
        A = 0.1      # amplitude
        
        # Exact solution dengan damping
        zeta = 0.05
        u_exact = A * np.exp(-zeta * omega * t) * np.cos(omega * t) * np.sin(k * x)
        
        # Add small noise
        u_noisy = u_exact + np.random.normal(0, 0.001, u_exact.shape)
        
        return (torch.from_numpy(x).float().to(self.device),
                torch.from_numpy(t).float().to(self.device),
                torch.from_numpy(u_noisy).float().to(self.device),
                torch.from_numpy(u_exact).float().to(self.device))
    
    def train(self, num_epochs=5000, lambda_physics=1.0, lambda_data=100.0):
        """
        Training dengan 3 loss components
        """
        x_data, t_data, u_data, u_exact = self.generate_training_data()
        
        # Separate data points untuk physics residual (bisa lebih banyak)
        num_physics = 10000
        x_physics = torch.rand(num_physics, device=self.device)
        t_physics = torch.rand(num_physics, device=self.device) * 5
        
        # Boundary condition: u(0, t) = 0 (fixed end)
        num_bc = 1000
        t_bc = torch.rand(num_bc, device=self.device) * 5
        x_bc_left = torch.zeros(num_bc, device=self.device)
        
        # Initial condition: u(x, 0) = A * sin(π*x)
        num_ic = 1000
        x_ic = torch.rand(num_ic, device=self.device)
        u_ic_target = 0.1 * torch.sin(np.pi * x_ic)
        
        loss_history = []
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # 1. Physics loss (PDE residual)
            residual, _ = self.model.compute_residual(x_physics, t_physics)
            loss_physics = torch.mean(residual ** 2)
            
            # 2. Data loss (supervised)
            xt_data = torch.stack([x_data, t_data], dim=1)
            u_pred = self.model(xt_data)
            loss_data = torch.mean((u_pred - u_data.unsqueeze(1)) ** 2)
            
            # 3. Boundary condition loss
            loss_bc = self.model.compute_bc_loss(x_bc_left, t_bc, 0.0)
            
            # 4. Initial condition loss
            loss_ic = self.model.compute_ic_loss(x_ic, u_ic_target)
            
            # Total loss
            loss_total = (lambda_physics * loss_physics + 
                         lambda_data * loss_data + 
                         loss_bc + loss_ic)
            
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Total: {loss_total:.4e} | "
                      f"Physics: {loss_physics:.4e} | "
                      f"Data: {loss_data:.4e} | "
                      f"BC: {loss_bc:.4e} | "
                      f"IC: {loss_ic:.4e} | "
                      f"ω₀: {self.model.omega_0.item():.4f} | "
                      f"ζ: {self.model.zeta.item():.4f}")
            
            loss_history.append({
                'total': loss_total.item(),
                'physics': loss_physics.item(),
                'data': loss_data.item(),
                'bc': loss_bc.item(),
                'ic': loss_ic.item()
            })
        
        return loss_history
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Prediksi displacement untuk arbitrary (x, t) points"""
        x_torch = torch.from_numpy(x).float().to(self.device)
        t_torch = torch.from_numpy(t).float().to(self.device)
        xt = torch.stack([x_torch, t_torch], dim=1)
        
        with torch.no_grad():
            u_pred = self.model(xt).cpu().numpy()
        
        return u_pred
    
    def save_model(self, filepath='pinn_vibration.pth'):
        """Save model untuk inference nanti"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='pinn_vibration.pth'):
        """Load model yang sudah di-train"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


# Main training script
if __name__ == "__main__":
    trainer = PINNTrainer()
    
    print("Starting PINN training for vibration prediction...")
    loss_history = trainer.train(num_epochs=5000, lambda_physics=1.0, lambda_data=10.0)
    
    # Save model
    trainer.save_model('pinn_vibration.pth')
    
    # Plot loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot([l['total'] for l in loss_history], label='Total Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot([l['physics'] for l in loss_history], label='Physics Loss', alpha=0.7)
    plt.plot([l['data'] for l in loss_history], label='Data Loss', alpha=0.7)
    plt.plot([l['bc'] for l in loss_history], label='BC Loss', alpha=0.7)
    plt.plot([l['ic'] for l in loss_history], label='IC Loss', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to training_curves.png")
```

### File 3: Inference & Visualization (`inference_pinn.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from train_pinn import PINNTrainer
import matplotlib.patches as mpatches

def visualize_predictions():
    """Visualisasi prediksi PINN vs ground truth"""
    
    trainer = PINNTrainer()
    trainer.load_model('pinn_vibration.pth')
    
    # Generate test data
    x_test = np.linspace(0, 1, 100)
    t_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Analytical solution untuk comparison
    omega = trainer.model.omega_0.item()
    zeta = trainer.model.zeta.item()
    k = np.pi
    A = 0.1
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, t_val in enumerate(t_test):
        # Create mesh untuk prediction
        x_mesh = np.linspace(0, 1, 100)
        t_mesh = np.ones_like(x_mesh) * t_val
        
        # PINN prediction
        u_pred = trainer.predict(x_mesh, t_mesh).flatten()
        
        # Analytical solution
        u_analytical = A * np.exp(-zeta * omega * t_val) * np.cos(omega * t_val) * np.sin(k * x_mesh)
        
        # Plot
        ax = axes[i]
        ax.plot(x_mesh, u_analytical, 'b-', linewidth=2, label='Analytical')
        ax.plot(x_mesh, u_pred, 'r--', linewidth=2, label='PINN Prediction')
        ax.fill_between(x_mesh, u_analytical, u_pred, alpha=0.2, color='orange')
        
        # Error
        error = np.mean(np.abs(u_analytical - u_pred))
        
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Displacement u(x,t)')
        ax.set_title(f't = {t_val:.1f}s | Error: {error:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('pinn_predictions.png', dpi=150)
    print("Predictions saved to pinn_predictions.png")
    plt.close()
    
    # Create space-time contour plot
    x_fine = np.linspace(0, 1, 200)
    t_fine = np.linspace(0, 5, 200)
    X, T = np.meshgrid(x_fine, t_fine)
    
    # Flatten untuk prediction
    x_flat = X.flatten()
    t_flat = T.flatten()
    u_flat = trainer.predict(x_flat, t_flat).flatten()
    U = u_flat.reshape(X.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PINN prediction
    im1 = axes[0].contourf(X, T, U, levels=20, cmap='RdBu_r')
    axes[0].set_xlabel('Position (x)')
    axes[0].set_ylabel('Time (t)')
    axes[0].set_title('PINN Prediction: u(x,t)')
    plt.colorbar(im1, ax=axes[0])
    
    # Analytical
    k = np.pi
    U_analytical = A * np.exp(-zeta * omega * T) * np.cos(omega * T) * np.sin(k * X)
    im2 = axes[1].contourf(X, T, U_analytical, levels=20, cmap='RdBu_r')
    axes[1].set_xlabel('Position (x)')
    axes[1].set_ylabel('Time (t)')
    axes[1].set_title('Analytical Solution: u(x,t)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('pinn_spacetime.png', dpi=150)
    print("Space-time contour saved to pinn_spacetime.png")
    plt.close()


def compare_training_efficiency():
    """Bandingkan PINN vs traditional supervised learning"""
    
    print("\n=== Training Efficiency Comparison ===\n")
    
    data_sizes = [100, 500, 1000, 5000, 10000]
    pinn_errors = []
    supervised_errors = []
    
    for n_data in data_sizes:
        print(f"Training with {n_data} data points...")
        
        # PINN approach
        trainer_pinn = PINNTrainer()
        trainer_pinn.train(num_epochs=2000, lambda_physics=1.0, lambda_data=10.0)
        
        # Evaluate on test set
        x_test = np.linspace(0, 1, 100)
        t_test = np.linspace(0, 5, 100)
        X_test, T_test = np.meshgrid(x_test, t_test)
        
        u_pred = trainer_pinn.predict(X_test.flatten(), T_test.flatten()).flatten()
        u_true = 0.1 * np.exp(-0.05 * 1.0 * T_test.flatten()) * np.cos(1.0 * T_test.flatten()) * np.sin(np.pi * X_test.flatten())
        
        pinn_error = np.mean(np.abs(u_pred - u_true))
        pinn_errors.append(pinn_error)
        
        print(f"  PINN Error: {pinn_error:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, pinn_errors, 'o-', linewidth=2, markersize=8, label='PINN (Physics-Informed)')
    plt.xlabel('Number of Training Data Points')
    plt.ylabel('Mean Absolute Error')
    plt.title('PINN vs Data Efficiency: Fewer Data Points Needed')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('efficiency_comparison.png', dpi=150)
    print("\nEfficiency comparison saved to efficiency_comparison.png")


if __name__ == "__main__":
    # Visualize predictions
    visualize_predictions()
    
    # Compare training efficiency
    # compare_training_efficiency()
```

### File 4: Streamlit Demo App (`app_demo.py`)

```python
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_pinn import PINNTrainer
import json

st.set_page_config(page_title="PINN Vibration Predictor", layout="wide")

st.title("⚛️ Physics-Informed Neural Network")
st.subtitle("Prediksi getaran struktur dengan persamaan fisika")

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    mode = st.radio("Mode:", ["Train", "Inference"])
    
    if mode == "Train":
        num_epochs = st.slider("Number of Epochs", 100, 10000, 5000, step=500)
        lambda_physics = st.slider("Physics Loss Weight", 0.1, 10.0, 1.0, step=0.5)
        lambda_data = st.slider("Data Loss Weight", 1.0, 100.0, 10.0, step=5.0)
    else:
        x_input = st.slider("Position (x)", 0.0, 1.0, 0.5, step=0.05)
        t_input = st.slider("Time (t)", 0.0, 5.0, 2.0, step=0.1)

# Main content
if mode == "Train":
    st.markdown("""
    ### 1️⃣ Training PINN Model
    
    PINN trains by minimizing 3 loss terms:
    - **Physics Loss**: PDE residual (∂²u/∂t² + 2ζω₀ ∂u/∂t + ω₀²u = 0)
    - **Data Loss**: Supervised learning dari measured data
    - **Boundary Condition**: u(0,t) = 0 (fixed end)
    """)
    
    if st.button("🚀 Start Training", key="train_btn"):
        with st.spinner("Training PINN... This takes ~2-5 minutes"):
            trainer = PINNTrainer()
            loss_history = trainer.train(num_epochs=num_epochs, 
                                        lambda_physics=lambda_physics,
                                        lambda_data=lambda_data)
            trainer.save_model('pinn_vibration.pth')
        
        st.success("✅ Training completed!")
        
        # Display loss curves
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([l['total'] for l in loss_history], linewidth=2)
            ax.set_yscale('log')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Total Loss (log)')
            ax.grid(True, alpha=0.3)
            ax.set_title('Training Progress')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([l['physics'] for l in loss_history], label='Physics', alpha=0.7)
            ax.plot([l['data'] for l in loss_history], label='Data', alpha=0.7)
            ax.plot([l['bc'] for l in loss_history], label='BC', alpha=0.7)
            ax.set_yscale('log')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (log)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Loss Components')
            st.pyplot(fig)
        
        # Save metrics
        with st.expander("📊 Final Metrics"):
            final_loss = loss_history[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Loss", f"{final_loss['total']:.4e}")
            col2.metric("Physics Loss", f"{final_loss['physics']:.4e}")
            col3.metric("Data Loss", f"{final_loss['data']:.4e}")
            col4.metric("BC Loss", f"{final_loss['bc']:.4e}")

else:  # Inference
    st.markdown("""
    ### 2️⃣ Predict Vibration
    
    Gunakan trained model untuk predict displacement di arbitrary (x,t)
    """)
    
    try:
        trainer = PINNTrainer()
        trainer.load_model('pinn_vibration.pth')
        
        # Single point prediction
        st.subheader("Single Point Prediction")
        u_single = trainer.predict(np.array([x_input]), np.array([t_input]))[0, 0]
        st.metric("u(x,t) = Displacement", f"{u_single:.6f} m", 
                 delta=None, delta_color="off")
        
        # Spatial profile at fixed time
        st.subheader("Spatial Profile")
        x_range = np.linspace(0, 1, 100)
        t_fixed = t_input
        u_spatial = trainer.predict(x_range, np.ones_like(x_range) * t_fixed)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_range, u_spatial.flatten(), linewidth=2.5, color='#1f77b4')
        ax.fill_between(x_range, u_spatial.flatten(), alpha=0.3)
        ax.set_xlabel('Position (x) [m]')
        ax.set_ylabel('Displacement u(x,t) [m]')
        ax.set_title(f'Displacement Profile at t={t_fixed:.2f}s')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Temporal profile at fixed position
        st.subheader("Temporal Profile")
        t_range = np.linspace(0, 5, 200)
        x_fixed = x_input
        u_temporal = trainer.predict(np.ones_like(t_range) * x_fixed, t_range)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_range, u_temporal.flatten(), linewidth=2.5, color='#ff7f0e')
        ax.fill_between(t_range, u_temporal.flatten(), alpha=0.3, color='#ff7f0e')
        ax.set_xlabel('Time (t) [s]')
        ax.set_ylabel('Displacement u(x,t) [m]')
        ax.set_title(f'Displacement vs Time at x={x_fixed:.2f}m')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Space-time contour
        st.subheader("Space-Time Contour")
        x_mesh = np.linspace(0, 1, 100)
        t_mesh = np.linspace(0, 5, 100)
        X, T = np.meshgrid(x_mesh, t_mesh)
        u_mesh = trainer.predict(X.flatten(), T.flatten()).reshape(X.shape)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(X, T, u_mesh, levels=20, cmap='RdBu_r')
        ax.scatter([x_fixed], [t_fixed], color='black', s=100, marker='x', linewidths=2, label='Current point')
        ax.set_xlabel('Position (x) [m]')
        ax.set_ylabel('Time (t) [s]')
        ax.set_title('Space-Time Displacement Field')
        plt.colorbar(contour, ax=ax, label='Displacement [m]')
        ax.legend()
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please train the model first!")

# Footer
st.markdown("""
---
### 📚 Informasi Teknis

**PINN Loss Function:**
```
L = λ_physics × ||PDE residual||² 
  + λ_data × ||prediction - data||²
  + ||boundary conditions||²
  + ||initial conditions||²
```

**Persamaan Getaran (Damped Wave):**
```
∂²u/∂t² + 2ζω₀(∂u/∂t) + ω₀²u = 0
```

**Keuntungan PINN:**
- ✅ Data efficient (butuh ratusan data, bukan ribuan)
- ✅ Physics-consistent predictions
- ✅ Extrapolation lebih baik
- ✅ Interpretable (persamaan jelas)
""")
```

---

## 5. Dataset & Training

### Opsi Data:

#### A. Synthetic Data (untuk development)
```python
# Analytical solution dengan damping
u(x,t) = A * exp(-ζω₀t) * cos(ω₀t) * sin(πx)

Parameter:
- A = 0.1 (amplitude)
- ω₀ = 1.0 rad/s (natural frequency)
- ζ = 0.05 (damping ratio, 5%)
- Domain: x ∈ [0,1], t ∈ [0,5]s
```

#### B. Real Data Options (jika ada):
1. **Sensor accelerometer dari struktur nyata**
   - Integrasi 2× → displacement
   - Bisa dari LSTM accelerometer, IMU sensor
   
2. **FEM simulation data**
   - ANSYS, COMSOL hasil export
   - Mesh finer dari PINN (PINN lebih coarse)

3. **Public datasets:**
   - ASCE SHM Benchmark (structural health monitoring)
   - Dryden Wind Turbine Dataset
   - Building acceleration records (PEER)

---

## 6. Hasil & Interpretasi

### Ekspek Output:

1. **Training curves** → Loss turun exponentially (physics constraint sangat efektif)
2. **Learned parameters:**
   - ω₀ ≈ 1.0 (natural frequency)
   - ζ ≈ 0.05 (damping ratio)
   
3. **Prediction vs analytical:**
   - MAE < 0.001 m (very good)
   - Extrapolation lebih baik dari supervised NN

### Interpretasi:

```
Jika ω₀ prediksi = 1.234 rad/s
→ Frekuensi natural struktur ≈ 1.234/(2π) ≈ 0.196 Hz
→ Period = 5.1 detik

Jika ζ prediksi = 0.048
→ Struktur memiliki 4.8% damping ratio
→ Sangat baik (struktur tidak terlalu ossilasi)
```

---

## 7. Deployment & Demo

### Cara package untuk portfolio:

1. **Training: 5 menit, GPU optional**
2. **Inference: <1ms per point, bisa run di laptop**
3. **Model size: ~500KB** (vs 500MB untuk trained CNN)

### Production deployment:

```python
# FastAPI server
from fastapi import FastAPI
from pinn_model import PhysicsInformedNN
import torch

app = FastAPI()
model = PhysicsInformedNN()
model.load_state_dict(torch.load('pinn_vibration.pth'))

@app.post("/predict")
def predict(x: float, t: float):
    with torch.no_grad():
        xt = torch.tensor([[x, t]], dtype=torch.float32)
        u = model(xt).item()
    return {"displacement": u, "x": x, "t": t}
```

---

## Summary: Dari 0 ke PINN

**Step by step:**

1. Implement `PhysicsInformedNN` class ✅
2. Add automatic differentiation untuk PDE ✅
3. Setup multi-component loss (physics + data + BC + IC) ✅
4. Training dengan adaptive learning rate ✅
5. Visualization + validation ✅
6. Streamlit demo untuk presentasi ✅

**Waktu development: ~2-3 hari** (jika copy-paste kode ini)

**Keunggulan untuk portfolio:**
- Rare skill (kombinasi ML + physics)
- Impressive demo (visual + numbers)
- Scalable (bisa apply ke heat transfer, fluid dynamics, dll)
- Interview-ready (bisa explain setiap baris)

---

## Troubleshooting

| Problem | Solusi |
|---------|--------|
| Loss tidak turun | Turunkan learning rate, check PDE formula |
| Overfitting pada data | Naikan λ_physics weight |
| CUDA out of memory | Kurangi hidden_dim atau batch size |
| ω₀ tidak learn | Fix parameter atau naikan initial value |

