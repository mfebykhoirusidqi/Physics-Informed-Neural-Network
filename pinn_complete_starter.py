"""
Physics-Informed Neural Network untuk Prediksi Getaran Struktur
Complete starter code - Runnable dengan "python pinn_complete_starter.py"

Requirements:
pip install torch numpy matplotlib scipy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. PINN MODEL CLASS
# =====================================================================

class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network untuk damped wave equation"""
    
    def __init__(self, hidden_dim=256, num_layers=5, device='cpu'):
        super().__init__()
        self.device = device
        
        # Build network: (x, t) → u
        layers = []
        input_dim = 2
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())  # Tanh lebih baik untuk oscillatory solutions
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Learnable physical parameters
        self.omega_0 = nn.Parameter(torch.tensor([1.0], device=device))
        self.zeta = nn.Parameter(torch.tensor([0.05], device=device))
        
        # Initialize weights
        for name, param in self.network.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, 0.0, 0.1)
    
    def forward(self, xt):
        """Forward pass: (x, t) → displacement u"""
        return self.network(xt)
    
    def pde_residual(self, x, t):
        """
        Compute PDE residual: ∂²u/∂t² + 2ζω₀(∂u/∂t) + ω₀²u
        
        Returns residual (should be close to 0) and u prediction
        """
        x = x.unsqueeze(1) if x.dim() == 1 else x
        t = t.unsqueeze(1) if t.dim() == 1 else t
        
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        # Forward pass
        u = self.forward(xt)
        
        # Gradient: ∂u/∂x, ∂u/∂t
        grads = torch.autograd.grad(
            u, xt, 
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        
        # Second gradient: ∂²u/∂t²
        u_tt = torch.autograd.grad(
            u_t, xt,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # Second gradient: ∂u/∂t again for derivative
        u_t_xt = torch.autograd.grad(
            u_t, xt,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True
        )[0]
        u_t_t = u_t_xt[:, 1:2]  # ∂²u/∂t²
        
        # PDE: ∂²u/∂t² + 2ζω₀(∂u/∂t) + ω₀²u = 0
        residual = (u_tt + 
                   2 * self.zeta * self.omega_0 * u_t + 
                   (self.omega_0 ** 2) * u)
        
        return residual, u
    
    def boundary_condition_loss(self, x_bc, t_bc, bc_value=0.0):
        """Loss untuk boundary condition: u(0,t) = bc_value"""
        xt_bc = torch.cat([x_bc.unsqueeze(1), t_bc.unsqueeze(1)], dim=1)
        u_bc = self.forward(xt_bc)
        return torch.mean((u_bc - bc_value) ** 2)
    
    def initial_condition_loss(self, x_ic, u_ic):
        """Loss untuk initial condition: u(x,0) = u_ic(x)"""
        t_zero = torch.zeros_like(x_ic)
        xt_ic = torch.cat([x_ic.unsqueeze(1), t_zero.unsqueeze(1)], dim=1)
        u_pred = self.forward(xt_ic)
        u_ic = u_ic.unsqueeze(1) if u_ic.dim() == 1 else u_ic
        return torch.mean((u_pred - u_ic) ** 2)


# =====================================================================
# 2. DATA GENERATION
# =====================================================================

def generate_synthetic_data(num_data=5000, num_physics=10000):
    """Generate training data dari analytical solution"""
    
    # Physical parameters
    omega = 1.0      # Natural frequency [rad/s]
    zeta = 0.05      # Damping ratio
    A = 0.1          # Amplitude
    k = np.pi        # Wave number
    
    # Data points (sparse, with noise)
    x_data = np.random.uniform(0, 1, num_data)
    t_data = np.random.uniform(0, 5, num_data)
    
    # Analytical solution dengan damping
    u_analytical = A * np.exp(-zeta * omega * t_data) * np.cos(omega * t_data) * np.sin(k * x_data)
    u_noisy = u_analytical + np.random.normal(0, 0.001, u_analytical.shape)
    
    # Physics points (collocation points untuk PDE)
    x_physics = np.random.uniform(0, 1, num_physics)
    t_physics = np.random.uniform(0, 5, num_physics)
    
    # Boundary condition points: x=0
    t_bc = np.random.uniform(0, 5, 2000)
    x_bc = np.zeros_like(t_bc)
    
    # Initial condition: u(x,0) = A*sin(πx)
    x_ic = np.linspace(0, 1, 2000)
    u_ic = A * np.sin(k * x_ic)
    
    return {
        'x_data': torch.from_numpy(x_data).float(),
        'u_data': torch.from_numpy(u_noisy).float(),
        'x_physics': torch.from_numpy(x_physics).float(),
        't_physics': torch.from_numpy(t_physics).float(),
        't_data': torch.from_numpy(t_data).float(),
        't_bc': torch.from_numpy(t_bc).float(),
        'x_bc': torch.from_numpy(x_bc).float(),
        'x_ic': torch.from_numpy(x_ic).float(),
        'u_ic': torch.from_numpy(u_ic).float(),
        'u_analytical': u_analytical,
        'x_analytical': x_data,
        't_analytical': t_data,
    }


# =====================================================================
# 3. TRAINING
# =====================================================================

def train_pinn(model, data, num_epochs=2000, device='cpu', 
               lambda_physics=1.0, lambda_data=10.0):
    """Train PINN dengan multi-component loss menggunakan kombinasi Adam dan L-BFGS"""
    
    # Move data to device
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    
    loss_history = []
    
    print("\n" + "="*70)
    print("Starting PINN Training for Vibration Prediction")
    print("="*70)
    print(f"Device: {device}")
    print(f"Physics loss weight (λ_physics): {lambda_physics}")
    print(f"Data loss weight (λ_data): {lambda_data}")
    print("="*70 + "\n")

    def compute_loss():
        residual, _ = model.pde_residual(data['x_physics'], data['t_physics'])
        loss_physics = torch.mean(residual ** 2)
        
        xt_data = torch.cat([data['x_data'].unsqueeze(1), 
                            data['t_data'].unsqueeze(1)], dim=1)
        u_pred = model(xt_data)
        loss_data = torch.mean((u_pred - data['u_data'].unsqueeze(1)) ** 2)
        
        loss_bc = model.boundary_condition_loss(data['x_bc'], data['t_bc'], bc_value=0.0)
        loss_ic = model.initial_condition_loss(data['x_ic'], data['u_ic'])
        
        loss_total = (lambda_physics * loss_physics + 
                     lambda_data * loss_data + 
                     loss_bc + loss_ic)
                     
        return loss_total, loss_physics, loss_data, loss_bc, loss_ic
        
    def record_loss(epoch, l_tot, l_phys, l_dat, l_bc, l_ic):
        if (epoch + 1) % 500 == 0 or epoch == 0:
            omega_pred = model.omega_0.item()
            zeta_pred = model.zeta.item()
            print(f"Epoch {epoch+1:5d} | Loss: {l_tot:.4e} | Phys: {l_phys:.4e} | Data: {l_dat:.4e} | ω₀: {omega_pred:.4f} | ζ: {zeta_pred:.4f}")
        
        loss_history.append({
            'epoch': epoch,
            'total': l_tot.item(),
            'physics': l_phys.item(),
            'data': l_dat.item(),
            'bc': l_bc.item(),
            'ic': l_ic.item(),
            'omega': model.omega_0.item(),
            'zeta': model.zeta.item(),
        })

    # === TAHAP 1: ADAM === #
    epochs_adam = int(num_epochs * 0.8)
    print(f"--- TAHAP 1: Adam Optimizer ({epochs_adam} Epochs) ---")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=1000, gamma=0.9)
    
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        loss_total, loss_physics, loss_data, loss_bc, loss_ic = compute_loss()
        
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step()
        
        record_loss(epoch, loss_total, loss_physics, loss_data, loss_bc, loss_ic)

    # === TAHAP 2: L-BFGS === #
    epochs_lbfgs = num_epochs - epochs_adam
    if epochs_lbfgs > 0:
        print(f"\n--- TAHAP 2: L-BFGS Optimizer ({epochs_lbfgs} Iterations) ---")
        
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(), 
            lr=1.0, 
            max_iter=20, 
            max_eval=25, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9, 
            history_size=50, 
            line_search_fn="strong_wolfe"
        )

        epoch_lbfgs = [epochs_adam]
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_total, loss_physics, loss_data, loss_bc, loss_ic = compute_loss()
            loss_total.backward()
            record_loss(epoch_lbfgs[0], loss_total, loss_physics, loss_data, loss_bc, loss_ic)
            epoch_lbfgs[0] += 1
            return loss_total
            
        while epoch_lbfgs[0] < num_epochs:
            try:
                optimizer_lbfgs.step(closure)
                if epoch_lbfgs[0] >= num_epochs:
                    break
            except Exception:
                break
            
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Final Parameters: ω₀ = {model.omega_0.item():.4f}, ζ = {model.zeta.item():.4f}")
    print("="*70 + "\n")
    
    return loss_history


# =====================================================================
# 4. INFERENCE & VISUALIZATION
# =====================================================================

def predict(model, x, t, device='cpu'):
    """Predict displacement untuk arbitrary (x,t)"""
    x_torch = torch.from_numpy(x).float().to(device)
    t_torch = torch.from_numpy(t).float().to(device)
    xt = torch.cat([x_torch.unsqueeze(1), t_torch.unsqueeze(1)], dim=1)
    
    with torch.no_grad():
        u_pred = model(xt)
    
    return u_pred.cpu().numpy()


def plot_results(model, loss_history, data, device='cpu'):
    """Visualisasi hasil training dan prediksi"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    epochs = [l['epoch'] for l in loss_history]
    ax1.semilogy([l['total'] for l in loss_history], label='Total', linewidth=2)
    ax1.semilogy([l['physics'] for l in loss_history], label='Physics', alpha=0.7)
    ax1.semilogy([l['data'] for l in loss_history], label='Data', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter learning
    ax2 = plt.subplot(2, 3, 2)
    ax2_twin = ax2.twinx()
    ax2.plot([l['omega'] for l in loss_history], 'b-', linewidth=2, label='ω₀')
    ax2_twin.plot([l['zeta'] for l in loss_history], 'r-', linewidth=2, label='ζ')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ω₀ [rad/s]', color='b')
    ax2_twin.set_ylabel('ζ (damping ratio)', color='r')
    ax2.set_title('Learned Physical Parameters')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # 3. Snapshots di berbagai waktu
    times_to_plot = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(times_to_plot)))
    
    ax3 = plt.subplot(2, 3, 3)
    x_test = np.linspace(0, 1, 100)
    
    for t_val, color in zip(times_to_plot, colors):
        t_test = np.ones_like(x_test) * t_val
        u_pred = predict(model, x_test, t_test, device)
        ax3.plot(x_test, u_pred.flatten(), color=color, linewidth=2, 
                label=f't={t_val:.1f}s')
    
    ax3.set_xlabel('Position x [m]')
    ax3.set_ylabel('Displacement u [m]')
    ax3.set_title('Spatial Profiles at Different Times')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Space-time contour: PINN prediction
    ax4 = plt.subplot(2, 3, 4)
    x_mesh = np.linspace(0, 1, 100)
    t_mesh = np.linspace(0, 5, 150)
    X, T = np.meshgrid(x_mesh, t_mesh)
    U_pred = predict(model, X.flatten(), T.flatten(), device).reshape(X.shape)
    
    contour1 = ax4.contourf(X, T, U_pred, levels=20, cmap='RdBu_r')
    ax4.set_xlabel('Position x [m]')
    ax4.set_ylabel('Time t [s]')
    ax4.set_title('PINN Prediction: u(x,t)')
    plt.colorbar(contour1, ax=ax4, label='Displacement [m]')
    
    # 5. Space-time contour: Analytical
    ax5 = plt.subplot(2, 3, 5)
    omega = model.omega_0.item()
    zeta = model.zeta.item()
    k = np.pi
    A = 0.1
    U_analytical = A * np.exp(-zeta * omega * T) * np.cos(omega * T) * np.sin(k * X)
    
    contour2 = ax5.contourf(X, T, U_analytical, levels=20, cmap='RdBu_r')
    ax5.set_xlabel('Position x [m]')
    ax5.set_ylabel('Time t [s]')
    ax5.set_title('Analytical Solution: u(x,t)')
    plt.colorbar(contour2, ax=ax5, label='Displacement [m]')
    
    # 6. Error distribution
    ax6 = plt.subplot(2, 3, 6)
    error = np.abs(U_pred - U_analytical)
    im = ax6.contourf(X, T, error, levels=20, cmap='YlOrRd')
    ax6.set_xlabel('Position x [m]')
    ax6.set_ylabel('Time t [s]')
    ax6.set_title(f'Absolute Error (MAE: {np.mean(error):.4e})')
    plt.colorbar(im, ax=ax6, label='Error [m]')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'pinn_results_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Results saved: pinn_results_{timestamp}.png")
    plt.show()


# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================

def main():
    """Run complete PINN training and visualization"""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating synthetic training data...")
    data = generate_synthetic_data(num_data=5000, num_physics=10000)
    
    # Create model
    print("Initializing PINN model...")
    model = PhysicsInformedNN(hidden_dim=256, num_layers=5, device=device)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    loss_history = train_pinn(
        model, data, 
        num_epochs=5000,
        device=device,
        lambda_physics=1.0,
        lambda_data=10.0
    )
    
    # Visualize
    print("Generating visualizations...")
    plot_results(model, loss_history, data, device)
    
    # Save model
    torch.save(model.state_dict(), 'pinn_vibration_model.pth')
    print("✅ Model saved: pinn_vibration_model.pth")
    
    return model, loss_history, data


if __name__ == "__main__":
    model, loss_history, data = main()
    print("\n✅ PINN training completed successfully!")
