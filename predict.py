"""
PINN Inference Script
======================
Standalone API for loading and querying a trained Physics-Informed Neural Network.

Usage:
    python predict.py --x 0.25 0.5 0.75 --t 1.0 2.0 3.0
    python predict.py --grid --nx 50 --nt 80 --t_max 5.0
    python predict.py --params-only
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_complete_starter import PhysicsInformedNN


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_path: str = "pinn_vibration_model.pth",
    hidden_dim: int = 256,
    num_layers: int = 5,
    device: str = "cpu",
) -> PhysicsInformedNN:
    """
    Load a saved PINN model from disk.

    Args:
        model_path : Path to the .pth checkpoint file.
        hidden_dim : Width of hidden layers (must match trained model).
        num_layers : Number of hidden layers (must match trained model).
        device     : 'cpu' or 'cuda'.

    Returns:
        model : Loaded PINN in eval mode.

    Raises:
        FileNotFoundError : If model_path does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training first: python pinn_complete_starter.py"
        )

    model = PhysicsInformedNN(hidden_dim=hidden_dim, num_layers=num_layers, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def predict(
    model: PhysicsInformedNN,
    x: np.ndarray,
    t: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict structural displacement u(x, t).

    Args:
        model  : Trained PhysicsInformedNN.
        x      : Position array [m], shape (N,) or scalar.
        t      : Time array [s],     shape (N,) or scalar.
        device : 'cpu' or 'cuda'.

    Returns:
        u : Predicted displacement [m], shape (N,).
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    t = np.asarray(t, dtype=np.float32).flatten()

    if len(x) != len(t):
        raise ValueError("x and t must have the same number of elements.")

    x_torch = torch.from_numpy(x).to(device)
    t_torch = torch.from_numpy(t).to(device)
    xt = torch.stack([x_torch, t_torch], dim=1)

    with torch.no_grad():
        u = model(xt)

    return u.cpu().numpy().flatten()


def predict_grid(
    model: PhysicsInformedNN,
    x_range: tuple = (0.0, 1.0),
    t_range: tuple = (0.0, 5.0),
    nx: int = 100,
    nt: int = 150,
    device: str = "cpu",
) -> tuple:
    """
    Predict on a regular spatial-temporal grid.

    Args:
        model   : Trained PhysicsInformedNN.
        x_range : (x_min, x_max) in meters.
        t_range : (t_min, t_max) in seconds.
        nx      : Grid resolution in space.
        nt      : Grid resolution in time.
        device  : 'cpu' or 'cuda'.

    Returns:
        X : Meshgrid array (nt × nx) of x positions.
        T : Meshgrid array (nt × nx) of time values.
        U : Predicted displacement (nt × nx).
    """
    x = np.linspace(*x_range, nx, dtype=np.float32)
    t = np.linspace(*t_range, nt, dtype=np.float32)
    X, T = np.meshgrid(x, t)

    U = predict(model, X.flatten(), T.flatten(), device).reshape(X.shape)
    return X, T, U


def get_learned_parameters(model: PhysicsInformedNN) -> dict:
    """
    Retrieve the physical parameters discovered by the PINN.

    Returns:
        dict with keys: omega_0, zeta, omega_d (damped freq), tau (decay constant).
    """
    omega_0 = model.omega_0.item()
    zeta    = model.zeta.item()
    omega_d = omega_0 * np.sqrt(max(0.0, 1.0 - zeta**2))
    tau     = 1.0 / (zeta * omega_0) if (zeta * omega_0) > 0 else float("inf")

    return {
        "omega_0": omega_0,
        "zeta":    zeta,
        "omega_d": omega_d,
        "tau":     tau,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PINN Inference — Physics-Informed Neural Network",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",  type=str,   default="pinn_vibration_model.pth",
                        help="Path to trained .pth file")
    parser.add_argument("--device", type=str,   default="cpu", choices=["cpu", "cuda"],
                        help="Compute device")

    # Point-wise mode
    parser.add_argument("--x", type=float, nargs="+",
                        help="Position values [m]  (paired with --t)")
    parser.add_argument("--t", type=float, nargs="+",
                        help="Time values [s]       (paired with --x)")

    # Grid mode
    parser.add_argument("--grid",    action="store_true",
                        help="Predict on a spatial-temporal grid and save CSV")
    parser.add_argument("--nx",      type=int, default=50,  help="Grid x-resolution")
    parser.add_argument("--nt",      type=int, default=80,  help="Grid t-resolution")
    parser.add_argument("--t_max",   type=float, default=5.0, help="Max time [s]")
    parser.add_argument("--out",     type=str, default="pinn_grid_predictions.csv",
                        help="Output CSV for grid mode")

    # Params only
    parser.add_argument("--params-only", action="store_true",
                        help="Only print learned physical parameters")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Load model ────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("⚛️  PINN Inference Engine")
    print(f"{'─'*55}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")

    try:
        model = load_model(args.model, device=args.device)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # ── Learned parameters ────────────────────────────────────
    params = get_learned_parameters(model)
    print(f"\n📊 Learned Physical Parameters:")
    print(f"   ω₀  (natural frequency) : {params['omega_0']:.6f} rad/s")
    print(f"   ζ   (damping ratio)     : {params['zeta']:.6f}")
    print(f"   ωd  (damped frequency)  : {params['omega_d']:.6f} rad/s")
    print(f"   τ   (decay constant)    : {params['tau']:.4f} s")

    if args.params_only:
        return

    # ── Point-wise inference ──────────────────────────────────
    if args.x is not None and args.t is not None:
        if len(args.x) != len(args.t):
            print("\n❌ --x and --t must have the same number of values.")
            sys.exit(1)

        x_arr = np.array(args.x)
        t_arr = np.array(args.t)
        u_arr = predict(model, x_arr, t_arr, args.device)

        print(f"\n📈 Point-wise Predictions u(x, t):")
        print(f"   {'x [m]':>10}  {'t [s]':>10}  {'u [m]':>14}")
        print(f"   {'─'*10}  {'─'*10}  {'─'*14}")
        for xi, ti, ui in zip(x_arr, t_arr, u_arr):
            print(f"   {xi:10.4f}  {ti:10.4f}  {ui:14.8f}")

    # ── Grid inference ────────────────────────────────────────
    elif args.grid:
        print(f"\n🔢 Computing grid ({args.nx} × {args.nt}) …")
        X, T, U = predict_grid(
            model,
            x_range=(0.0, 1.0),
            t_range=(0.0, args.t_max),
            nx=args.nx,
            nt=args.nt,
            device=args.device,
        )

        import pandas as pd
        df = pd.DataFrame({
            "x": X.flatten(),
            "t": T.flatten(),
            "u_predicted": U.flatten(),
        })
        df.to_csv(args.out, index=False)
        print(f"   Saved {len(df):,} predictions → {args.out}")
        print(f"   u range: [{U.min():.4e}, {U.max():.4e}] m")

    else:
        print("\n💡 Usage examples:")
        print("   python predict.py --x 0.5 --t 1.0")
        print("   python predict.py --x 0.25 0.5 0.75 --t 1.0 2.0 3.0")
        print("   python predict.py --grid --nx 100 --nt 150")
        print("   python predict.py --params-only")

    print(f"\n{'─'*55}\n")


if __name__ == "__main__":
    main()
