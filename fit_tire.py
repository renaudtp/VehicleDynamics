"""
Fit tire_models.PacejkaTireModel's (B, C, E, mu, load_sensitivity) against
the REAL Bakker/Pacejka/Nyborg (1987) tire (reference_tire.py).

My simplified model uses a single global (B, C, E) across all loads, with
only D(Fz) varying — the real '87 formula lets B and E vary with Fz too
(through the BCD/D ratio and the a6/a7 quadratic-in-Fz terms). That's a
real, expected source of residual error, not a bug: it's the cost of
collapsing a load-dependent-shape model down to a load-dependent-peak-only
model for simplicity. This script quantifies exactly how much that costs.

Fz_nom is fixed at 4000 N (matches tire_models.py's default) rather than
fit, because (mu, load_sensitivity, Fz_nom) are not jointly identifiable
from force data alone — only the combination mu/Fz_nom matters. Fitting
D(Fz) = c1*Fz - c2*Fz^2 directly and then recovering
mu = c1, load_sensitivity = c2*Fz_nom/c1 sidesteps that degeneracy.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from reference_tire import bakker87_lateral_force

FZ_NOM = 4000.0  # N, matches tire_models.PacejkaTireModel default


def fit_model(alpha_rad, Fz_N, B, C, E, mu, c2):
    D = mu * Fz_N - c2 * Fz_N**2
    Bx = B * alpha_rad
    return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))


def build_dataset():
    # Typical passenger-tire load range: 2-8 kN per tire (~450-1800 lbf),
    # covering light-to-heavy load transfer conditions.
    Fz_kN_vals = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    alpha_deg_vals = np.linspace(-15, 15, 61)

    Fz_grid, alpha_grid = np.meshgrid(Fz_kN_vals, alpha_deg_vals, indexing="ij")
    Fy_real = bakker87_lateral_force(alpha_grid, Fz_grid)

    alpha_rad = np.deg2rad(alpha_grid).ravel()
    Fz_N = (Fz_grid * 1000.0).ravel()
    Fy = Fy_real.ravel()
    return alpha_rad, Fz_N, Fy, Fz_kN_vals, alpha_deg_vals, Fy_real


def main():
    alpha_rad, Fz_N, Fy, Fz_kN_vals, alpha_deg_vals, Fy_real = build_dataset()

    def wrapper(X, B, C, E, mu, c2):
        a, f = X
        return fit_model(a, f, B, C, E, mu, c2)

    p0 = [10.0, 1.9, 0.0, 1.0, 1e-5]
    bounds = ([1.0, 0.3, -3.0, 0.3, 0.0], [40.0, 3.0, 3.0, 3.0, 1e-3])

    popt, pcov = curve_fit(wrapper, (alpha_rad, Fz_N), Fy, p0=p0,
                            bounds=bounds, maxfev=20000)
    B_fit, C_fit, E_fit, mu_fit, c2_fit = popt
    load_sensitivity_fit = c2_fit * FZ_NOM / mu_fit

    Fy_pred = fit_model(alpha_rad, Fz_N, *popt)
    rmse = np.sqrt(np.mean((Fy_pred - Fy) ** 2))
    max_err = np.max(np.abs(Fy_pred - Fy))
    # Normalize against the real data's overall force range, not peak,
    # so error is judged relative to the span actually being fit.
    data_range = Fy.max() - Fy.min()
    print("Fitted PacejkaTireModel parameters (Fz_nom fixed at "
          f"{FZ_NOM:.0f} N):")
    print(f"  B = {B_fit:.3f}")
    print(f"  C = {C_fit:.3f}")
    print(f"  E = {E_fit:.3f}")
    print(f"  mu = {mu_fit:.3f}")
    print(f"  load_sensitivity = {load_sensitivity_fit:.4f}")
    print(f"RMSE across full grid: {rmse:.1f} N  "
          f"({100*rmse/data_range:.2f}% of data range)")
    print(f"Max abs error: {max_err:.1f} N")

    # Per-load breakdown, since global-fit error is not uniform across Fz.
    print("\nPer-load fit quality:")
    for Fz_kN in Fz_kN_vals:
        mask = np.isclose(Fz_N, Fz_kN * 1000.0)
        e = np.sqrt(np.mean((Fy_pred[mask] - Fy[mask]) ** 2))
        print(f"  Fz={Fz_kN:.0f} kN: RMSE = {e:.1f} N")

    # Plot: fitted vs real at a few representative loads.
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_loads_kN = [2.0, 4.0, 6.0, 8.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_loads_kN)))
    for Fz_kN, color in zip(plot_loads_kN, colors):
        Fy_true = bakker87_lateral_force(alpha_deg_vals, Fz_kN)
        Fy_fit = fit_model(np.deg2rad(alpha_deg_vals),
                            np.full_like(alpha_deg_vals, Fz_kN * 1000.0),
                            *popt)
        ax.plot(alpha_deg_vals, Fy_true, color=color,
                label=f"Real (Bakker '87), Fz={Fz_kN:.0f}kN")
        ax.plot(alpha_deg_vals, Fy_fit, color=color, linestyle="--",
                label=f"Fitted model, Fz={Fz_kN:.0f}kN")
    ax.set_xlabel("Slip angle [deg]")
    ax.set_ylabel("Lateral force Fy [N]")
    ax.set_title("Fitted PacejkaTireModel vs. real Bakker/Pacejka/Nyborg (1987) tire")
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("tire_fit_comparison.png", dpi=150)
    print("\nSaved plot to tire_fit_comparison.png")

    return popt, load_sensitivity_fit


if __name__ == "__main__":
    main()
