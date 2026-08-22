"""
Step-steer validation test.

Holds vx constant, applies a step input in steering angle at t=1s, and
plots yaw rate / lateral velocity response. This is the standard first
sanity check: the linear-tire response should settle to a steady-state
yaw rate close to the classic bicycle-model formula

    r_ss = vx * delta / (L + K * vx^2)

where L = a + b and K is the understeer gradient. Use this to confirm
your integration and equations are correct BEFORE trusting the
nonlinear Pacejka results, since Pacejka has no simple closed-form
check.
"""

import numpy as np
import matplotlib.pyplot as plt

from vehicle import VehicleParams, BicycleModel
from tire_models import LinearTireModel, PacejkaTireModel


def run_step_steer(tire_model, label, vx0=20.0, delta_deg=5.0,
                    t_end=4.0, dt=0.001, step_time=1.0):
    params = VehicleParams()
    model = BicycleModel(params, tire_model)

    state = np.array([vx0, 0.0, 0.0, 0.0, 0.0, 0.0])  # vx,vy,r,psi,X,Y
    delta = np.deg2rad(delta_deg)

    n_steps = int(t_end / dt)
    t_hist = np.zeros(n_steps)
    r_hist = np.zeros(n_steps)
    vy_hist = np.zeros(n_steps)
    ax_est = 0.0  # constant-speed test: no longitudinal load transfer

    for i in range(n_steps):
        t = i * dt
        current_delta = delta if t >= step_time else 0.0
        state = model.step_rk4(state, dt, current_delta, Fxf=0.0, Fxr=0.0,
                                ax_est=ax_est)
        t_hist[i] = t
        r_hist[i] = state[2]
        vy_hist[i] = state[1]

    # Analytical steady-state yaw rate for the LINEAR model, for comparison.
    # Closed form from solving vy_dot=0, r_dot=0 on the small-angle-linearized
    # 2-state model (equal front/rear cornering stiffness Ca, general a != b):
    #
    #   r_ss = Ca*delta*vx*(a+b) / (Ca*(a+b)^2 - (a-b)*m*vx^2)
    #
    # This uses the SAME small-angle slip approximation the linear tire model
    # implicitly assumes; the simulation itself uses exact atan2, so expect
    # a few percent difference at larger steer angles - that gap is a
    # feature of the check, not a bug, and should shrink as delta -> 0.
    L = params.a + params.b
    if isinstance(tire_model, LinearTireModel):
        c_alpha = tire_model.c_alpha
        num = c_alpha * delta * vx0 * L
        den = c_alpha * L ** 2 - (params.a - params.b) * params.m * vx0 ** 2
        r_ss_analytical = num / den
        print(f"[{label}] analytical steady-state yaw rate: "
              f"{r_ss_analytical:.4f} rad/s")

    print(f"[{label}] simulated steady-state yaw rate:  {r_hist[-1]:.4f} rad/s")
    return t_hist, r_hist, vy_hist


if __name__ == "__main__":
    linear_tire = LinearTireModel(c_alpha=90000.0)
    pacejka_tire = PacejkaTireModel(B=10.0, C=1.9, E=0.97, mu=1.0)

    t_lin, r_lin, vy_lin = run_step_steer(linear_tire, "Linear")
    t_pac, r_pac, vy_pac = run_step_steer(pacejka_tire, "Pacejka")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t_lin, r_lin, label="Linear tire")
    axes[0].plot(t_pac, r_pac, label="Pacejka tire", linestyle="--")
    axes[0].set_ylabel("Yaw rate r [rad/s]")
    axes[0].set_title("Step-steer response (5 deg step at t=1s, vx=20 m/s)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_lin, vy_lin, label="Linear tire")
    axes[1].plot(t_pac, vy_pac, label="Pacejka tire", linestyle="--")
    axes[1].set_ylabel("Lateral velocity vy [m/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("step_steer_response.png", dpi=150)
    print("Saved plot to step_steer_response.png")
