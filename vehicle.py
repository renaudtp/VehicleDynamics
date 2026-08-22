"""
Nonlinear bicycle model with RK4 integration.

State vector (body-frame + global pose):
    x = [vx, vy, r, psi, X, Y]

    vx, vy : longitudinal / lateral velocity in the body frame [m/s]
    r      : yaw rate [rad/s]
    psi    : heading angle [rad]
    X, Y   : global position [m]

Inputs:
    delta : front steer angle [rad]
    Fxf, Fxr : front/rear longitudinal tire force [N] (from a
               drivetrain/braking model you supply externally)

The tire model (linear or Pacejka) is injected as a dependency so the
same integrator works for either.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    m: float = 1500.0        # mass [kg]
    Iz: float = 2500.0       # yaw moment of inertia [kg*m^2]
    a: float = 1.2           # CG to front axle [m]
    b: float = 1.6           # CG to rear axle [m]
    h_cg: float = 0.5        # CG height [m], used for load transfer
    track: float = 1.6       # track width [m], used for lateral load transfer
    g: float = 9.81


class BicycleModel:
    def __init__(self, params: VehicleParams, tire_model, vx_floor=0.5):
        self.p = params
        self.tire = tire_model
        # Prevent division blow-up at near-zero forward speed.
        self.vx_floor = vx_floor

    def static_loads(self):
        p = self.p
        L = p.a + p.b
        Fzf = p.m * p.g * p.b / L
        Fzr = p.m * p.g * p.a / L
        return Fzf, Fzr

    def slip_angles(self, state, delta):
        vx, vy, r, psi, X, Y = state
        vx_safe = max(vx, self.vx_floor)
        alpha_f = delta - np.arctan2(vy + self.p.a * r, vx_safe)
        alpha_r = -np.arctan2(vy - self.p.b * r, vx_safe)
        return alpha_f, alpha_r

    def derivatives(self, state, delta, Fxf, Fxr, ax_est=0.0, ay_est=0.0):
        """Compute state derivative. ax_est/ay_est are the previous step's
        accelerations, used only to estimate load transfer (a one-step-lag
        approximation is standard practice and fine at simulation rates).
        """
        p = self.p
        vx, vy, r, psi, X, Y = state
        vx_safe = max(vx, self.vx_floor)

        alpha_f, alpha_r = self.slip_angles(state, delta)

        # --- load transfer (longitudinal only here; lateral splits L/R
        #     which requires the 4-wheel extension, not this 2D model) ---
        Fzf_static, Fzr_static = self.static_loads()
        L = p.a + p.b
        dFz_long = (p.m * p.h_cg * ax_est) / L
        Fzf = max(Fzf_static - dFz_long, 0.0)
        Fzr = max(Fzr_static + dFz_long, 0.0)

        Fyf = self.tire.lateral_force(alpha_f, Fzf)
        Fyr = self.tire.lateral_force(alpha_r, Fzr)

        # --- equations of motion ---
        vx_dot = (Fxf * np.cos(delta) - Fyf * np.sin(delta) + Fxr) / p.m + vy * r
        vy_dot = (Fxf * np.sin(delta) + Fyf * np.cos(delta) + Fyr) / p.m - vx * r
        r_dot = (p.a * (Fxf * np.sin(delta) + Fyf * np.cos(delta)) - p.b * Fyr) / p.Iz

        psi_dot = r
        X_dot = vx * np.cos(psi) - vy * np.sin(psi)
        Y_dot = vx * np.sin(psi) + vy * np.cos(psi)

        return np.array([vx_dot, vy_dot, r_dot, psi_dot, X_dot, Y_dot])

    def step_rk4(self, state, dt, delta, Fxf, Fxr, ax_est=0.0, ay_est=0.0):
        """One RK4 integration step. delta/Fxf/Fxr are held constant
        across the substeps (standard zero-order-hold assumption for a
        fixed-timestep simulation loop)."""
        f = self.derivatives
        k1 = f(state, delta, Fxf, Fxr, ax_est, ay_est)
        k2 = f(state + 0.5 * dt * k1, delta, Fxf, Fxr, ax_est, ay_est)
        k3 = f(state + 0.5 * dt * k2, delta, Fxf, Fxr, ax_est, ay_est)
        k4 = f(state + dt * k3, delta, Fxf, Fxr, ax_est, ay_est)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
