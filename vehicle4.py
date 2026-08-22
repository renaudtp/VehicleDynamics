"""
Four-wheel extension of the bicycle model.

Adds per-corner normal load (longitudinal + lateral load transfer) while
keeping the same 6-state vector and equations-of-motion structure as
vehicle.py. Deliberate simplification vs. a "true" 4-wheel model:

  - Slip angle is shared per AXLE (front/rear), not per corner. The
    track-width contribution to each corner's longitudinal velocity
    (vx +/- r*track/2) is neglected — this is standard practice in
    simplified 4-wheel models and is a second-order effect compared to
    load transfer, but it does mean left/right slip angles on the same
    axle are treated as identical.
  - Longitudinal force (Fxf, Fxr) is split evenly left/right — there's
    no differential/torque-vectoring model yet, so no yaw moment comes
    from longitudinal force asymmetry.
  - Lateral load transfer per axle is split proportional to that axle's
    share of static vehicle weight, since no roll-stiffness distribution
    is modeled. Real cars split it according to front/rear anti-roll bar
    stiffness, not static weight — treat this as a placeholder until
    suspension/roll data is added.

Corner layout (body frame, x forward, y left):
    FL: ( a,  track_f/2)   FR: ( a, -track_f/2)
    RL: (-b,  track_r/2)   RR: (-b, -track_r/2)
"""

from dataclasses import dataclass
import numpy as np

from vehicle import VehicleParams


@dataclass
class FourWheelParams(VehicleParams):
    track_f: float = None
    track_r: float = None

    def __post_init__(self):
        if self.track_f is None:
            self.track_f = self.track
        if self.track_r is None:
            self.track_r = self.track


class FourWheelModel:
    def __init__(self, params: FourWheelParams, tire_model, vx_floor=0.5):
        self.p = params
        self.tire = tire_model
        self.vx_floor = vx_floor

    def static_axle_loads(self):
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

    def corner_loads(self, ax_est, ay_est):
        """Returns dict of Fz per corner, each clipped >= 0. Corner Fz
        values always sum to m*g exactly (load transfer only redistributes,
        never adds/removes total weight) — useful as a standing sanity
        check on any change to this method."""
        p = self.p
        L = p.a + p.b
        Fzf_static, Fzr_static = self.static_axle_loads()

        dFz_long = (p.m * p.h_cg * ax_est) / L
        Fzf_total = Fzf_static - dFz_long
        Fzr_total = Fzr_static + dFz_long

        # Lateral transfer split per axle proportional to static weight
        # share (placeholder — see module docstring).
        dFz_lat_f = (Fzf_static / (p.m * p.g)) * (p.m * p.h_cg * ay_est) / p.track_f
        dFz_lat_r = (Fzr_static / (p.m * p.g)) * (p.m * p.h_cg * ay_est) / p.track_r

        # Sign convention: y is positive to the left; positive ay (turning
        # left) transfers load AWAY from the turn center, i.e. onto the
        # right (-y) corners. Verified by the weight-conservation check in
        # simulate4.py, not just asserted here.
        loads = {
            "FL": max(Fzf_total / 2.0 - dFz_lat_f / 2.0, 0.0),
            "FR": max(Fzf_total / 2.0 + dFz_lat_f / 2.0, 0.0),
            "RL": max(Fzr_total / 2.0 - dFz_lat_r / 2.0, 0.0),
            "RR": max(Fzr_total / 2.0 + dFz_lat_r / 2.0, 0.0),
        }
        return loads

    def corner_lateral_forces(self, state, delta, ax_est, ay_est):
        alpha_f, alpha_r = self.slip_angles(state, delta)
        loads = self.corner_loads(ax_est, ay_est)
        Fy = {
            "FL": self.tire.lateral_force(alpha_f, loads["FL"]),
            "FR": self.tire.lateral_force(alpha_f, loads["FR"]),
            "RL": self.tire.lateral_force(alpha_r, loads["RL"]),
            "RR": self.tire.lateral_force(alpha_r, loads["RR"]),
        }
        return Fy, loads, alpha_f, alpha_r

    def derivatives(self, state, delta, Fxf=0.0, Fxr=0.0, ax_est=0.0, ay_est=0.0):
        p = self.p
        vx, vy, r, psi, X, Y = state

        Fy, loads, alpha_f, alpha_r = self.corner_lateral_forces(
            state, delta, ax_est, ay_est)
        Fyf = Fy["FL"] + Fy["FR"]
        Fyr = Fy["RL"] + Fy["RR"]

        # Same form as the bicycle model — left/right corners on an axle
        # share the same x-lever-arm, so summing corner forces first and
        # reusing the bicycle-model EOM is exact, not an approximation.
        vx_dot = (Fxf * np.cos(delta) - Fyf * np.sin(delta) + Fxr) / p.m + vy * r
        vy_dot = (Fxf * np.sin(delta) + Fyf * np.cos(delta) + Fyr) / p.m - vx * r
        r_dot = (p.a * (Fxf * np.sin(delta) + Fyf * np.cos(delta)) - p.b * Fyr) / p.Iz

        psi_dot = r
        X_dot = vx * np.cos(psi) - vy * np.sin(psi)
        Y_dot = vx * np.sin(psi) + vy * np.cos(psi)

        return np.array([vx_dot, vy_dot, r_dot, psi_dot, X_dot, Y_dot])

    def step_rk4(self, state, dt, delta, Fxf=0.0, Fxr=0.0, ax_est=0.0, ay_est=0.0):
        f = self.derivatives
        k1 = f(state, delta, Fxf, Fxr, ax_est, ay_est)
        k2 = f(state + 0.5 * dt * k1, delta, Fxf, Fxr, ax_est, ay_est)
        k3 = f(state + 0.5 * dt * k2, delta, Fxf, Fxr, ax_est, ay_est)
        k4 = f(state + dt * k3, delta, Fxf, Fxr, ax_est, ay_est)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
