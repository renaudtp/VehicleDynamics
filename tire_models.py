"""
Tire force models.

Two interchangeable models are provided:
  - LinearTireModel   : Fy = -C_alpha * alpha   (small-slip approximation)
  - PacejkaTireModel  : full nonlinear "Magic Formula"

Both expose the same call signature so they can be swapped in the
vehicle model without changing any other code:

    Fy = tire_model.lateral_force(alpha, Fz)
    Fx = tire_model.longitudinal_force(kappa, Fz)
"""

import numpy as np


class LinearTireModel:
    """Small-slip-angle linear tire model. Good for validating the
    bicycle-model kinematics/integration before trusting Pacejka.
    """

    def __init__(self, c_alpha):
        """
        c_alpha : cornering stiffness [N/rad], positive scalar.
                  Typical passenger tire: 8e4 - 1.2e5 N/rad per tire,
                  scale by number of tires on the axle if lumping.
        """
        self.c_alpha = c_alpha

    def lateral_force(self, alpha, Fz=None):
        # Fz accepted for interface compatibility but unused (linear model
        # does not scale with load).
        # Sign convention matches the slip-angle definition used in
        # vehicle.py (alpha_f = delta - atan2(vy+a*r, vx)); do not flip
        # this sign without re-deriving stability (see simulate.py notes).
        return self.c_alpha * alpha

    def longitudinal_force(self, kappa, Fz=None, c_kappa=None):
        c_kappa = c_kappa if c_kappa is not None else self.c_alpha
        return c_kappa * kappa


class PacejkaTireModel:
    """Simplified Pacejka '94-style Magic Formula.

    F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
    D = mu * Fz   (peak force scales with normal load and friction)

    Default B, C, E are representative passenger-tire values, NOT a
    fitted dataset for a real tire. Treat them as tunable parameters.
    """

    def __init__(self, B=10.0, C=1.9, E=0.97, mu=1.0):
        self.B = B
        self.C = C
        self.E = E
        self.mu = mu

    def _magic_formula(self, x, Fz):
        B, C, E = self.B, self.C, self.E
        D = self.mu * Fz
        Bx = B * x
        return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

    def lateral_force(self, alpha, Fz):
        """alpha in radians, Fz in N (positive = normal load on this tire).
        Sign convention matches the slip-angle definition in vehicle.py —
        see the note in LinearTireModel.lateral_force."""
        return self._magic_formula(alpha, Fz)

    def longitudinal_force(self, kappa, Fz):
        """kappa = slip ratio (dimensionless), Fz in N."""
        return self._magic_formula(kappa, Fz)
