"""
Reference tire lateral-force model using REAL published coefficients.

Source: H. Pacejka, E. Bakker, L. Nyborg, "Tyre modelling for use in
vehicle dynamics studies," SAE Paper 870421, 1987. Coefficient table as
reproduced in Stanford's "The tire model" course note
(www-cdr.stanford.edu/dynamic/bywire/tires.pdf), which cites the same
SAE paper. This is a real fitted passenger-car tire, NOT an off-road
tire — treat this as the best available real ground truth for the
generic Pacejka SHAPE, not as evidence about off-road behavior.

IMPORTANT UNIT NOTE (differs from tire_models.py):
  - Fz is in kN here (not N)
  - alpha is in DEGREES here (not radians) — this is the original '87
    formula's convention, confirmed by the source's cornering-stiffness
    units ("force per degree") and its plotted axis ("slip angle (deg)")
  - Output Fy is in N
Do not mix these units with vehicle.py/tire_models.py without
converting — fit_tire.py handles that conversion explicitly.
"""

import numpy as np

# Table 1 coefficients for Fy, load influence only (a1..a8).
# Camber coefficients (a9..a13) are omitted — this module assumes
# gamma = 0 (no camber), which zeroes every camber-dependent term.
A1, A2, A3, A4, A5, A6, A7, A8 = -22.1, 1011.0, 1078.0, 1.82, 0.208, 0.000, -0.354, 0.707
C_FY = 1.30  # shape factor, given directly by the source (not fitted per-load)


def bakker87_lateral_force(alpha_deg, Fz_kN):
    """Real Bakker/Pacejka/Nyborg (1987) Fy formula, gamma=0.

    alpha_deg : slip angle in DEGREES (scalar or array)
    Fz_kN     : normal load in kN (scalar or array)
    returns   : lateral force in N
    """
    alpha_deg = np.asarray(alpha_deg, dtype=float)
    Fz_kN = np.asarray(Fz_kN, dtype=float)

    D = A1 * Fz_kN**2 + A2 * Fz_kN
    BCD = A3 * np.sin(2 * np.arctan(Fz_kN / A4)) * (1 - A5 * 0.0)  # gamma=0
    B = BCD / (C_FY * D)
    E = A6 * Fz_kN**2 + A7 * Fz_kN + A8

    Bx = B * alpha_deg
    return D * np.sin(C_FY * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
