# Bicycle Model — Step 1

Nonlinear planar bicycle model with RK4 integration, swappable
linear/Pacejka tire models, and a step-steer validation test.

## Files
- `tire_models.py` — `LinearTireModel` and `PacejkaTireModel`, same interface
- `vehicle.py` — `VehicleParams`, `BicycleModel` (slip angles, EOM, RK4 step)
- `simulate.py` — step-steer test; run with `python simulate.py`

## Validation status
- **Linear tire model: verified.** Simulated steady-state yaw rate matches
  the closed-form small-angle solution to within ~5%, and that residual
  shrinks as the steer angle decreases (it's the `atan2`-vs-small-angle
  gap between the simulation and the check itself, not a simulation bug).
- **Pacejka tire model: qualitatively correct, not independently verified.**
  It tracks the linear model near zero slip (as expected) and diverges at
  higher slip angles because `B, C, D, E` weren't tuned to match the
  linear model's `c_alpha`. There's no simple closed-form check for the
  nonlinear case — the next real validation step is comparing against
  published Pacejka coefficient sets and force curves from Milliken/Rajamani
  before trusting this for anything quantitative.

## Known simplifications (by design, for step 1)
- Longitudinal load transfer is a one-step-lag estimate (`ax_est` passed in
  externally) — there's no drivetrain/braking model yet, so `Fxf=Fxr=0` in
  the test.
- Lateral load transfer isn't modeled — the bicycle model has no left/right
  split. That needs the 4-wheel extension (step 2 in the roadmap).
- No terrain/surface model yet — `mu` is a flat constant.

## Next steps
1. Fit/verify Pacejka coefficients against a real published tire dataset.
2. Extend to 4 wheels for lateral load transfer and per-corner slip.
3. Add a longitudinal force model (engine torque → wheel torque → Fx) so
   `Fxf`/`Fxr` aren't hardcoded to zero.
4. Port the validated `BicycleModel.step_rk4` logic into a Unity
   `FixedUpdate` loop driving `Rigidbody.AddForceAtPosition`.
