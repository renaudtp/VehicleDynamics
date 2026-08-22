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
2. Add a longitudinal force model (engine torque → wheel torque → Fx) so
   `Fxf`/`Fxr` aren't hardcoded to zero.
3. Port the validated model logic into a Unity `FixedUpdate` loop driving
   `Rigidbody.AddForceAtPosition`.

---

# Step 2 — Four-wheel model with load transfer

## Files
- `vehicle4.py` — `FourWheelParams`, `FourWheelModel` (per-corner Fz,
  shared per-axle slip angle, same EOM structure as step 1)
- `simulate4.py` — two validation tests; run with `python simulate4.py`

## Design choice: load sensitivity is opt-in
The simplified Pacejka model from step 1 has peak force `D = mu*Fz`,
which is exactly linear in load. That means splitting an axle's load
unevenly between two tires at the same slip angle produces the identical
summed force as one tire carrying the total load — load transfer would
do literally nothing. Real tires don't work that way: grip grows
sub-linearly with load. `PacejkaTireModel` now takes a `load_sensitivity`
parameter (default `0.0`, which exactly reproduces step 1's results) that
makes `D` concave in `Fz` when set above zero.

## Validation status
- **Test A (regression, PASS, exact):** with load transfer disabled
  (`h_cg=0`) and a load-insensitive linear tire, the 4-wheel model
  reproduces the bicycle model's trajectory to floating-point precision
  (max state diff `0.00e+00`), not just "close." This confirms the
  per-corner EOM reduction is algebraically exact, not an approximation.
- **Test B (load transfer effect, PASS):** an 8° step-steer at 25 m/s
  with `h_cg=0.5` shows the outside-front corner loading up and the
  inside corners unloading, total corner `Fz` conserved to `m*g` at
  every timestep (asserted, not assumed), and — with `load_sensitivity`
  turned on — total available lateral force drops from ~14,680N to
  ~10,980N at the same steer input, purely from load transfer
  redistributing weight to a concave force-load curve.
- **Bug caught during this step:** the initial version estimated
  `ax`/`ay` for load transfer by finite-differencing velocity, which
  amplified noise right after the step input and pushed a corner's
  computed load negative; clipping that to zero silently broke the
  weight-conservation invariant. Fixed by computing `ax`/`ay`
  analytically from the model's own derivative (`vx_dot - vy*r`,
  `vy_dot + vx*r`) instead of differencing the trajectory.

## Known simplifications (by design)
- Slip angle is shared per axle, not per corner — the track-width
  contribution to each corner's longitudinal velocity is neglected
  (standard simplification, second-order vs. load transfer effects).
- Lateral load transfer is split per axle by static weight share, not
  roll-stiffness distribution — there's no suspension/roll model yet.
  Real cars split it by front/rear anti-roll bar stiffness.
- `Fxf`/`Fxr` still split evenly left/right — no differential or
  torque-vectoring yet, so no yaw moment from longitudinal asymmetry.
- No wheel lift-off handling — if computed load transfer would exceed
  a corner's static share, it's clamped at zero rather than modeled as
  an actual airborne wheel with redistributed contact loads.

## Next steps
1. Fit/verify Pacejka coefficients (and `load_sensitivity`) against a
   real published tire dataset.
2. Add roll-stiffness-based lateral load transfer distribution once
   suspension parameters are introduced.
3. Add a longitudinal force model (engine torque → wheel torque → Fx).
4. Port to Unity's `FixedUpdate` loop driving `Rigidbody.AddForceAtPosition`
   at each of the four contact points.
