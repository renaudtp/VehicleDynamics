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

---

# Step 3 — Fitting against a real published tire dataset

## Files
- `reference_tire.py` — the real Bakker/Pacejka/Nyborg (1987) Fy formula
  (SAE Paper 870421 coefficients, as reproduced in a Stanford course note
  citing that paper). This is a real fitted passenger-car tire — **not**
  off-road, but it's real measured-tire behavior rather than assumed
  constants, which is what this step needed.
- `fit_tire.py` — fits `tire_models.PacejkaTireModel`'s (B, C, E, mu,
  load_sensitivity) against the reference tire via `scipy.optimize.curve_fit`
  across a 2-8 kN load range and ±15° slip angle. Run with
  `python fit_tire.py`.

## Result
```
B = 7.635, C = 0.855, E = 0.169, mu = 1.840, load_sensitivity = 0.2803
(Fz_nom fixed at 4000 N)
RMSE across full grid: 396 N (3.4% of data range)
```
The fitted `load_sensitivity ≈ 0.28` is in the same neighborhood as the
`≈0.15` estimated by hand from Milliken's race-tire data in the previous
step — different tire, same real phenomenon, similar order of magnitude.

## Where the fit is trustworthy and where it isn't
Overall RMSE (3.4%) looks fine, but it's not uniform — per-load RMSE
ranges from 170N (mid-loads, 5-6kN) to ~500N (both extremes, 2kN and
8kN). The comparison plot shows why: the real tire's saturation point
and post-peak shape *change* with load (it saturates earlier and flatter
at low load), while this simplified model only lets the **peak** (`D`)
vary with load — `B`, `C`, `E` are fit as single global constants. At
mid-loads that's a good approximation; at the load extremes it visibly
isn't (the fitted curve keeps climbing at 2kN instead of plateauing like
the real tire does).

**Practical takeaway:** trust this fit most for loads in roughly the
3-7kN per-tire range (normal operating loads for the vehicle parameters
used in steps 1-2), and treat results at very light or very heavy corner
loads — exactly the loads produced by *large* weight transfer, off-road
articulation, or a wheel nearly lifting — with real skepticism. If that
matters for your use case, the fix is a load-dependent `B`/`E` (matching
the real formula's structure) rather than a global fit, which is a
natural next refinement rather than a fix to a bug.

## Honest limitation carried over from the last two steps
This is real tire data, but it's a dry-pavement passenger tire from the
1980s — still not off-road. Nothing in this step changes the earlier
conclusion: getting *off-road* accuracy requires the terramechanics model
(Bekker/Wong), not further tuning of a hard-surface Pacejka fit. What
this step does provide is a real, defensible baseline for the on-road
portion of the sim (and for validating the terrain model against, once
built — you'll want "hard surface = matches this fit" as a sanity check).

## Next steps
1. Terramechanics (Bekker/Wong pressure-sinkage) for deformable terrain
   — the actual off-road differentiator.
2. Load-dependent `B`/`E` in `PacejkaTireModel`, if accuracy at load
   extremes turns out to matter for your test scenarios.
3. Add a longitudinal force model (engine torque → wheel torque → Fx).
4. Port to Unity's `FixedUpdate` loop.
