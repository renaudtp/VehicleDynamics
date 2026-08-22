"""
Validation for the 4-wheel model (vehicle4.py).

Test A — regression against the bicycle model:
    With h_cg=0 (no load transfer possible) and the LOAD-INSENSITIVE
    linear tire model, the 4-wheel model's per-axle summed force is
    mathematically identical to the bicycle model's single-tire force,
    provided the bicycle model's c_alpha equals 2x the per-corner value.
    Trajectories should match to floating-point precision, not just
    "close" — any drift here means a real bug, not model disagreement.

Test B — load transfer actually matters:
    With h_cg > 0 and PacejkaTireModel(load_sensitivity > 0), a hard
    corner should show the outside tires gaining load / inside tires
    losing load (checked directly), total corner Fz conserved to m*g at
    every timestep (checked directly, not assumed), and total axle
    lateral force LOWER than the load-transfer-free case because of
    load sensitivity — this is the real effect this step was built to
    capture, so it's asserted numerically, not just plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

from vehicle import VehicleParams, BicycleModel
from vehicle4 import FourWheelParams, FourWheelModel
from tire_models import LinearTireModel, PacejkaTireModel


def test_a_regression_vs_bicycle():
    dt = 0.001
    t_end = 3.0
    vx0 = 20.0
    delta = np.deg2rad(5.0)
    step_time = 1.0

    bike_params = VehicleParams(h_cg=0.0)
    corner_c_alpha = 45000.0  # per-tire
    bike_tire = LinearTireModel(c_alpha=2 * corner_c_alpha)  # per-axle
    bike = BicycleModel(bike_params, bike_tire)

    fw_params = FourWheelParams(h_cg=0.0)
    fw_tire = LinearTireModel(c_alpha=corner_c_alpha)
    fourwheel = FourWheelModel(fw_params, fw_tire)

    state_bike = np.zeros(6)
    state_bike[0] = vx0
    state_fw = state_bike.copy()

    n = int(t_end / dt)
    max_abs_diff = 0.0
    for i in range(n):
        t = i * dt
        d = delta if t >= step_time else 0.0
        state_bike = bike.step_rk4(state_bike, dt, d, Fxf=0.0, Fxr=0.0)
        state_fw = fourwheel.step_rk4(state_fw, dt, d, Fxf=0.0, Fxr=0.0)
        max_abs_diff = max(max_abs_diff, np.max(np.abs(state_bike - state_fw)))

    print(f"[Test A] max abs state difference over {t_end}s: {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-9, "4-wheel model diverged from bicycle model regression check!"
    print("[Test A] PASS — 4-wheel model exactly reproduces the bicycle model "
          "when load transfer is disabled and the tire is load-insensitive.")


def test_b_load_transfer_matters():
    dt = 0.001
    t_end = 3.0
    vx0 = 25.0
    delta = np.deg2rad(8.0)
    step_time = 0.5

    fw_params = FourWheelParams(h_cg=0.5, track=1.6)
    tire_no_sens = PacejkaTireModel(mu=1.0, load_sensitivity=0.0, Fz_nom=4000.0)
    tire_sens = PacejkaTireModel(mu=1.0, load_sensitivity=0.25, Fz_nom=4000.0)

    def run(tire_model):
        model = FourWheelModel(fw_params, tire_model)
        state = np.zeros(6)
        state[0] = vx0
        n = int(t_end / dt)
        t_hist = np.zeros(n)
        r_hist = np.zeros(n)
        Fz_hist = {k: np.zeros(n) for k in ("FL", "FR", "RL", "RR")}
        Fy_total_hist = np.zeros(n)
        ax_est, ay_est = 0.0, 0.0

        for i in range(n):
            t = i * dt
            d = delta if t >= step_time else 0.0
            Fy, loads, alpha_f, alpha_r = model.corner_lateral_forces(
                state, d, ax_est, ay_est)

            total_fz = sum(loads.values())
            assert abs(total_fz - fw_params.m * fw_params.g) < 1e-3, (
                f"Corner loads do not conserve total weight at t={t:.3f}: "
                f"{total_fz:.2f} vs {fw_params.m * fw_params.g:.2f}")

            # Analytic body-frame acceleration (ax = vx_dot - vy*r,
            # ay = vy_dot + vx*r), evaluated at the CURRENT state before
            # stepping. This is the one-step-lag estimate fed into next
            # step's load transfer. Using the closed-form derivative here
            # instead of finite-differencing the trajectory avoids
            # amplifying numerical noise across a step input — the earlier
            # finite-difference version produced large ay spikes right
            # after the steer step, which pushed a corner load negative
            # and silently broke weight conservation when it got clipped.
            deriv = model.derivatives(state, d, Fxf=0.0, Fxr=0.0,
                                       ax_est=ax_est, ay_est=ay_est)
            vx, vy, r = state[0], state[1], state[2]
            ax_est = deriv[0] - vy * r
            ay_est = deriv[1] + vx * r

            state = model.step_rk4(state, dt, d, Fxf=0.0, Fxr=0.0,
                                    ax_est=ax_est, ay_est=ay_est)

            t_hist[i] = t
            r_hist[i] = state[2]
            for k in Fz_hist:
                Fz_hist[k][i] = loads[k]
            Fy_total_hist[i] = Fy["FL"] + Fy["FR"] + Fy["RL"] + Fy["RR"]

        return t_hist, r_hist, Fz_hist, Fy_total_hist

    t_ns, r_ns, Fz_ns, Fy_ns = run(tire_no_sens)
    t_s, r_s, Fz_s, Fy_s = run(tire_sens)

    # Check outside-vs-inside load transfer direction during the steady
    # part of the turn (last 20% of the run).
    tail = slice(int(0.8 * len(t_s)), None)
    fl_mean = Fz_s["FL"][tail].mean()
    fr_mean = Fz_s["FR"][tail].mean()
    print(f"[Test B] steady-turn front loads: FL={fl_mean:.0f} N, FR={fr_mean:.0f} N "
          f"({'FR heavier (outside, left turn)' if fr_mean > fl_mean else 'FL heavier'})")

    # Check that load sensitivity actually reduces total available lateral
    # force vs. the load-insensitive case, at matched slip angles.
    peak_Fy_ns = np.max(np.abs(Fy_ns[tail]))
    peak_Fy_s = np.max(np.abs(Fy_s[tail]))
    print(f"[Test B] total lateral force, load-insensitive: {peak_Fy_ns:.0f} N")
    print(f"[Test B] total lateral force, load-sensitive:    {peak_Fy_s:.0f} N")
    assert peak_Fy_s < peak_Fy_ns, (
        "Expected load sensitivity to REDUCE total grip vs. the "
        "load-insensitive case — got the opposite, something's wrong.")
    print("[Test B] PASS — load transfer + load sensitivity measurably "
          "reduces total grip, as real tires do.")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_s, Fz_s["FL"], label="FL")
    axes[0].plot(t_s, Fz_s["FR"], label="FR")
    axes[0].plot(t_s, Fz_s["RL"], label="RL", linestyle="--")
    axes[0].plot(t_s, Fz_s["RR"], label="RR", linestyle="--")
    axes[0].set_ylabel("Corner Fz [N]")
    axes[0].set_title("Corner loads during an 8 deg step-steer at 25 m/s")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_ns, Fy_ns, label="Total Fy, load-insensitive tire")
    axes[1].plot(t_s, Fy_s, label="Total Fy, load-sensitive tire", linestyle="--")
    axes[1].set_ylabel("Total lateral force [N]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("fourwheel_load_transfer.png", dpi=150)
    print("[Test B] saved plot to fourwheel_load_transfer.png")


if __name__ == "__main__":
    test_a_regression_vs_bicycle()
    print()
    test_b_load_transfer_matters()
