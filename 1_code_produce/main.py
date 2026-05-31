"""
Main experiment runner for Sun et al. (2024) replication.

Uses a unified ``simulate()`` loop driven by a generic ``Controller``.
Each controller defines its own ``update_interval``; the loop simply
asks the controller for an action at every T_s = 10 s step.

Baselines implemented
---------------------
1.  No Control          (NoControlController)
2.  Standalone MPC      (MPCController, T_c = 300 s, N_p,c = 2)

Future extensions
-----------------
3.  High-Frequency MPC  (HFMPCController, T_c = 60 s)
4.  DDPG (1-step / 10-step TD)
"""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from metanet.metanet_env import MetanetEnv
from config import (
    METANET_PARAMS, GUSSIAN_NOISE,
    TS, N_CTRL, M_STEP, NC_HF, M_STEP_HF, SEED, N_MULTI_START
)
from util import create_demand_1, create_demand_2

from controllers.base_controller import Controller
from controllers.no_control import NoControlController
from controllers.mpc_controller import MPCController
from controllers.ddpg_controller import DDPGController
from controllers.mpc_drl_controller import MPCDRLController


# ======================================================================
# Generic simulation loop
# ======================================================================

def simulate(
    env: MetanetEnv,
    controller: Controller,
    desc: str = "Simulation",
) -> dict:
    """
    Run a full 9000 s episode stepping the environment at T_s = 10 s.

    The controller decides internally whether to recompute or reuse
    its cached action at each step.

    Returns
    -------
    dict with keys:
        tts_all, twt_all, min_speed, queue_violation,
        Reward, Mean/Max computation time [s],
        rho_log, v_log, w_log, action_log
    """
    obs, _ = env.reset()
    controller.reset()

    rho_log, v_log, w_log, action_log = [], [], [], []
    solve_times: list[float] = []

    for t in tqdm(range(int(N_CTRL)), desc=desc):
        # Time the controller only when it actually re-solves
        if t % controller.update_interval == 0:
            t0 = time.perf_counter()
            action = controller.get_action(obs, t)
            dt = time.perf_counter() - t0
            solve_times.append(dt)
        else:
            action = controller.get_action(obs, t)

        obs, reward, terminated, truncated, info = env.step(action)

        rho_t, v_t, w_t, _, _ = env.parse_obs(obs)
        rho_log.append(rho_t)
        v_log.append(v_t)
        w_log.append(w_t)
        action_log.append(action.copy())

        if terminated or truncated:
            break

    mean_time = float(np.mean(solve_times)) if solve_times else None
    max_time  = float(np.max(solve_times))  if solve_times else None

    return {
        "tts_all":                  env.ep_tts,
        "twt_all":                  env.ep_twt,
        "min_speed":                env.ep_min_speed,
        "queue_violation":          env.ep_queue_violation,
        "Reward":                   -env.ep_tts,
        "Mean computation time [s]": mean_time,
        "Max computation time [s]":  max_time,
        # Trajectories for plotting
        "rho_log":   np.array(rho_log),
        "v_log":     np.array(v_log),
        "w_log":     np.array(w_log),
        "action_log": np.array(action_log),
    }


# ======================================================================
# Run experiments (all 6 scenarios) for a given controller class
# ======================================================================

def run_experiments(
    controller_cls,
    controller_kwargs: dict,
    env_params: dict,
    label: str = "Controller",
) -> pd.DataFrame:
    """
    Evaluate one controller on all 6 scenarios: Demand 1/2 × Low/Med/High noise.

    Parameters
    ----------
    controller_cls : type
        Controller subclass (e.g. NoControlController, MPCController).
    controller_kwargs : dict
        Extra keyword arguments passed to the controller constructor.
        For MPC this includes mpc_params, w_tts, w_du, etc.
    env_params : dict
        Physical parameters for the environment (e.g. METANET_PARAMS["real"]).
        The env always uses these "real" parameters for simulation.
    label : str
        Display name for print / logging.
    """
    results = []

    demands = {
        "Demand 1": create_demand_1(),
        "Demand 2": create_demand_2(),
    }
    noise_levels = ["Low", "Medium", "High"]

    print(f"\n{'='*60}")
    print(f"STARTING {label.upper()} EXPERIMENTS")
    print(f"{'='*60}\n")

    for demand_name, demand_data in demands.items():
        for noise_name in noise_levels:
            print(f"\n--- {demand_name} | {noise_name} Noise ---")

            env = MetanetEnv(
                metanet_param=env_params,
                demand=demand_data,
                noise_cfg=GUSSIAN_NOISE[noise_name],
                seed=SEED,
            )

            # Build controller
            if controller_cls is NoControlController:
                ctrl = controller_cls(v_free=env_params["v_free_1"])
            elif controller_cls is MPCController:
                ctrl = controller_cls(env=env, demand=demand_data, **controller_kwargs)
            elif controller_cls is DDPGController:
                tag = f"{demand_name}_{noise_name}"
                ctrl = controller_cls(
                    demand=demand_data, model_tag=tag, **controller_kwargs
                )
            elif controller_cls is MPCDRLController:
                tag = f"{demand_name}_{noise_name}"
                ctrl = controller_cls(
                    env=env, demand=demand_data, model_tag=tag, **controller_kwargs
                )
            else:
                # Generic fallback – pass env + demand + kwargs
                ctrl = controller_cls(env=env, demand=demand_data, **controller_kwargs)

            desc = f"[{label} | {demand_name} | {noise_name} Noise]"
            metrics = simulate(env, ctrl, desc=desc)

            results.append({
                "Scenario":                  demand_name,
                "Noise":                     noise_name,
                "TTS (veh·h)":               metrics["tts_all"],
                "TWT (veh·h)":               metrics["twt_all"],
                "Min Speed (km/h)":          metrics["min_speed"],
                "Queue Violation":           metrics["queue_violation"],
                "Total Reward":              metrics["Reward"],
                "Mean computation time [s]": metrics["Mean computation time [s]"],
                "Max computation time [s]":  metrics["Max computation time [s]"],
            })

    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"{label.upper()} EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False, float_format="%.2f"))
    print("=" * 80 + "\n")
    return df


# ======================================================================
# Plotting utilities
# ======================================================================

def plot_results(metrics: dict, title: str = "", save_path: str | None = None):
    """
    Plot density heatmap, speed heatmap, queue lengths, and control inputs.
    """
    time_axis = np.arange(len(metrics["rho_log"])) * TS  # seconds

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Density heatmap ---
    ax = axes[0, 0]
    im = ax.imshow(
        metrics["rho_log"].T, aspect="auto", origin="lower",
        extent=[0, time_axis[-1], 0.5, metrics["rho_log"].shape[1] + 0.5],
        cmap="YlOrRd",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Segment")
    ax.set_title("Density [veh/km/lane]")
    fig.colorbar(im, ax=ax)

    # --- Speed heatmap ---
    ax = axes[0, 1]
    im = ax.imshow(
        metrics["v_log"].T, aspect="auto", origin="lower",
        extent=[0, time_axis[-1], 0.5, metrics["v_log"].shape[1] + 0.5],
        cmap="RdYlGn",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Segment")
    ax.set_title("Speed [km/h]")
    fig.colorbar(im, ax=ax)

    # --- Queue lengths ---
    ax = axes[1, 0]
    w_log = metrics["w_log"]
    ax.plot(time_axis, w_log[:, 0], label="Mainstream origin", linewidth=1.2)
    if w_log.shape[1] > 1:
        ax.plot(time_axis, w_log[:, 1], label="On-ramp origin", linewidth=1.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Queue [veh]")
    ax.set_title("Queue Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Control inputs ---
    ax = axes[1, 1]
    a_log = metrics["action_log"]
    ax.plot(time_axis, a_log[:, 0], label="VSL1 [km/h]", linewidth=1.2)
    ax.plot(time_axis, a_log[:, 1], label="VSL2 [km/h]", linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(time_axis, a_log[:, 2], label="Ramp rate", color="tab:green",
             linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("VSL [km/h]")
    ax2.set_ylabel("Ramp metering rate")
    ax.set_title("Control Inputs")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
    return fig


# ======================================================================
# Comparison table: controllers as columns, scenarios × metrics as rows
# ======================================================================

def print_comparison_table(controller_results: dict[str, pd.DataFrame]):
    """
    Print a pivot-style table where:
      Rows   = (Scenario, Noise) × Metric
      Columns = Controller names

    Parameters
    ----------
    controller_results : dict
        Mapping from controller label to its results DataFrame.
    """
    metrics_cols = [
        "TTS (veh·h)", "TWT (veh·h)", "Min Speed (km/h)",
        "Queue Violation", "Mean computation time [s]", "Max computation time [s]",
    ]
    ctrl_names = list(controller_results.keys())

    # Collect all scenarios
    first_df = next(iter(controller_results.values()))
    scenarios = list(zip(first_df["Scenario"], first_df["Noise"]))

    # Build header
    ctrl_header = "  ".join(f"{c:>18s}" for c in ctrl_names)
    header = f"{'Scenario':<22s} {'Metric':<30s} {ctrl_header}"
    sep = "=" * len(header)

    print(f"\n\n{sep}")
    print("COMBINED RESULTS COMPARISON")
    print(sep)
    print(header)
    print("-" * len(header))

    for scenario, noise in scenarios:
        print(f"\n{scenario}, {noise} Noise")
        print("-" * len(header))

        for metric in metrics_cols:
            values = []
            for ctrl_name in ctrl_names:
                df = controller_results[ctrl_name]
                row = df[(df["Scenario"] == scenario) & (df["Noise"] == noise)]
                if len(row) > 0:
                    val = row[metric].values[0]
                    if val is None:
                        values.append(f"{'—':>18s}")
                    else:
                        values.append(f"{val:>18.2f}")
                else:
                    values.append(f"{'—':>18s}")

            val_str = "  ".join(values)
            print(f"{'':22s} {metric:<30s} {val_str}")

    print(sep + "\n")


# ======================================================================
# Main entry point
# ======================================================================

if __name__ == "__main__":
    import os

    env_params = METANET_PARAMS["real"]        # env always uses real params
    mpc_params = METANET_PARAMS["estimated"]   # MPC uses estimated params (model mismatch)

    # ---- No Control ----
    df_nc = run_experiments(
        NoControlController, {},
        env_params=env_params, label="No Control",
    )
    df_nc.to_csv("../2_data_produced/results/no_control_results.csv", index=False)

    # ---- Standalone MPC ----
    df_mpc = run_experiments(
        MPCController,
        {"mpc_params": mpc_params, "w_tts": 1.0, "w_du": 0.4, "num_starts": N_MULTI_START, "verbose": True},
        env_params=env_params, label="Standalone MPC",
    )
    df_mpc.to_csv("../2_data_produced/results/mpc_results.csv", index=False)

    # ---- High-Frequency MPC (Tc = 60 s) ----
    df_hfmpc = run_experiments(
        MPCController,
        {
            "mpc_params": mpc_params,
            "Nc": NC_HF,
            "M_step": M_STEP_HF,
            "w_tts": 1.0,
            "w_du": 0.4,
            "num_starts": N_MULTI_START,
            "verbose": True,
        },
        env_params=env_params, label="HF-MPC",
    )
    df_hfmpc.to_csv("../2_data_produced/results/hf_mpc_results.csv", index=False)

    # ---- Standalone DRL (1-step TD) ----
    model_dir_1 = os.path.join("..", "2_data_produced", "drl_models", "ddpg_1step_td")
    if os.path.isdir(model_dir_1):
        df_drl_1 = run_experiments(
            DDPGController,
            {"n_step": 1, "training": False, "model_dir": model_dir_1},
            env_params=env_params, label="DDPG 1-step TD",
        )
        df_drl_1.to_csv("../2_data_produced/results/drl_1step_results.csv", index=False)
    else:
        df_drl_1 = None
        print(f"[SKIP] DDPG 1-step TD models not found in {model_dir_1}")
        print(f"       Run: python train_drl.py --n_step 1")

    # ---- Standalone DRL (10-step TD) ----
    model_dir_10 = os.path.join("..", "2_data_produced", "drl_models", "ddpg_10step_td")
    if os.path.isdir(model_dir_10):
        df_drl_10 = run_experiments(
            DDPGController,
            {"n_step": 10, "training": False, "model_dir": model_dir_10},
            env_params=env_params, label="DDPG 10-step TD",
        )
        df_drl_10.to_csv("../2_data_produced/results/drl_10step_results.csv", index=False)
    else:
        df_drl_10 = None
        print(f"[SKIP] DDPG 10-step TD models not found in {model_dir_10}")
        print(f"       Run: python train_drl.py --n_step 10")

    # ---- Combined MPC-DRL (1-step TD) ----
    mpc_drl_dir_1 = os.path.join("..", "2_data_produced", "mpc_drl_models", "mpc_drl_1step_td")
    if os.path.isdir(mpc_drl_dir_1):
        df_mpc_drl_1 = run_experiments(
            MPCDRLController,
            {
                "n_step": 1,
                "model_dir": mpc_drl_dir_1,
                "mpc_params": mpc_params,
                "w_tts": 1.0,
                "w_du": 0.4,
                "verbose": True,
            },
            env_params=env_params, label="MPC-DRL 1-step TD",
        )
        df_mpc_drl_1.to_csv("../2_data_produced/results/mpc_drl_1step_results.csv", index=False)
    else:
        df_mpc_drl_1 = None
        print(f"[SKIP] MPC-DRL 1-step TD models not found in {mpc_drl_dir_1}")
        print(f"       Run: python train_mpc_drl.py --n_step 1")

    # ---- Combined MPC-DRL (10-step TD) ----
    mpc_drl_dir_10 = os.path.join("..", "2_data_produced", "mpc_drl_models", "mpc_drl_10step_td")
    if os.path.isdir(mpc_drl_dir_10):
        df_mpc_drl_10 = run_experiments(
            MPCDRLController,
            {
                "n_step": 10,
                "model_dir": mpc_drl_dir_10,
                "mpc_params": mpc_params,
                "w_tts": 1.0,
                "w_du": 0.4,
                "verbose": True,
            },
            env_params=env_params, label="MPC-DRL 10-step TD",
        )
        df_mpc_drl_10.to_csv("../2_data_produced/results/mpc_drl_10step_results.csv", index=False)
    else:
        df_mpc_drl_10 = None
        print(f"[SKIP] MPC-DRL 10-step TD models not found in {mpc_drl_dir_10}")
        print(f"       Run: python train_mpc_drl.py --n_step 10")

    # ---- Print combined summary (pivot table: controllers as columns) ----

    comparison = {}

    comparison["No Control"] = df_nc
    comparison["Standalone MPC"] = df_mpc
    if df_drl_10 is not None:
        comparison["Standalone DRL (10-step TD)"] = df_drl_10
    comparison["HF MPC"] = df_hfmpc
    if df_mpc_drl_10 is not None:
        comparison["MPC-DRL (10-step TD)"] = df_mpc_drl_10
    if df_drl_1 is not None:
        comparison["Standalone DRL (1-step TD)"] = df_drl_1        
    if df_mpc_drl_1 is not None:
        comparison["MPC-DRL (1-step TD)"] = df_mpc_drl_1


    print_comparison_table(comparison)
